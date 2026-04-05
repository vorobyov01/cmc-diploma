"""
Tensor Parallelism support for auto_LiRPA bound computation.

This module implements distributed bound propagation using Tensor Parallelism,
inspired by Megatron-LM architecture but adapted for backward bound propagation
(CROWN / α-CROWN).

Key design:
  - AllReduce is implemented via DifferentiableAllReduce (torch.autograd.Function)
    so that α-CROWN gradient optimization works correctly through the TP layer.
  - Forward of AllReduce = SUM across ranks.
  - Backward of AllReduce = identity (∂(Σ x_r)/∂x_r = 1).
"""
import torch
import torch.distributed as dist
from .linear import BoundLinear


class DifferentiableAllReduce(torch.autograd.Function):
    """AllReduce that preserves the autograd graph.

    Forward: out = AllReduce(input)  — SUM across all ranks.
    Backward: grad passes through unchanged (identity).

    This is correct because AllReduce(x) = Σ_r x_r, so ∂/∂x_r = 1.
    Each rank already holds the correct local partial gradient.
    """

    @staticmethod
    def forward(ctx, tensor):
        output = tensor.contiguous().clone()
        dist.all_reduce(output, op=dist.ReduceOp.SUM, async_op=False)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def _tp_all_reduce(value):
    """Apply differentiable AllReduce to a tensor, tuple, or list of tensors."""
    if isinstance(value, torch.Tensor):
        return DifferentiableAllReduce.apply(value.contiguous())
    if isinstance(value, (tuple, list)):
        items = []
        for item in value:
            if isinstance(item, torch.Tensor):
                item = DifferentiableAllReduce.apply(item.contiguous())
            items.append(item)
        return type(value)(items)
    return value


class BoundLinearTP_Col(BoundLinear):
    """
    Column Parallel Linear Layer for Tensor Parallelism.

    In forward pass: weights are split along output dimension (columns).
    In backward CROWN: incoming A matrices are split, requires AllReduce to combine.

    This is used for layers that expand dimension (e.g., MLP expansion layers).
    """

    def __init__(self, attr=None, inputs=None, output_index=0, options=None):
        super().__init__(attr, inputs, output_index, options)
        if attr is None or 'transB' not in attr:
            self.transB = 1
        self._refresh_dist_state()

    def _refresh_dist_state(self):
        if dist.is_available() and dist.is_initialized():
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
            self.use_tp = True
        else:
            self.use_tp = False
            self.world_size = 1
            self.rank = 0

    def forward(self, x, w, b=None):
        self._refresh_dist_state()
        return super().forward(x, w, b)

    def bound_backward(self, last_lA, last_uA, *x, start_node=None,
                       reduce_bias=True, **kwargs):
        result = super().bound_backward(last_lA, last_uA, *x, start_node=start_node,
                                       reduce_bias=reduce_bias, **kwargs)

        self._refresh_dist_state()

        if self.use_tp and self.world_size > 1:
            lA_x, uA_x = result[0][0]
            lbias, ubias = result[1], result[2]

            if lA_x is not None and isinstance(lA_x, torch.Tensor):
                lA_x = DifferentiableAllReduce.apply(lA_x.contiguous())
            if uA_x is not None and isinstance(uA_x, torch.Tensor):
                uA_x = DifferentiableAllReduce.apply(uA_x.contiguous())

            result[0][0] = (lA_x, uA_x)
            lbias = _tp_all_reduce(lbias)
            ubias = _tp_all_reduce(ubias)
            result = (result[0], lbias, ubias)

        return result


class BoundLinearTP_Row(BoundLinear):
    """
    Row Parallel Linear Layer for Tensor Parallelism.

    In forward pass: weights are split along input dimension (rows).
    In backward CROWN: incoming A matrices are replicated, output A is automatically split.

    This is used for layers that compress dimension (e.g., MLP compression layers).
    """

    def __init__(self, attr=None, inputs=None, output_index=0, options=None):
        super().__init__(attr, inputs, output_index, options)
        if attr is None or 'transB' not in attr:
            self.transB = 1
        self._refresh_dist_state()

    def _refresh_dist_state(self):
        if dist.is_available() and dist.is_initialized():
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
            self.use_tp = True
        else:
            self.use_tp = False
            self.world_size = 1
            self.rank = 0

    def forward(self, x, w, b=None):
        self._refresh_dist_state()
        output = super().forward(x, w, b)
        if self.use_tp and self.world_size > 1 and isinstance(output, torch.Tensor):
            output = DifferentiableAllReduce.apply(output.contiguous())
        return output

    def interval_propagate(self, *v, C=None, w=None):
        self._refresh_dist_state()
        lower, upper = super().interval_propagate(*v, C=C, w=w)
        if self.use_tp and self.world_size > 1:
            if isinstance(lower, torch.Tensor):
                lower = DifferentiableAllReduce.apply(lower.contiguous())
            if isinstance(upper, torch.Tensor):
                upper = DifferentiableAllReduce.apply(upper.contiguous())
        return lower, upper

    def bound_backward(self, last_lA, last_uA, *x, start_node=None,
                       reduce_bias=True, **kwargs):
        self._refresh_dist_state()
        result = super().bound_backward(last_lA, last_uA, *x, start_node=start_node,
                                        reduce_bias=reduce_bias, **kwargs)
        return result
