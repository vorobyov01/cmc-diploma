"""
Tensor Parallelism support for auto_LiRPA bound computation.

This module implements distributed bound propagation using Tensor Parallelism,
inspired by Megatron-LM architecture but adapted for backward bound propagation (CROWN).

NOTE: This is a first version/prototype. For full TP support, the model layers
need to be properly sharded and the bound computation needs to handle sharded
weights and activations correctly.
"""
import os
import torch
import torch.distributed as dist
from .linear import BoundLinear

_TP_DEBUG = os.environ.get('TP_DEBUG', '0') == '1'

def _tp_log(msg):
    if _TP_DEBUG and dist.is_initialized():
        rank = dist.get_rank()
        import sys
        print(f"[TP rank{rank}] {msg}", flush=True, file=sys.stderr)


class BoundLinearTP_Col(BoundLinear):
    """
    Column Parallel Linear Layer for Tensor Parallelism.
    
    In forward pass: weights are split along output dimension (columns).
    In backward CROWN: incoming A matrices are split, requires AllReduce to combine.
    
    This is used for layers that expand dimension (e.g., MLP expansion layers).
    """
    
    def __init__(self, attr=None, inputs=None, output_index=0, options=None):
        super().__init__(attr, inputs, output_index, options)
        # For custom TP ops, weights are provided in (out_features, in_features).
        # BoundLinear expects transB=1 for this layout.
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

    @staticmethod
    def _all_reduce_inplace(value):
        if isinstance(value, torch.Tensor):
            dist.all_reduce(value, op=dist.ReduceOp.SUM, async_op=False)
        elif isinstance(value, (tuple, list)):
            for item in value:
                if isinstance(item, torch.Tensor):
                    dist.all_reduce(item, op=dist.ReduceOp.SUM, async_op=False)

    def forward(self, x, w, b=None):
        self._refresh_dist_state()
        _tp_log("Col.forward enter")
        result = super().forward(x, w, b)
        _tp_log("Col.forward exit")
        return result

    def bound_backward(self, last_lA, last_uA, *x, start_node=None,
                       reduce_bias=True, **kwargs):
        """
        Backward bound propagation with Tensor Parallelism.

        For Column Parallel layers:
        - last_lA/last_uA are split along output dimension (already sharded)
        - We compute partial products: partial_A = last_A @ W_local
        - Then AllReduce to get full A for previous layer
        """
        _tp_log(f"Col.bound_backward enter last_lA={type(last_lA).__name__} last_uA={type(last_uA).__name__}")
        # Call parent to handle most of the logic
        result = super().bound_backward(last_lA, last_uA, *x, start_node=start_node,
                                       reduce_bias=reduce_bias, **kwargs)

        self._refresh_dist_state()

        # If TP is enabled, perform AllReduce on A matrices and biases
        if self.use_tp and self.world_size > 1:
            # Extract A matrices for input (x[0])
            lA_x, uA_x = result[0][0]
            lbias, ubias = result[1], result[2]

            _tp_log(f"Col.bound_backward lA_x={lA_x is not None} uA_x={uA_x is not None} "
                    f"lbias_type={type(lbias).__name__} ubias_type={type(ubias).__name__}")

            # AllReduce to combine partial results from all GPUs
            if lA_x is not None and isinstance(lA_x, torch.Tensor):
                _tp_log(f"Col.bound_backward all_reduce lA_x shape={lA_x.shape}")
                dist.all_reduce(lA_x, op=dist.ReduceOp.SUM, async_op=False)
            if uA_x is not None and isinstance(uA_x, torch.Tensor):
                _tp_log(f"Col.bound_backward all_reduce uA_x shape={uA_x.shape}")
                dist.all_reduce(uA_x, op=dist.ReduceOp.SUM, async_op=False)
            _tp_log(f"Col.bound_backward all_reduce lbias")
            self._all_reduce_inplace(lbias)
            _tp_log(f"Col.bound_backward all_reduce ubias")
            self._all_reduce_inplace(ubias)

        _tp_log("Col.bound_backward exit")
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
        _tp_log("Row.forward enter")
        output = super().forward(x, w, b)
        if self.use_tp and self.world_size > 1 and isinstance(output, torch.Tensor):
            _tp_log(f"Row.forward all_reduce shape={output.shape}")
            dist.all_reduce(output, op=dist.ReduceOp.SUM, async_op=False)
        _tp_log("Row.forward exit")
        return output

    def interval_propagate(self, *v, C=None, w=None):
        self._refresh_dist_state()
        _tp_log("Row.interval_propagate enter")
        lower, upper = super().interval_propagate(*v, C=C, w=w)
        if self.use_tp and self.world_size > 1:
            if isinstance(lower, torch.Tensor):
                _tp_log(f"Row.interval_propagate all_reduce lower shape={lower.shape}")
                dist.all_reduce(lower, op=dist.ReduceOp.SUM, async_op=False)
            if isinstance(upper, torch.Tensor):
                _tp_log(f"Row.interval_propagate all_reduce upper shape={upper.shape}")
                dist.all_reduce(upper, op=dist.ReduceOp.SUM, async_op=False)
        _tp_log("Row.interval_propagate exit")
        return lower, upper

    def bound_backward(self, last_lA, last_uA, *x, start_node=None,
                       reduce_bias=True, **kwargs):
        """
        Backward bound propagation with Tensor Parallelism.

        For Row Parallel layers:
        - last_lA/last_uA are replicated (full matrices on all GPUs)
        - We compute: A_prev_local = last_A @ W_local
        - Result is automatically split (no communication needed!)
        """
        _tp_log(f"Row.bound_backward enter last_lA={type(last_lA).__name__} last_uA={type(last_uA).__name__}")
        self._refresh_dist_state()
        result = super().bound_backward(last_lA, last_uA, *x, start_node=start_node,
                                        reduce_bias=reduce_bias, **kwargs)
        _tp_log("Row.bound_backward exit (no all_reduce)")
        return result


