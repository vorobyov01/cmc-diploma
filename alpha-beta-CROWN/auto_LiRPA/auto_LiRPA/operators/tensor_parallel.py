"""
Tensor Parallelism support for auto_LiRPA bound computation.

This module implements distributed bound propagation using Tensor Parallelism,
inspired by Megatron-LM architecture but adapted for backward bound propagation (CROWN).

NOTE: This is a first version/prototype. For full TP support, the model layers
need to be properly sharded and the bound computation needs to handle sharded
weights and activations correctly.
"""
import torch
import torch.distributed as dist
from .linear import BoundLinear


class BoundLinearTP_Col(BoundLinear):
    """
    Column Parallel Linear Layer for Tensor Parallelism.
    
    In forward pass: weights are split along output dimension (columns).
    In backward CROWN: incoming A matrices are split, requires AllReduce to combine.
    
    This is used for layers that expand dimension (e.g., MLP expansion layers).
    """
    
    def __init__(self, attr=None, inputs=None, output_index=0, options=None):
        super().__init__(attr, inputs, output_index, options)
        # Check if distributed is initialized (optional for first version)
        if dist.is_initialized():
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
            self.use_tp = True
        else:
            self.use_tp = False
            self.world_size = 1
            self.rank = 0
        
    def bound_backward(self, last_lA, last_uA, *x, start_node=None,
                       reduce_bias=True, **kwargs):
        """
        Backward bound propagation with Tensor Parallelism.
        
        For Column Parallel layers:
        - last_lA/last_uA are split along output dimension (already sharded)
        - We compute partial products: partial_A = last_A @ W_local
        - Then AllReduce to get full A for previous layer
        """
        # Call parent to handle most of the logic
        result = super().bound_backward(last_lA, last_uA, *x, start_node=start_node,
                                       reduce_bias=reduce_bias, **kwargs)
        
        # If TP is enabled, perform AllReduce on A matrices and biases
        if self.use_tp and self.world_size > 1:
            # Extract A matrices for input (x[0])
            lA_x, uA_x = result[0][0]
            lbias, ubias = result[1], result[2]
            
            # AllReduce to combine partial results from all GPUs
            if lA_x is not None and isinstance(lA_x, torch.Tensor):
                dist.all_reduce(lA_x, op=dist.ReduceOp.SUM, async_op=False)
            if uA_x is not None and isinstance(uA_x, torch.Tensor):
                dist.all_reduce(uA_x, op=dist.ReduceOp.SUM, async_op=False)
            if isinstance(lbias, torch.Tensor):
                dist.all_reduce(lbias, op=dist.ReduceOp.SUM, async_op=False)
            if isinstance(ubias, torch.Tensor):
                dist.all_reduce(ubias, op=dist.ReduceOp.SUM, async_op=False)
        
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
        if dist.is_initialized():
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
            self.use_tp = True
        else:
            self.use_tp = False
            self.world_size = 1
            self.rank = 0
        
    def bound_backward(self, last_lA, last_uA, *x, start_node=None,
                       reduce_bias=True, **kwargs):
        """
        Backward bound propagation with Tensor Parallelism.
        
        For Row Parallel layers:
        - last_lA/last_uA are replicated (full matrices on all GPUs)
        - We compute: A_prev_local = last_A @ W_local
        - Result is automatically split (no communication needed!)
        """
        # For Row Parallel, the incoming A matrices are replicated
        # We just compute locally and the result is automatically sharded
        # No AllReduce needed!
        result = super().bound_backward(last_lA, last_uA, *x, start_node=start_node,
                                       reduce_bias=reduce_bias, **kwargs)
        
        # The result A matrices are already correctly sharded
        # No communication needed for Row Parallel layers
        return result


