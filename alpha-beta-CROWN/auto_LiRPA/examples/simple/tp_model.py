"""
Simple Tensor Parallel model for testing distributed bound computation.
"""
import torch
import torch.nn as nn
import torch.distributed as dist


class ColumnParallelLinear(nn.Module):
    """
    Column Parallel Linear layer - splits weights along output dimension.
    """
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        
        # Split output dimension
        assert out_features % self.world_size == 0, \
            f"out_features ({out_features}) must be divisible by world_size ({self.world_size})"
        self.local_out_features = out_features // self.world_size
        
        # Create local weight (sharded)
        self.weight = nn.Parameter(
            torch.randn(self.local_out_features, in_features)
        )
        if bias:
            self.bias = nn.Parameter(torch.randn(self.local_out_features))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x):
        # Local computation
        output = x.matmul(self.weight.t())
        if self.bias is not None:
            output = output + self.bias
        
        # AllGather to get full output (for forward pass)
        if dist.is_initialized() and self.world_size > 1:
            output_list = [torch.zeros_like(output) for _ in range(self.world_size)]
            dist.all_gather(output_list, output)
            output = torch.cat(output_list, dim=-1)
        
        return output


class RowParallelLinear(nn.Module):
    """
    Row Parallel Linear layer - splits weights along input dimension.
    """
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        
        # Split input dimension
        assert in_features % self.world_size == 0, \
            f"in_features ({in_features}) must be divisible by world_size ({self.world_size})"
        self.local_in_features = in_features // self.world_size
        
        # Create local weight (sharded)
        self.weight = nn.Parameter(
            torch.randn(out_features, self.local_in_features)
        )
        if bias:
            # Bias is replicated (not sharded) for Row Parallel
            self.bias = nn.Parameter(torch.randn(out_features))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x):
        # Split input along last dimension
        if dist.is_initialized() and self.world_size > 1:
            # x should already be split, but we ensure it
            local_x = x[..., self.rank * self.local_in_features:(self.rank + 1) * self.local_in_features]
        else:
            local_x = x
        
        # Local computation
        output = local_x.matmul(self.weight.t())
        
        # AllReduce to combine partial results
        if dist.is_initialized() and self.world_size > 1:
            dist.all_reduce(output, op=dist.ReduceOp.SUM)
        
        if self.bias is not None:
            output = output + self.bias
        
        return output


class SimpleTPModel(nn.Module):
    """
    Simple 2-layer MLP with Tensor Parallelism.
    Architecture: Input -> ColumnParallel (expand) -> ReLU -> RowParallel (compress) -> Output
    """
    def __init__(self, input_dim=2, hidden_dim=4, output_dim=1):
        super().__init__()
        # Column Parallel: expands dimension
        self.layer1 = ColumnParallelLinear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        # Row Parallel: compresses dimension
        self.layer2 = RowParallelLinear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x


