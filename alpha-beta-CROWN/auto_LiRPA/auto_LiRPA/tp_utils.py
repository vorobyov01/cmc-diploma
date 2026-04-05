"""Tensor Parallelism utilities for auto_LiRPA.

Provides tp_shard_bounded_module() for post-hoc sharding of a BoundedModule:
after constructing a BoundedModule from a regular (single-GPU) model, this
function walks the bound graph and converts BoundLinear nodes into
BoundLinearTP_Col / BoundLinearTP_Row with appropriately sharded weights.

This eliminates the need to write models with ColumnParallelLinear /
RowParallelLinear and custom ONNX ops — any standard nn.Module with
nn.Linear layers can be automatically sharded for TP verification.
"""
from collections import deque
from typing import List, TYPE_CHECKING

import torch
import torch.nn as nn

from .operators.linear import BoundLinear
from .operators.tensor_parallel import BoundLinearTP_Col, BoundLinearTP_Row
from .operators.leaf import BoundParams

if TYPE_CHECKING:
    from .bound_general import BoundedModule


def _topological_order(model: 'BoundedModule') -> list:
    """Return bound nodes in topological order (Kahn's algorithm)."""
    in_degree: dict[str, int] = {}
    for node in model.nodes():
        in_degree[node.name] = len(node.inputs)

    queue: deque[str] = deque()
    for name in model.root_names:
        if in_degree.get(name, 0) == 0:
            queue.append(name)

    order = []
    while queue:
        name = queue.popleft()
        node = model[name]
        order.append(node)
        for next_name in node.output_name:
            in_degree[next_name] -= 1
            if in_degree[next_name] == 0:
                queue.append(next_name)
    return order


def _shard_param(param_node: BoundParams, dim: int,
                 rank: int, world_size: int) -> None:
    """Shard a BoundParams parameter in-place along the given dimension."""
    w = param_node.param.data
    size = w.shape[dim]
    assert size % world_size == 0, (
        f"Dimension {dim} of param {param_node.name} (size={size}) "
        f"is not divisible by world_size={world_size}")
    local_size = size // world_size
    start = rank * local_size
    slices = [slice(None)] * w.ndim
    slices[dim] = slice(start, start + local_size)
    param_node.param = nn.Parameter(
        w[tuple(slices)].contiguous().clone(),
        requires_grad=param_node.param.requires_grad,
    )


def tp_shard_bounded_module(
    model: 'BoundedModule',
    world_size: int,
    rank: int,
) -> List[str]:
    """Replace BoundLinear nodes with TP-sharded equivalents in-place.

    Uses alternating Column/Row parallelism (Megatron-LM pattern):
      - Odd layers (1st, 3rd, ...): ColumnParallel — shard output dim
      - Even layers (2nd, 4th, ...): RowParallel — shard input dim

    Args:
        model: A BoundedModule built from a regular (non-TP) model.
        world_size: Total number of TP ranks.
        rank: This process's rank.

    Returns:
        List of sharded node names (for debugging/logging).
    """
    if world_size <= 1:
        return []

    topo = _topological_order(model)
    linears = [n for n in topo if type(n) is BoundLinear]

    if not linears:
        return []

    sharded_names: List[str] = []

    for idx, node in enumerate(linears):
        is_col = (idx % 2 == 0)

        weight_node = node.inputs[1]
        assert isinstance(weight_node, BoundParams), (
            f"Expected BoundParams for weight of {node.name}, "
            f"got {type(weight_node).__name__}")
        bias_node = node.inputs[2] if len(node.inputs) > 2 else None

        if is_col:
            # Column Parallel: shard along output features.
            # transB=1 → weight shape [out, in] → shard dim 0
            # transB=0 → weight shape [in, out] → shard dim 1
            shard_dim = 0 if node.transB else 1
            _shard_param(weight_node, shard_dim, rank, world_size)

            if bias_node is not None and isinstance(bias_node, BoundParams):
                _shard_param(bias_node, 0, rank, world_size)

            node.__class__ = BoundLinearTP_Col
        else:
            # Row Parallel: shard along input features.
            # transB=1 → weight shape [out, in] → shard dim 1
            # transB=0 → weight shape [in, out] → shard dim 0
            shard_dim = 1 if node.transB else 0
            _shard_param(weight_node, shard_dim, rank, world_size)

            # Row parallel: bias stays full (AllReduce aggregates partial sums)

            node.__class__ = BoundLinearTP_Row

        node._refresh_dist_state()
        sharded_names.append(node.name)

    return sharded_names
