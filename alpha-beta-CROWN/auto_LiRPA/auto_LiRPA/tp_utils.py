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

    Uses paired Column/Row parallelism (Megatron-LM pattern):
      - Pairs consecutive Linear layers as (Col, Row)
      - Col: shard output dim; Row: shard input dim
      - The last layer is left un-sharded if there's an odd number
        (output layer is typically small and sharding it would break
        the C matrix / output specification)

    Args:
        model: A BoundedModule built from a regular (non-TP) model.
        world_size: Total number of TP ranks.
        rank: This process's rank.

    Returns:
        List of sharded node names (for debugging/logging).
    """
    if world_size <= 1:
        return []

    model._tp_active_cached = True

    topo = _topological_order(model)
    linears = [n for n in topo if type(n) is BoundLinear]

    if not linears:
        return []

    # Pair consecutive linears as (Col, Row). Leave last un-sharded if odd.
    n_pairs = len(linears) // 2
    sharded_names: List[str] = []

    for pair_idx in range(n_pairs):
        col_node = linears[pair_idx * 2]
        row_node = linears[pair_idx * 2 + 1]

        # --- Column Parallel ---
        col_weight = col_node.inputs[1]
        assert isinstance(col_weight, BoundParams), (
            f"Expected BoundParams for weight of {col_node.name}, "
            f"got {type(col_weight).__name__}")
        col_bias = col_node.inputs[2] if len(col_node.inputs) > 2 else None

        shard_dim = 0 if col_node.transB else 1
        _shard_param(col_weight, shard_dim, rank, world_size)
        if col_bias is not None and isinstance(col_bias, BoundParams):
            _shard_param(col_bias, 0, rank, world_size)

        col_node.__class__ = BoundLinearTP_Col
        col_node._refresh_dist_state()
        sharded_names.append(col_node.name)

        # --- Row Parallel ---
        row_weight = row_node.inputs[1]
        assert isinstance(row_weight, BoundParams), (
            f"Expected BoundParams for weight of {row_node.name}, "
            f"got {type(row_weight).__name__}")

        shard_dim = 1 if row_node.transB else 0
        _shard_param(row_weight, shard_dim, rank, world_size)

        row_node.__class__ = BoundLinearTP_Row
        row_node._refresh_dist_state()
        sharded_names.append(row_node.name)

    return sharded_names
