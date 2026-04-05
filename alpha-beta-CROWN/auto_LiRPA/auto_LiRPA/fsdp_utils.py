"""Fully-Sharded Data Parallelism (FSDP) utilities for auto_LiRPA verification.

FSDP shards each BoundLinear's weight across GPUs.  Before any operation
that needs the full weight (IBP forward, CROWN backward), the weight is
reconstructed via AllGather.  After compute_bounds finishes, gathered
weights can be freed to return to the sharded state.

Unlike Tensor Parallelism, FSDP introduces *no* accuracy loss -- the
computation is mathematically identical to single-GPU.
"""

import torch
import torch.distributed as dist
import torch.nn as nn

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .bound_general import BoundedModule


def fsdp_shard_bounded_module(model: 'BoundedModule', world_size: int,
                              rank: int, dummy_input: torch.Tensor = None):
    """Shard BoundLinear weights across *world_size* GPUs.

    After sharding, each BoundParams holding a weight matrix stores only
    1/world_size of the rows.  Calls to ``BoundParams.forward()`` will
    transparently AllGather the full weight when needed.

    Args:
        model: A ``BoundedModule`` already constructed with full weights.
        world_size: Total number of GPUs.
        rank: Index of the current GPU (0-based).
        dummy_input: A tensor with the correct input shape, used to
            refresh the computation graph shapes after sharding.
    """
    from .operators.linear import BoundLinear
    from .operators.leaf import BoundParams

    sharded_count = 0
    for node in model.nodes():
        if not isinstance(node, BoundLinear):
            continue
        if len(node.inputs) < 2:
            continue
        w_node = node.inputs[1]
        if not isinstance(w_node, BoundParams):
            continue

        W = w_node.param.data  # (out_features, in_features)
        if W.ndim < 2 or W.shape[0] % world_size != 0:
            continue

        chunk = W.shape[0] // world_size
        shard = W[rank * chunk : (rank + 1) * chunk].contiguous()

        w_node.param = nn.Parameter(shard, requires_grad=W.requires_grad)
        w_node._fsdp_world_size = world_size
        w_node._fsdp_rank = rank
        w_node._fsdp_shard_dim = 0

        # Invalidate cached values so they get re-computed via AllGather.
        w_node.forward_value = None

        sharded_count += 1

    if rank == 0:
        print(f"[FSDP] Sharded {sharded_count} weight matrices across "
              f"{world_size} GPUs")

    # Refresh graph shapes: a forward pass will trigger AllGather in each
    # BoundParams.forward(), producing correct full-sized forward_values.
    if dummy_input is not None:
        _refresh_graph_shapes(model, dummy_input)


def _refresh_graph_shapes(model: 'BoundedModule', dummy_input: torch.Tensor):
    """Run a forward pass to update output_shape / forward_value metadata."""
    with torch.no_grad():
        model.forward(dummy_input)


def fsdp_free_gathered_weights(model: 'BoundedModule'):
    """Free full-sized forward_values cached during compute_bounds.

    Call this between compute_bounds invocations to release GPU memory
    occupied by AllGathered weight copies.  The next compute_bounds will
    re-gather them on demand.
    """
    from .operators.leaf import BoundParams

    for node in model.nodes():
        if isinstance(node, BoundParams) and hasattr(node, '_fsdp_world_size'):
            node.forward_value = None
            node.interval = None
            node.lower = None
            node.upper = None
