"""Verify TP=N CROWN bounds match single-GPU CROWN bounds.

Run: torchrun --nproc_per_node=2 verify_tp.py
"""
import os, sys, copy
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../auto_LiRPA"))

import torch
import torch.nn as nn
import torch.distributed as dist

from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
from auto_LiRPA.tp_utils import tp_shard_bounded_module


def main():
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    ws = int(os.environ.get("WORLD_SIZE", "1"))
    dist.init_process_group("nccl", rank=rank, world_size=ws)
    dev = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(dev)

    torch.manual_seed(42)

    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, 10),
    )
    model.to(dev).eval()

    x = torch.randn(1, 1, 28, 28, device=dev).clamp(0, 1)
    eps = 0.02
    x_L = (x - eps).clamp(0, 1)
    x_U = (x + eps).clamp(0, 1)

    # --- Single GPU (reference, no TP) ---
    dummy = torch.empty_like(x)
    lirpa_ref = BoundedModule(copy.deepcopy(model), dummy, device=dev)
    ptb_ref = PerturbationLpNorm(norm=float("inf"), x_L=x_L.clone(), x_U=x_U.clone())
    bx_ref = BoundedTensor(x.clone(), ptb_ref)
    lb_ref, ub_ref = lirpa_ref.compute_bounds(x=(bx_ref,), method="CROWN")

    # --- TP=N ---
    lirpa_tp = BoundedModule(copy.deepcopy(model), dummy, device=dev)
    tp_shard_bounded_module(lirpa_tp, ws, rank, dummy_input=dummy)
    ptb_tp = PerturbationLpNorm(norm=float("inf"), x_L=x_L.clone(), x_U=x_U.clone())
    bx_tp = BoundedTensor(x.clone(), ptb_tp)
    lb_tp, ub_tp = lirpa_tp.compute_bounds(x=(bx_tp,), method="CROWN")

    # --- Compare ---
    lb_diff = (lb_ref - lb_tp).abs().max().item()
    ub_diff = (ub_ref - ub_tp).abs().max().item()

    if rank == 0:
        print(f"Reference lb: {lb_ref.detach().cpu().tolist()}")
        print(f"TP={ws}    lb: {lb_tp.detach().cpu().tolist()}")
        print(f"Reference ub: {ub_ref.detach().cpu().tolist()}")
        print(f"TP={ws}    ub: {ub_tp.detach().cpu().tolist()}")
        print(f"Max |lb_diff|: {lb_diff:.8f}")
        print(f"Max |ub_diff|: {ub_diff:.8f}")

        tol = 1e-4
        if lb_diff < tol and ub_diff < tol:
            print(f"PASS: bounds match within tolerance {tol}")
        else:
            print(f"FAIL: bounds differ beyond tolerance {tol}")

    # Also check rank consistency
    lb_tp_gathered = [torch.zeros_like(lb_tp) for _ in range(ws)]
    ub_tp_gathered = [torch.zeros_like(ub_tp) for _ in range(ws)]
    dist.all_gather(lb_tp_gathered, lb_tp)
    dist.all_gather(ub_tp_gathered, ub_tp)

    if rank == 0:
        rank_lb_diff = (lb_tp_gathered[0] - lb_tp_gathered[1]).abs().max().item()
        rank_ub_diff = (ub_tp_gathered[0] - ub_tp_gathered[1]).abs().max().item()
        print(f"Cross-rank |lb_diff|: {rank_lb_diff:.8f}")
        print(f"Cross-rank |ub_diff|: {rank_ub_diff:.8f}")
        if rank_lb_diff < 1e-6 and rank_ub_diff < 1e-6:
            print("PASS: ranks produce identical bounds")
        else:
            print("FAIL: ranks produce different bounds")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
