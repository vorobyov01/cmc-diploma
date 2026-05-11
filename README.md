# cmc-diploma

```bash
git clone https://github.com/vorobyov01/cmc-diploma.git
cd cmc-diploma
uv sync                          # использует uv.lock (torch cu128 для CUDA 12.8)
source .venv/bin/activate
python3 alpha-beta-CROWN/auto_LiRPA/examples/simple/toy.py
NCCL_P2P_DISABLE=1 python -m torch.distributed.run --nproc_per_node=2 \
  alpha-beta-CROWN/auto_LiRPA/examples/simple/test_tp_verification.py
```