# cmc-diploma

```bash
git clone https://github.com/vorobyov01/cmc-diploma.git
cd cmc-diploma
uv venv
source .venv/bin/activate
uv pip install torch torchvision
cd alpha-beta-CROWN/auto_LiRPA
uv pip install -e .
python3 examples/simple/toy.py
torchrun --nproc_per_node=2 examples/simple/test_tp_verification.py
```