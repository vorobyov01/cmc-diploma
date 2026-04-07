# AGENTS.md

## Диплом

**Тема:** Параллелизмы в верификации нейронных сетей

**Суть работы:** Формальная верификация нейронных сетей -- задача доказательства того, что сеть удовлетворяет заданным спецификациям (например, робастность к adversarial-атакам). Задача NP-полна для ReLU-сетей. Основной инструмент -- Alpha-Beta-CROWN (победитель VNN-COMP), использующий bound propagation на GPU. Цель диплома -- применить параллелизмы (Tensor Parallelism и др.) к верификации для масштабирования на более крупные сети и несколько GPU.

Все что касается текста диплома и презентации должно быть написано на русском.
Можно использовать общеизвестные термины на английском (например, LLM, Data Parallel, FSDP). В таком случае термины нужно расшифровать в обзоре литературы.

## Structure

- `./alpha-beta-CROWN` -- форк alpha-beta-CROWN с реализацией TP и FSDP верификации
- `./diploma-paper` -- текст диплома (LaTeX)

## GPU-виртуалка

Подключение:
```bash
ssh e5wf3lg2dxaoi6-64411394@ssh.runpod.io -i /Users/svorobyov/Documents/cmc_mmp/vastai
```

## Полезные команды

### Тесты на VM

```bash
# Подключение
ssh e5wf3lg2dxaoi6-64411394@ssh.runpod.io -i /Users/svorobyov/Documents/cmc_mmp/vastai

# Pull и активация окружения
cd /root/cmc-diploma && git pull && source .venv/bin/activate && cd alpha-beta-CROWN/auto_LiRPA

# Регрессионный тест (single GPU, без distributed)
python3 examples/simple/toy.py

# TP-тест на 2 GPU (обязательно через torchrun)
torchrun --nproc_per_node=2 examples/simple/test_tp_verification.py

# Если порт занят — сменить или убить старые процессы
torchrun --nproc_per_node=2 --master_port=29510 examples/simple/test_tp_verification.py
killall -9 python3 torchrun 2>/dev/null
```

### Эксперимент CROWN: OOM (1 GPU) vs TP=2

```bash
cd /root/cmc-diploma && git pull && source .venv/bin/activate
cd alpha-beta-CROWN/experiments/crown

# 1) Single GPU
python run.py --mode single --input-dim 4096 --hidden-dim 262144 --batch-size 2048

# 2) TP=2
torchrun --nproc_per_node=2 run.py --mode tp --input-dim 4096 --hidden-dim 262144 --batch-size 2048
```

### Эксперимент α-CROWN: численное сравнение single vs TP

```bash
cd /root/cmc-diploma && git pull && source .venv/bin/activate
cd alpha-beta-CROWN/experiments/alpha_crown

# 1) Single GPU — сохранить reference (веса + bounds + входы)
python run.py --mode single --method alpha-CROWN --save ref.pt

# 2) TP=1 — проверка корректности (bounds должны совпасть с reference)
torchrun --nproc_per_node=1 run.py --mode tp --method alpha-CROWN --compare ref.pt

# 3) TP=2 — те же веса, но шардинг на 2 GPU
torchrun --nproc_per_node=2 run.py --mode tp --method alpha-CROWN --compare ref.pt
```

### Эксперимент TP: корректность и звуковость (VNN-COMP модели)

```bash
cd /root/cmc-diploma && git pull && source .venv/bin/activate
cd alpha-beta-CROWN/experiments/vnncomp_tp

# Скачать ONNX-модели VNN-COMP MNIST-FC (если ещё не скачаны)
bash download_mnist_fc.sh

# Запуск: корректность TP=2 bounds vs single-GPU на ONNX моделях
torchrun --nproc_per_node=2 verify_tp.py
```

### Эксперимент FSDP: корректность bounds

```bash
cd /root/cmc-diploma && git pull && source .venv/bin/activate
cd alpha-beta-CROWN/experiments/fsdp_crown

# Проверка корректности: FSDP=2 bounds vs single-GPU
# Тестирует IBP + CROWN на MLP (256x2,4,6) и ONNX (256x2,4)
# Ожидание: все lb_diff = ub_diff = 0.00 (побитово идентичны)
torchrun --nproc_per_node=2 verify_fsdp.py
```

### Эксперимент FSDP: потребление GPU-памяти

```bash
cd /root/cmc-diploma && git pull && source .venv/bin/activate
cd alpha-beta-CROWN/experiments/fsdp_crown

# Сравнение памяти single-GPU vs FSDP=2
# Тестирует MLP с различной шириной (256..8192) и глубиной (2..8),
# а также ONNX-модели VNN-COMP
# Выводит baseline memory (хранение весов) и peak memory (compute_bounds)
torchrun --nproc_per_node=2 memory_experiment.py

# Результаты сохраняются в fsdp_memory_results.json
```

### Запуск abcrown с FSDP

```bash
cd /root/cmc-diploma && git pull && source .venv/bin/activate
cd alpha-beta-CROWN/complete_verifier

# Single GPU (baseline):
python abcrown.py --config ../experiments/fsdp_crown/mnist_256x6.yaml

# FSDP=2 (через torchrun):
torchrun --nproc_per_node=2 ../experiments/fsdp_crown/run_abcrown_fsdp.py \
  --config ../experiments/fsdp_crown/mnist_256x6.yaml
```

### Отладка NCCL

```bash
# Включить debug-логи TP-операторов (если добавлены)
TP_DEBUG=1 torchrun --nproc_per_node=2 examples/simple/test_tp_verification.py

# Проверить что NCCL вообще работает
torchrun --nproc_per_node=2 --master_port=29515 /tmp/nccl_test.py

# Уменьшить NCCL timeout для быстрого обнаружения deadlock (по умолчанию 30 мин)
NCCL_TIMEOUT=30 torchrun --nproc_per_node=2 examples/simple/test_tp_verification.py
```

### Эксперимент DP+FSDP: Domain-Parallel BaB

```bash
cd /root/cmc-diploma && git pull && source .venv/bin/activate
cd alpha-beta-CROWN/complete_verifier

# Single GPU (baseline):
CUDA_VISIBLE_DEVICES=0 python ../experiments/fsdp_crown/run_abcrown_fsdp.py \
  --config ../experiments/fsdp_crown/mnist_fc_bab_hard.yaml

# DP=2 + FSDP (domain-parallel BaB):
torchrun --nproc_per_node=2 ../experiments/fsdp_crown/run_abcrown_fsdp.py \
  --config ../experiments/fsdp_crown/mnist_fc_bab_hard.yaml
```

### Ключевые файлы

| Файл | Описание |
|------|----------|
| **Tensor Parallelism** | |
| `experiments/tp_model.py` | Общий модуль: TP-модель, dense-модель, copy weights |
| `experiments/crown/run.py` | Эксперимент CROWN: OOM single vs TP memory |
| `experiments/alpha_crown/run.py` | Эксперимент α-CROWN: численное сравнение bounds |
| `experiments/vnncomp_tp/verify_tp.py` | Корректность и звуковость TP на VNN-COMP моделях |
| `auto_LiRPA/auto_LiRPA/operators/tensor_parallel.py` | BoundLinearTP_Col/Row + DifferentiableAllReduce |
| `auto_LiRPA/auto_LiRPA/tp_utils.py` | tp_shard_bounded_module, зонирование подграфов |
| **FSDP** | |
| `experiments/fsdp_crown/verify_fsdp.py` | Корректность FSDP bounds (IBP + CROWN) |
| `experiments/fsdp_crown/memory_experiment.py` | Сравнение памяти single vs FSDP |
| `experiments/fsdp_crown/run_abcrown_fsdp.py` | Обёртка для запуска abcrown через torchrun |
| `experiments/fsdp_crown/mnist_256x6.yaml` | YAML-конфиг для abcrown (ONNX + VNNLIB) |
| `experiments/fsdp_crown/mnist_fc_bab_hard.yaml` | YAML-конфиг для BaB (eps=0.05, timeout=120) |
| `auto_LiRPA/auto_LiRPA/fsdp_utils.py` | fsdp_shard_bounded_module, gather/free хуки |
| **Domain-Parallel BaB** | |
| `complete_verifier/bab_parallel.py` | scatter_domain_dict, gather_result_dict |
| `complete_verifier/bab.py` | Интеграция DP в split_domain + anti-deadlock |
| **Общие** | |
| `auto_LiRPA/auto_LiRPA/backward_bound.py` | CROWN backward (TP zone logic + FSDP hooks) |
| `auto_LiRPA/auto_LiRPA/interval_bound.py` | IBP forward (isinstance fix + FSDP hooks) |
| `auto_LiRPA/auto_LiRPA/bound_general.py` | BoundedModule.__init__ (JIT trace + forward) |
| `complete_verifier/beta_CROWN_solver.py` | LiRPANet (FSDP auto-hook в __init__) |

## TODO
1. ✅ Эксперимент CROWN: single GPU vs TP=2 (OOM / memory)
2. ✅ Зафиксировать результаты TP в тексте диплома
3. ✅ Корректность и звуковость TP bounds на VNN-COMP моделях
4. ✅ Реализация FSDP для bound propagation (IBP + CROWN)
5. ✅ Эксперименты FSDP: корректность + память
6. ✅ Зафиксировать результаты FSDP в тексте диплома
7. ✅ Интеграция FSDP с полной верификацией (β-CROWN + BaB)
8. ✅ Domain-Parallel BaB: scatter/gather доменов + anti-deadlock
9. Зафиксировать результаты DP+FSDP BaB в тексте диплома
