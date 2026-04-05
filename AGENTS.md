# AGENTS.md

## Диплом

**Тема:** Параллелизмы в верификации нейронных сетей

**Суть работы:** Формальная верификация нейронных сетей -- задача доказательства того, что сеть удовлетворяет заданным спецификациям (например, робастность к adversarial-атакам). Задача NP-полна для ReLU-сетей. Основной инструмент -- Alpha-Beta-CROWN (победитель VNN-COMP), использующий bound propagation на GPU. Цель диплома -- применить параллелизмы (Tensor Parallelism и др.) к верификации для масштабирования на более крупные сети и несколько GPU.

Все что касается текста диплома и презентации должно быть написано на русском.
Можно использовать общеизвестные термины на английском (например, LLM, Data Parallel, FSDP). В таком случае термины нужно расшифровать в обзоре литературы.

## Structure

- `./alpha-beta-CROWN` -- форк alpha-beta-CROWN с реализацией Tensor Parallel верификации
- `./diploma-paper` -- текст диплома (LaTeX)

## GPU-виртуалка

Подключение:
```bash
ssh awmfmmsr5lyv1r-6441173c@ssh.runpod.io -i /Users/svorobyov/Documents/cmc_mmp/vastai
```

## Полезные команды

### Тесты на VM

```bash
# Подключение
ssh awmfmmsr5lyv1r-6441173c@ssh.runpod.io -i /Users/svorobyov/Documents/cmc_mmp/vastai

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

### Отладка NCCL

```bash
# Включить debug-логи TP-операторов (если добавлены)
TP_DEBUG=1 torchrun --nproc_per_node=2 examples/simple/test_tp_verification.py

# Проверить что NCCL вообще работает
torchrun --nproc_per_node=2 --master_port=29515 /tmp/nccl_test.py

# Уменьшить NCCL timeout для быстрого обнаружения deadlock (по умолчанию 30 мин)
NCCL_TIMEOUT=30 torchrun --nproc_per_node=2 examples/simple/test_tp_verification.py
```

### Ключевые файлы

| Файл | Описание |
|------|----------|
| `experiments/tp_model.py` | Общий модуль: TP-модель, dense-модель, copy weights |
| `experiments/crown/run.py` | Эксперимент CROWN: OOM single vs TP memory |
| `experiments/alpha_crown/run.py` | Эксперимент α-CROWN: численное сравнение bounds |
| `auto_LiRPA/auto_LiRPA/operators/tensor_parallel.py` | BoundLinearTP_Col/Row + DifferentiableAllReduce |
| `auto_LiRPA/auto_LiRPA/interval_bound.py:222` | isinstance fix для IBP first linear |
| `auto_LiRPA/auto_LiRPA/bound_general.py` | BoundedModule.__init__ (JIT trace + forward) |

## TODO
1. ✅ Эксперимент CROWN: single GPU vs TP=2 (OOM / memory)
2. ✅ Зафиксировать результаты в тексте диплома
3. Эксперимент α-CROWN: верификация корректности (TP=1 vs single) и TP=2
4. β-CROWN + BaB интеграция (future work)
