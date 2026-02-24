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
ssh [ask user runpod.ai command] -i /Users/svorobyov/Documents/cmc_mmp/vastai
```

## Полезные команды

### Тесты на VM

```bash
# Подключение
ssh [ask user runpod.ai command] -i /Users/svorobyov/Documents/cmc_mmp/vastai

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
| `examples/simple/tp_model.py` | TP-модель (ColumnParallel, RowParallel, ONNX symbolic) |
| `examples/simple/test_tp_verification.py` | Тест IBP + CROWN на 2 GPU |
| `auto_LiRPA/operators/tensor_parallel.py` | BoundLinearTP_Col/Row — bound propagation |
| `auto_LiRPA/bound_general.py` | BoundedModule.__init__ (JIT trace + forward) |
| `auto_LiRPA/parse_graph.py:206` | `torch.jit._get_trace_graph` вызов |

## TODO
1. Поставить эксперимент с Tensor Parallel верификацией нейронных сетей
