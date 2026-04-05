# Changelog

## 2026-04-05 — Исправление OOM и contiguous-ошибки в TP bound propagation

### Контекст

При запуске TP-эксперимента (`torchrun --nproc_per_node=2 oom_tp_experiment.py --mode tp --method CROWN`) возникали две ошибки, из-за которых TP-режим вообще не работал, тогда как single GPU проходил успешно.

### Ошибка 1: `ValueError: Tensors must be contiguous`

**Файл:** `auto_LiRPA/operators/tensor_parallel.py`

**Симптом:** NCCL `all_reduce` падал с `ValueError: Tensors must be contiguous` в `BoundLinearTP_Col.bound_backward`.

**Причина:** Тензоры `lA_x`/`uA_x`, возвращаемые из `BoundLinear.bound_backward`, могут быть non-contiguous views (результат matmul, transpose и т.д.). NCCL `all_reduce` требует contiguous-тензоры.

**Решение:**
- Добавлен `.contiguous()` перед каждым `dist.all_reduce` в `BoundLinearTP_Col.bound_backward`
- Хелпер `_all_reduce_inplace` заменён на `_contiguous_all_reduce` — возвращает значение (не in-place), корректно обрабатывает non-contiguous тензоры, включая вложенные tuple/list
- Аналогичные `.contiguous()` добавлены превентивно в `BoundLinearTP_Row.forward` и `interval_propagate`
- Структура `result` перестроена для корректного обновления после contiguous-копирования

### Ошибка 2: OOM 4096 GiB при вычислении intermediate bounds

**Файл:** `auto_LiRPA/interval_bound.py`

**Симптом:** `CUDA out of memory. Tried to allocate 4096.00 GiB` — при том, что single GPU (без TP) проходил за 26.8 GB.

**Причина:** Функция `check_IBP_first_linear` использует точную проверку типа `type(node) == BoundLinear`. Для `BoundLinearTP_Col` (подкласс `BoundLinear`) эта проверка возвращала `False`, из-за чего вместо дешёвого IBP (O(hidden × input) памяти) для intermediate bounds использовался CROWN backward. Это приводило к попытке аллоцировать A-матрицу размера (batch=2048, hidden/2=131072, input=4096) = 4096 GiB.

**Решение:** Заменён `type(node) == BoundLinear` на `isinstance(node, BoundLinear)`, что корректно распознаёт TP-подклассы. IBP для первого линейного слоя безопасен для TP: каждый GPU считает bounds для своего шарда hidden dimension через унаследованный `interval_propagate`.

## 2026-03-08 — Добавлен эксперимент OOM (single GPU) vs TP=2

### Что добавлено

**Файл:** `alpha-beta-CROWN/auto_LiRPA/examples/simple/oom_tp_experiment.py`

Добавлен отдельный скрипт для воспроизводимого эксперимента:
- `--mode single` — обычная dense-модель на одной GPU (базовый сценарий для OOM)
- `--mode tp` — TP-модель (`SimpleTPModel`) для запуска через `torchrun --nproc_per_node=2`
- единые параметры (`input_dim`, `hidden_dim`, `batch_size`, `method`) для честного сравнения
- печать пикового потребления памяти (`max_memory_allocated`, `max_memory_reserved`) по ранкам
- явная детекция OOM с кодом выхода `2`

### Как запускать

```bash
# 1) Single GPU: пытаемся получить OOM
python3 examples/simple/oom_tp_experiment.py \
  --mode single --method CROWN \
  --input-dim 4096 --hidden-dim 262144 --batch-size 2048

# 2) Те же параметры, TP=2
torchrun --nproc_per_node=2 examples/simple/oom_tp_experiment.py \
  --mode tp --method CROWN \
  --input-dim 4096 --hidden-dim 262144 --batch-size 2048
```

Ожидаемое поведение: `single` падает с OOM, при этом `tp` на тех же параметрах проходит и показывает меньшую память на ранк.

## 2026-02-24 — Исправление JIT-трассировки и запуска TP-тестов

### Контекст

При создании `BoundedModule` из TP-модели auto_LiRPA вызывает `torch.jit.trace` для построения ONNX-графа. Внутри TP-слоёв вызывался `dist.all_reduce`, который JIT не может трассировать — падал с ошибкой `RuntimeError: Tried to trace ProcessGroup`. После исправления трассировки обнаружились дополнительные проблемы в тестовом скрипте, из-за которых `torchrun` не мог корректно запустить воркеры.

### Ошибка 1: JIT не может трассировать `dist.all_reduce`

**Файл:** `alpha-beta-CROWN/auto_LiRPA/examples/simple/tp_model.py`

**Симптом:** `RuntimeError: Tried to trace ProcessGroup but it is not part of the active trace`

**Причина:** `TPLinearRowOp.forward()` безусловно вызывал `dist.all_reduce()` во время JIT-трассировки. JIT tracer выполняет forward pass для построения графа, но не умеет сериализовать объект `ProcessGroup`.

**Решение:** Добавлен guard `torch.jit.is_tracing()` в двух местах:
- `TPLinearRowOp.forward()` — пропуск `dist.all_reduce` при трассировке
- `RowParallelLinear.forward()` — пропуск `x.size(-1) == self.in_features` (конвертация трассированного тензора в bool вызывала `TracerWarning`)

Это корректно, т.к. `symbolic()` метод независимо создаёт правильный ONNX-узел `customOp::TPLinearRow`, а реальный `all_reduce` выполняется в `BoundLinearTP_Row`.

### Ошибка 2: Двойной spawn при запуске через `torchrun`

**Файл:** `alpha-beta-CROWN/auto_LiRPA/examples/simple/test_tp_verification.py`

**Симптом:** NCCL `Connection refused` при создании `BoundedModule`

**Причина:** `torchrun --nproc_per_node=2` запускает 2 процесса, каждый из которых вызывал `main()`. Функция `main()` не проверяла наличие `torchrun` и через `mp.spawn` создавала ещё 2 дочерних процесса — итого 4 процесса конкурировали за NCCL.

**Решение:** В начале `main()` добавлена проверка `LOCAL_RANK in os.environ`. Если переменная установлена (torchrun-режим), `run_worker` вызывается напрямую.

### Ошибка 3: Перезапись `MASTER_PORT` при torchrun

**Файл:** тот же `test_tp_verification.py`

**Симптом:** Воркеры зависали навечно — ни одной строки вывода

**Причина:** `run_worker()` безусловно устанавливал `os.environ['MASTER_PORT'] = '29500'`, перезаписывая порт, на котором `torchrun` фактически слушал. `dist.init_process_group` подключался к порту 29500, где никто не слушал → бесконечное ожидание.

**Решение:** `MASTER_ADDR` и `MASTER_PORT` устанавливаются только если не заданы в окружении (`if 'MASTER_PORT' not in os.environ`).

### Верификация

На VM (2× A40, NCCL backend) успешно пройдены:
- `python3 examples/simple/toy.py` — регрессия (IBP, CROWN, alpha-CROWN)
- `torchrun --nproc_per_node=2 examples/simple/test_tp_verification.py` — IBP и CROWN bounds на 2 GPU

### Коммиты

| Хеш | Описание |
|------|----------|
| `d305336` | `torch.jit.is_tracing()` guard в TP-слоях |
| `91a4f91` | Определение torchrun-режима через `LOCAL_RANK` |
| `d471023` | Условная установка `MASTER_PORT` |
| `b02cd18` | Удаление временного debug-логирования |
