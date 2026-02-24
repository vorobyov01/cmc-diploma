# Changelog

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
