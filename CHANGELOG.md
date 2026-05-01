# Changelog

## 2026-05-01 — Расширение FSDP на BoundConv и CIFAR-100 ResNet (VNN-COMP'24)

### Что сделано

Шардирование FSDP распространено с `BoundLinear` на `BoundConv` (веса
`[out_c, in_c, kH, kW]` шардируются по `dim=0` — оси `out_channels`,
аналогично `out_features` у BoundLinear).

| Файл | Изменение |
|------|-----------|
| `fsdp_utils.py` | `shardable_types = (BoundLinear, BoundConv)`; защита от двойного шардирования; `fsdp_free_node` использует `delattr` вместо `= None`; `fsdp_gather_node` использует `getattr(..., None)` |
| `backward_bound.py` | AllGather/free для `(BoundLinear, BoundConv)` в CROWN backward |
| `interval_bound.py` | free-хуки для `BoundConv` после IBP; `_delete_unused_bounds` защищена от AttributeError (FSDP мог уже удалить атрибут) |
| `experiments/fsdp_crown/cifar100/` | `download_cifar100.sh`, `cifar100_medium_fair.yaml`, `cifar100_large_fair.yaml` |

### Результаты на CIFAR-100 ResNet (VNN-COMP'24), fair-условия

`force_synchronous`, одинаковый batch=64, max_iterations=5.

| Модель | Конфигурация | Domains | BaB rounds | Peak GPU |
|--------|-------------|--------:|-----------:|---------:|
| ResNet-medium | Single GPU | 130 | 5 | ~960 МБ |
| ResNet-medium | FSDP=2 | 130 | 5 | ~930 МБ / rank |
| ResNet-large | Single GPU | 0 (verified at α-CROWN init) | 0 | 4 610.6 МБ |
| ResNet-large | FSDP=2 | 0 (verified at α-CROWN init) | 0 | 4 641.5 МБ / rank |

**Корректность:** bounds идентичны на обоих режимах.

**Память:** BoundConv-шардирование технически работает, но экономии не даёт.
ResNet-large весит ~95 МБ весов, однако alpha-тензоры (per-neuron ReLU slopes)
занимают >4 ГБ и являются узким местом. AllGather-overhead сопоставим с
экономией от шардирования весов.

### Вывод

FSDP для BoundConv реализован и верифицирован на VNN-COMP'24 CIFAR-100 ResNet.
Для значимой экономии в BaB-режиме необходимо дополнительно шардировать
alpha-тензоры — следующий шаг разработки.

---

## 2026-05-01 — Честное сравнение FSDP vs single в полной верификации

### Что обнаружено

Старые результаты «4–27× экономии памяти при DP=2+FSDP в BaB» были артефактом
несимметричной конфигурации:

- single-GPU выполнял `auto_enlarge_batch_size` и наращивал batch до десятков
  тысяч, заполняя память;
- FSDP-режим (`_dp_active`) держал batch фиксированным (нужно для синхронности
  AllGather), плюс отключал `early_stop` и `pruning_in_iteration`.

В итоге сравнивались **разные нагрузки** — single-GPU считал bounds на огромном
батче, FSDP — на маленьком. Поэтому single-GPU расходовал много памяти, и
получалось «27×».

### Что сделано

Добавлена опция `bab.force_synchronous` (CLI: `--force_synchronous`), которая
заставляет single-GPU пройти ту же ветку, что и `_dp_active`: фиксированный
batch, без early-stop / pruning-in-iteration / auto-enlarge. Это позволяет
прогнать оба режима с **идентичным workload** и честно сравнить пиковую память.

| Файл | Изменение |
|------|-----------|
| `complete_verifier/arguments.py` | Новый CLI-аргумент `--force_synchronous` (`bab.force_synchronous`) |
| `complete_verifier/bab.py` | `_sync_mode = _dp_active or arguments.Config['bab']['force_synchronous']` подменяет `_dp_active` в обеих местах |
| `experiments/fsdp_crown/mnist_fc_fair_512.yaml` | Новый YAML: `batch_size: 512`, `auto_enlarge_batch_size: false`, `pruning_in_iteration: false`, `force_synchronous: true`, `max_iterations: 10` |
| `experiments/fsdp_crown/mnist_fc_fair_4096.yaml` | Аналогично, `batch_size: 4096` |
| `experiments/vnncomp_tp/download_mnist_fc.sh` | URL VNN-COMP MNIST-FC переехал на VNN-COMP/vnncomp2022_benchmarks, файлы `.gz` |

### Результаты на mnist-net_256x6, eps=0.05, ровно 10 раундов BaB

| Конфигурация | Batch | Domains | BaB rounds | Peak GPU |
|--------------|------:|--------:|-----------:|---------:|
| Single GPU   |   512 |   6 236 |         10 | 211.7 МБ |
| FSDP=2       |   512 |   6 236 |         10 | 215.7 МБ / rank |
| Single GPU   | 4 096 |  49 888 |         10 | 1 537.3 МБ |
| FSDP=2       | 4 096 |  49 888 |         10 | 1 541.4 МБ / rank |

**Корректность:** workload идентичен (тот же batch, число доменов, раундов).

**Память:** на mnist-net_256x6 веса занимают ~1.5 МБ при общем peak ~1.5 ГБ
(99.9% памяти — A-матрицы и активации). Шардирование 1.5 МБ не даёт измеримой
экономии, AllGather добавляет ~4 МБ overhead. **FSDP+BaB не даёт выигрыша на
маленьких моделях**, но и не вредит точности/корректности.

Реальная экономия пиковой памяти от FSDP видна на **широких моделях** (Эксп.~4
из текста диплома): 34–39% при $h{=}4096{-}8192$, $d{=}4{-}8$.

### Замечание про runpod-инфраструктуру

На новом поде (2× A40 на PCIe, без NVLink) NCCL P2P зависает.
Обходное решение: `NCCL_P2P_DISABLE=1` перед `torchrun`.

## 2026-05-01 — Честное сравнение FSDP в верификации ViT и попытка показать экономию

### Что сделано

1. Скачан VNN-COMP'23 ViT (`pgd_2_3_16`, `ibp_3_3_8`) +
   `experiments/fsdp_crown/vit/{download_vit.sh, vit_pgd_fair.yaml}`.
2. Прогнано честное сравнение на `pgd_2_3_16` (10 раундов BaB, fixed batch=32,
   force_synchronous): **78 доменов на обоих режимах, peak 1 020 МБ обе
   стороны**. Старые «7.5×» экономии (10 356 vs 1 382 МБ) были тем же
   артефактом auto-enlarge_batch_size, что и для MNIST-FC.
3. Сгенерирован «реалистичный» по размеру весов MLP
   (`experiments/fsdp_crown/wide_mlp/create_wide_mlp.py`,
   `wide_mlp_4096x4.onnx`, 53.6 М параметров, ~204 МБ весов) +
   YAML-конфиги `wide_mlp_fair.yaml` (BaB) и `wide_mlp_incomplete.yaml`.
4. Прогнано на нём: \\
   - **BaB fair (batch=64, max_iterations=5):** single peak 4 199 МБ vs.
     FSDP=2 4 748 МБ/rank — **FSDP проигрывает на 13%** из-за overhead
     AllGather полной матрицы (64 МБ за раз) поверх α/β-оптимизационных тензоров.
   - **incomplete (CROWN, batch=1):** single 4 199 МБ vs. FSDP=2 4 748 МБ/rank —
     тот же knock-on overhead в abcrown-pipeline.
5. Воспроизведён прямой `compute_bounds(CROWN)` через
   `experiments/fsdp_crown/memory_experiment.py`: на широких MLP экономия
   **34–39%** peak GPU памяти (h=4096..8192, d=4..8), bounds побитово идентичны.
6. Написан `vit_memory_experiment.py` с собственной TinyViT (без norm,
   без BatchNorm-on-tokens). Запуск через CROWN/IBP упирается в softmax
   (`BoundReduceMax/BoundReciprocal` не поддержаны для возмущённых индексов
   без adhoc-tuning, как в `exp_configs/vnncomp23/vit.yaml`). Скрипт оставлен
   в репозитории как отправная точка для дальнейшей работы.

### Дополнительно

- `auto_LiRPA/operators/reshape.py` — патч JIT-baked batch-dim теперь
  применяется только когда `prod(new_shape) == x.numel()`. Иначе
  reshape вида `[B, N, D] -> [B*N, D]` ломался: shape[0] становился
  равным `B` вместо `B*N`.

### Итог для слайдов

- **Эксп.~4** (incomplete-верификация на широких MLP, прямой
  `compute_bounds`): 34–39% экономия — *настоящий* результат FSDP.
- **Эксп.~5** (BaB на mnist-net_256x6): идентичный workload, FSDP
  даёт +4 МБ overhead AllGather, экономии нет (модель слишком мелкая).
- **Эксп.~6** (BaB на VNN-COMP ViT pgd_2_3_16): то же — workload идентичен,
  экономии нет (веса 0.3 МБ vs $A$-матрицы 1 ГБ).
- В BaB-режиме на крупных моделях FSDP остаётся способом
  *поместить* модель в память нескольких GPU — независимая полезная
  функциональность, отдельная от прямой экономии в incomplete.

## 2026-04-07 — FSDP-верификация Vision Transformer (ViT)

### Что сделано

Успешно запущена FSDP-верификация (complete verification, BaB) на модели Vision Transformer из VNN-COMP 2023 (`pgd_2_3_16`, 2 transformer-блока, 3 heads, ~75K параметров). Для этого потребовалось исправить обработку динамических batch-размеров в auto_LiRPA.

### Проблема: JIT-трассировка и динамические формы

При конвертации ONNX-модели ViT в auto_LiRPA используется `torch.jit.trace`, который превращает `Shape(input)` → `Slice` → `Concat` цепочки в константы с batch=1. Когда BaB увеличивает batch (разбиение доменов), batch-размер в reshape-целях остаётся равным 1, что приводит к ошибкам формы тензоров.

Пример: `Conv [2, 48, 2, 2] → Reshape [1, 48, -1] → [1, 48, 8]` вместо `[2, 48, 4]`.

### Исправления

| Файл | Изменение |
|------|-----------|
| `auto_LiRPA/operators/reshape.py` | `BoundReshape.forward()`: замена shape[0] на фактический batch_size при несовпадении с JIT-baked значением |
| `auto_LiRPA/operators/constant.py` | `BoundConstantOfShape.forward()`: аналогичная коррекция batch-размера в аргументе формы |
| `auto_LiRPA/operators/slice_concat.py` | `BoundConcat.forward()`: expand тензоров с batch=1 до max_batch перед `torch.cat` (для JIT-baked констант вроде cls_token expansion) |
| `auto_LiRPA/bound_general.py` | Хранение `_batch_size` в `set_input()` и передача в операторы через `get_forward_value()` |

### Результаты: pgd_2_3_16, prop_7715 (VNN-COMP 2023)

| Метрика | Single GPU | FSDP 2GPU |
|---------|-----------|-----------|
| Результат | timeout (100s) | timeout (100s) |
| CROWN bounds | `-2.195, 3.895, 4.679, ...` | `-2.195, 3.895, 4.679, ...` **(идентичны)** |
| Alpha-CROWN bound | -0.626 | -0.661 |
| Specs verified (incomplete) | 8 / 9 | 8 / 9 |
| BaB rounds | 32 | 24 |
| Domains visited | ~13 000 | ~1 278 |
| Peak GPU memory | 10 356 MB | 1 382 MB / rank |

**Корректность:** CROWN bounds побитово идентичны между single GPU и FSDP 2GPU. Alpha-CROWN отличается незначительно из-за разных условий ранней остановки (FSDP отключает early_stop_patience для синхронизации AllGather).

**Память:** FSDP использует ~7.5× меньше памяти на GPU (частично из-за отключённого auto_enlarge_batch_size в FSDP-режиме, необходимого для синхронизации итераций).

**Производительность:** Single GPU исследует ~10× больше доменов за тот же timeout благодаря увеличенным batch-размерам и отсутствию AllGather-overhead.

## 2026-04-07 — Domain-Parallel BaB: полная верификация на нескольких GPU

### Что сделано

Реализован Domain-Parallel Branch-and-Bound: каждый GPU обрабатывает свою порцию доменов при BaB-верификации, а веса остаются шардированными через FSDP. Это даёт как ускорение (каждый GPU считает batch/N доменов), так и экономию памяти (A-матрицы и промежуточные тензоры пропорциональны batch/N, веса шардированы).

### Новые файлы

| Файл | Описание |
|------|----------|
| `complete_verifier/bab_parallel.py` | scatter_domain_dict, gather_result_dict, pickle-based GPU list gathering |
| `experiments/fsdp_crown/mnist_fc_bab_hard.yaml` | YAML-конфиг для BaB с eps=0.05 (долгая задача для тестирования) |

### Изменённые файлы

| Файл | Изменение |
|------|-----------|
| `complete_verifier/bab.py` | Интеграция DP в `split_domain`: scatter перед `update_bounds`, gather после; отключение `stop_criterion` и `early_stop_patience` при DP для предотвращения FSDP deadlock; отключение `auto_batch_size` при DP для синхронизации очередей доменов |

### Архитектура Domain-Parallel BaB

```
1. [все ranks]  pick_out(batch=B) — одинаковые домены на всех GPU
2. [все ranks]  branching + build_history → 2B child-доменов
3. [scatter]    d_local = scatter_domain_dict(d, rank, ws) — каждый получает B/ws
4. [каждый]     ret_local = net.update_bounds(d_local) — compute_bounds на batch/ws
5. [gather]     ret = gather_result_dict(ret_local, ws) — AllGather обратно
6. [все ranks]  domains.add(ret, d) — одинаковые очереди на всех GPU
```

Ключевые решения:
- **Anti-deadlock**: `stop_criterion_func=lambda x: False` + `early_stop_patience=1e9` при DP, чтобы все ranks выполняли одинаковое число итераций CROWN-optimized (иначе один rank выходит раньше → deadlock на FSDP AllGather)
- **NCCL-only**: `_gather_list` сериализует Python-объекты (betas, split_history) через pickle → GPU byte tensors → NCCL AllGather (т.к. `all_gather_object` требует gloo backend)
- **Cross-device safety**: тензоры перемещаются на CPU перед pickle и обратно на local CUDA после unpickle

### Результаты

**mnist-net_256x6, eps=0.05, BaB timeout=120s:**

| Конфигурация | Batch | Peak GPU    | BaB rounds | Domains  |
|--------------|-------|-------------|------------|----------|
| Single GPU   | 4096  | 24,130 MB   | 11         | 393,952  |
| DP=2 + FSDP  | 4096  | 6,372 MB    | 14         | —        |
| Single GPU   | 512   | 6,046 MB    | 11         | 65,628   |
| DP=2 + FSDP  | 512   | 226 MB      | 24         | 20,572   |

Экономия памяти: **~4× при batch=4096**, **~27× при batch=512** (каждый GPU хранит A-матрицы только для batch/N доменов + FSDP шардирует веса).

### Ограничения

- Тестировалось только на 2 GPU (архитектурно поддерживает N GPU)
- Только MLP (mnist-net_256x6), activation split branching
- Не тестировался случай `safe` (только timeout)

## 2026-04-05 — Реализация FSDP для верификации нейронных сетей

### Что сделано

Реализован Fully Sharded Data Parallelism (FSDP) для bound propagation в auto_LiRPA — альтернатива Tensor Parallelism, дающая побитово идентичные bounds при значительной экономии памяти.

### Новые файлы

| Файл | Описание |
|------|----------|
| `auto_LiRPA/auto_LiRPA/fsdp_utils.py` | Утилиты FSDP: шардирование, послойный AllGather/free |
| `experiments/fsdp_crown/verify_fsdp.py` | Тест корректности FSDP bounds |
| `experiments/fsdp_crown/memory_experiment.py` | Эксперимент: сравнение памяти single vs FSDP |
| `experiments/fsdp_crown/run_abcrown_fsdp.py` | Обёртка для запуска abcrown через torchrun с FSDP |
| `experiments/fsdp_crown/mnist_256x6.yaml` | YAML-конфиг для abcrown (ONNX + VNNLIB) |

### Изменённые файлы

| Файл | Изменение |
|------|-----------|
| `auto_LiRPA/backward_bound.py` | FSDP hooks: fsdp_gather_node перед BoundLinear.bound_backward, fsdp_free_node после |
| `auto_LiRPA/interval_bound.py` | FSDP hooks: fsdp_free_node после interval_propagate для BoundLinear |
| `complete_verifier/beta_CROWN_solver.py` | Auto-FSDP hook в LiRPANet.__init__: шардирование при dist.is_initialized() |

### Архитектура FSDP

1. **fsdp_shard_bounded_module** — обходит граф BoundedModule, шардирует BoundParams (веса линейных слоёв) по dim=0
2. **fsdp_gather_node** — AllGather для одного BoundParams перед использованием слоя
3. **fsdp_free_node** — освобождение полной матрицы после использования (остаётся только шард)
4. В каждый момент времени в GPU-памяти находится не более одной полной весовой матрицы

### Результаты экспериментов

**Корректность (8 тестов, IBP + CROWN, PyTorch + ONNX):**
Все тесты PASS, lb_diff = ub_diff = 0.00 (побитово идентичны single-GPU).

**Память (baseline = хранение весов, peak = во время compute_bounds):**

| Модель | Baseline savings | Peak savings |
|--------|-----------------|--------------|
| MLP h=256, d=4 | 57% | — (overhead > savings) |
| MLP h=1024, d=4 | 79% | — |
| MLP h=4096, d=4 | 82% | 34% |
| MLP h=8192, d=4 | 82% | 34% |
| MLP h=4096, d=8 | 90% | 39% |

### Сравнение TP vs FSDP

| Свойство | TP | FSDP |
|----------|----|----|
| Peak memory savings | ~2× | 1.3–1.4× |
| Точность bounds | Потеря (IBP fallback) | Побитово идентичны |
| Совместимость с β-CROWN | Сложно (зоны, β-координация) | Просто (граф не меняется) |
| Объём реализации | ~500 строк | ~100 строк |

### Обновление текста диплома

- Переписан раздел 5 «Применение параллелизма к верификации»: FSDP теперь как полноценный метод, сравнение TP vs FSDP, теоретический анализ памяти
- Добавлены эксперименты 4 (корректность FSDP) и 5 (память FSDP) в раздел 6
- Обновлено заключение (раздел 7): два подхода, компромисс память-точность, FSDP как путь к полной верификации

## 2026-04-05 — Корректность и звуковость TP bounds на VNN-COMP моделях

### Что сделано

Реализовано зонирование шардированных подграфов для корректного CROWN backward через несколько TP-зон. Добавлен тест корректности на VNN-COMP ONNX-моделях (MNIST-FC 256×2, 256×4, 256×6).

### Ключевые изменения

- `_mark_sharded_zone` в `tp_utils.py`: BFS-разметка зон между Col–Row парами
- `BoundLinearTP_Col.bound_backward`: пропуск AllReduce если start_node в той же зоне
- `_tp_accumulate_bias` в `backward_bound.py`: зоно-зависимый skip DifferentiableAllReduce
- `get_sparse_C`: force `sparse_intermediate_bounds = False` для узлов в зонах
- `ibp_intermediate = True` для BoundLinearTP_Col (IBP для промежуточных bounds в зонах)

### Результаты

- 256×2 (1 пара): |lb_diff| = 3e-8 (машинная точность)
- 256×4 (2 пары): |lb_diff| = 2.5, SOUND (IBP fallback теряет точность)
- 256×6 (3 пары): |lb_diff| = 792, SOUND
- Кросс-ранговая согласованность: 0.00 для всех моделей

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
