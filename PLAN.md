## Статус: текст диплома

- ✅ Пункт 1.1 — отличия CROWN / α-CROWN / α-β-CROWN, полная vs неполная верификация
- ✅ Пункт 2.1 — задача обучения LLM, трансформеры, матричные умножения
- ✅ Пункт 2.2 — тензорный параллелизм (TP): разбиение весов, AllGather / ReduceScatter
- ✅ Пункт 2.3 и 2.4 — Data Parallel, FSDP
- ✅ Пункт 3 — почему TP подходит, TP для полной верификации, FSDP как альтернатива
- ✅ Пункт 4 — эксперименты: OOM, корректность, VNN-COMP модели, FSDP bounds + память
- ✅ Пункт 5 — заключение, future work

## Статус: реализация

### Tensor Parallelism (неполная верификация)
- ✅ TP-операторы для auto_LiRPA (BoundLinearTP_Col/Row)
- ✅ Зонирование подграфов для CROWN backward
- ✅ Эксперимент OOM single vs TP=2
- ✅ Корректность на VNN-COMP ONNX-моделях (MNIST-FC)

### FSDP (неполная верификация)
- ✅ Послойный AllGather/free для bound propagation
- ✅ Побитово идентичные bounds (8 тестов)
- ✅ Экономия памяти 34–39% peak на широких MLP

### FSDP + Data Parallel (полная верификация, BaB)
- ✅ Domain-parallel BaB: scatter доменов по GPU, gather результатов
- ✅ Защита от deadlock (отключение early exit в CROWN-optimized)
- ✅ NCCL-only gathering (pickle → GPU byte tensors)
- ✅ Тесты: batch=4096 и batch=512, mnist-net_256x6, eps=0.05

### Результаты DP+FSDP для BaB

| Конфигурация    | Batch | Peak GPU    | BaB rounds | Domains  |
|-----------------|-------|-------------|------------|----------|
| Single GPU      | 4096  | 24,130 MB   | 11         | 393,952  |
| DP=2 + FSDP     | 4096  | 6,372 MB    | 14         | —        |
| Single GPU      | 512   | 6,046 MB    | 11         | 65,628   |
| DP=2 + FSDP     | 512   | 226 MB      | 24         | 20,572   |

### Что НЕ тестировалось
- Более 2 GPU
- Свёрточные сети (только MLP)
- Input split branching (только activation split)
- Случай когда BaB реально верифицирует (safe), а не timeout
