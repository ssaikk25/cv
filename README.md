# AI Challenge — Stage 1: сегментация области правки

Бинарная сегментация: по изображению предсказать маску области, которая была изменена
(отредактирована/инпейнтирована). Метрика — попиксельное сравнение с ground-truth маской.

## Статус

- [x] Разведка данных (структура, форматы, объёмы)
- [ ] Воспроизведение бейзлайна (U-Net + ResNet34)
- [ ] Локальный валидатор (group split по маске)
- [ ] Улучшения (энкодер, размер входа, лоссы, TTA)
- [ ] Финальная посылка

## Структура репозитория

```
.
├── README.md              # этот файл
├── .gitignore             # исключает данные (десятки ГБ) и артефакты обучения
├── code/
│   └── code/
│       ├── baseline.py        # U-Net + ResNet34 на segmentation-models-pytorch
│       ├── requirements.txt
│       └── README.md          # инструкция от организаторов
└── data/                  # ← НЕ в репозитории (см. «Данные»)
```

## Данные

Датасет **не хранится в репозитории** — это ~52 ГБ распакованных данных и ~56 ГБ архивов.
Он скачивается/выдаётся отдельно и распаковывается **в корень проекта** (пути в CSV
относительные, поэтому корень распаковки критичен):

| Архив | Распакованный размер | Куда распаковать | Содержимое |
|---|---|---|---|
| `train_stage1.zip` | 36.3 ГБ | корень проекта | `stage1/train.csv`, `stage1/train/{src,img,mask}` |
| `test_stage1.zip` | 0.55 ГБ | корень проекта | `test_stage1/{test.csv,submission.csv,test_stage1_img}` |
| `submission.zip` | 0.01 ГБ | корень проекта | `submission.csv`, `predictions/*_pred.png` |
| `nanobanana.zip` | 15.7 ГБ | корень проекта | `nanobanana/pico_nanobanana/{src,img}` + `pico_nanobanana.csv` |

Ожидаемая раскладка после распаковки:

```
stage1/train.csv             103 699 строк: orgl_img_path, chng_img_path, gt_path
stage1/train/src/            42 530 jpg   — оригиналы
stage1/train/img/           103 699 jpg   — изменённые изображения (вход модели)
stage1/train/mask/           87 799 png   — маски (0/255), ground truth
test_stage1/test.csv          2 160 строк: img_path
test_stage1/test_stage1_img/  2 160 jpg
test_stage1/submission.csv    шаблон посылки
```

### Особенности датасета (проверено)

1. **Оригинал есть не везде.** `orgl_img_path` пустой у 55 460 из 103 699 строк —
   доступно только 42 530 оригиналов. Если использовать src как дополнительный вход,
   придётся ограничиться подвыборкой ~41 %.
2. **Маски переиспользуются до 5 раз** с разными изменёнными картинками
   (87 799 уникальных масок на 103 699 строк). Случайный split даёт утечку валидации —
   сплитить надо **по `gt_path`** (GroupShuffleSplit).
3. **Размеры не совпадают.** `chng`/`gt` согласованы между собой (маска в разрешении
   изменённого изображения), а `src` бывает другого размера (например 612×612 против
   608×608). Ресайз обязателен.
4. **Маски — антиалиасингованные 0/255** (RGB, каналы идентичны), в предсказании
   сохраняются как одноканальные PNG в исходном разрешении.
5. **`test.csv` лежит в `test_stage1/`**, а бейзлайн по умолчанию ищет `stage1/test.csv`.
   Поскольку пути внутри `test.csv` (`test_stage1_img/...`) относительны `DATA_DIR`,
   для инференса нужно: `--data-dir <корень>/test_stage1 --test-csv test.csv`.

## Окружение

`torch` требует Python 3.10–3.12; системный Python 3.14 не подойдёт (колёс нет).

```powershell
py -3.11 -m venv .venv          # или conda create -n aic python=3.11 -y
.\.venv\Scripts\Activate.ps1
pip install -U pip
pip install -r code\code\requirements.txt
python -c "import torch, smp, cv2; print(torch.__version__, torch.cuda.is_available())"
```

Локальная машина: **RTX 3050 Laptop, 4 ГБ VRAM**, драйвер 596.21.
Бейзлайн рассчитан на ~4.5 ГБ при `batch_size=32, img_size=256`, поэтому дефолты
надо уменьшать (`--batch-size 8 --img-size 256`) либо использовать AMP.

## Обучение и предсказание

```powershell
# обучение
python code\code\baseline.py train --data-dir "D:\ai challenge" `
  --epochs 15 --batch-size 8 --img-size 256 --num-workers 4

# предсказание по тесту
python code\code\baseline.py predict `
  --data-dir "D:\ai challenge\test_stage1" --test-csv test.csv `
  --checkpoint checkpoints\baseline_unet_resnet34.best.pth
```

Чекпоинты и `config_train.json` пишутся рядом со скриптом (`code/code/checkpoints/`)
и в git не попадают — параметры запуска фиксируются в `config_train.json` для
воспроизведения результата.

## Git / GitHub

Данные и чекпоинты в репозиторий не коммитятся (`.gitignore`). Первичная настройка:

```powershell
git init -b main
git add .
git commit -m "Stage 1: разведка данных и бейзлайн"
git remote add origin git@github.com:<USER>/<REPO>.git
git push -u origin main
```

Если репозиторий уже создан на GitHub с README, вместо `git init` +
`git remote add` удобнее `git clone` и перенос файлов.
