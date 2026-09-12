# AI Challenge Stage 1 — baseline (UNet + ResNet34)

Минимальный бейзлайн для бинарной сегментации

## Данные

Распакуйте датасет и укажите корень (в `baseline.py` — `DATA_DIR`):

```
DATA_DIR/
  stage1/train.csv          # orgl_img_path, chng_img_path, gt_path
  stage1/test.csv           # img_path
  stage1/train/...
```

Пути в CSV — **относительные** от `DATA_DIR`. Переопределить: `--data-dir /path/to/your_dir`.

## Зависимости

Python 3.10+. В каталоге с файлами:

```bash
python -m venv .venv && source .venv/bin/activate   # или conda create -n aic python=3.11 -y && conda activate aic
pip install -U pip
pip install -r requirements.txt
```

Проверка: `python -c "import torch,smp,cv2; print(torch.__version__, torch.cuda.is_available())"`

Без GPU: в `requirements.txt` замените строку `--extra-index-url` на `https://download.pytorch.org/whl/cpu`.

## Обучение

```bash
python baseline.py train --data-dir /path/to/your_dir
```

Полезные флаги: `--epochs 15`, `--batch-size 32`, `--img-size 256`, `--lr 1e-4`, `--checkpoint-dir code/checkpoints`.

Рядом со скриптом: `checkpoints/*.pth`, `config_train.json` (аргументы train — **для воспроизведения решения**).

Структура:

```
code/
  baseline.py
  config_train.json     # создаётся при train
  checkpoints/...
```

## Submit

Запустите `predict` по выданному `test.csv` и передайте результат для проверки.

Пример:

```bash
python baseline.py predict --data-dir /path/to/your_dir \
  --test-csv stage1/test.csv \
  --checkpoint checkpoints/baseline_unet_resnet34.best.pth
```

Формат файлов:
- `test.csv`: колонка `img_path`.
- `submission.csv`: колонки `img_path`, `prediction_path` (путь к предсказанной маске).

## VRAM

Ориентир: **~4.5 GB** при дефолтах (`batch_size=32`, `img_size=256`). 
