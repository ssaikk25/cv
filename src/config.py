"""Пути к данным и выходным артефактам, общие для обучения и инференса."""

from pathlib import Path

# Корень проекта: каталог на уровень выше пакета src.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Обучающая выборка. Пути внутри train.csv (stage1/train/...) относительны этого каталога.
TRAIN_DATA_DIR = PROJECT_ROOT / "train_stage1"
TRAIN_CSV = TRAIN_DATA_DIR / "stage1" / "train.csv"

# Тестовая выборка. Пути внутри test.csv (test_stage1_img/...) относительны этого каталога.
TEST_DATA_DIR = PROJECT_ROOT / "test_stage1" / "test_stage1"
TEST_CSV = TEST_DATA_DIR / "test.csv"

# Куда складываем чекпоинты, предсказания и посылку. Каталог не попадает в git.
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
CHECKPOINT_DIR = OUTPUTS_DIR / "checkpoints"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"
FIGURES_DIR = OUTPUTS_DIR / "figures"

# Ограничение из правил конкурса: не более 100 GFLOPs на одно изображение.
# Считаются строгие FLOPs, как это делает torch.utils.flop_counter.FlopCounterMode.
MAX_GFLOPS = 100.0
