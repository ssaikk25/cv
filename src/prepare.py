"""Предобработка разметки: кэширует долю площади правки для каждой маски.

Нужно, чтобы отделить настоящие позитивы от "чистых" строк, где маска пустая.
Такие строки — это готовые негативные примеры (правок нет, значит ложных
срабатываний быть не должно), критичные для составляющей FPR метрики AIC.

Запуск из корня проекта: python -m src.prepare
Аргументы позволяют указать свои пути (например, на Kaggle).
"""

import argparse
import csv
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from . import config


def _area_of(item):
    """Возвращает (путь, доля площади правки) для одной маски."""
    data_dir, gt_rel = item
    try:
        with Image.open(Path(data_dir) / gt_rel) as im:
            mask = np.asarray(im.convert("L"))
        return gt_rel, float((mask > 128).mean())
    except Exception:
        return gt_rel, None


def parse_args():
    parser = argparse.ArgumentParser(description="Кэш статистики масок")
    parser.add_argument("--data-dir", type=str, default=str(config.TRAIN_DATA_DIR))
    parser.add_argument("--train-csv", type=str, default=str(config.TRAIN_CSV))
    parser.add_argument("--output", type=str,
                        default=str(config.OUTPUTS_DIR / "mask_stats.json"))
    return parser.parse_args()


def main():
    args = parse_args()
    gt_paths = []
    with open(args.train_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            gt_paths.append(row["gt_path"].strip())

    unique = sorted(set(gt_paths))
    print(f"уникальных масок: {len(unique)}")

    stats = {}
    done = 0
    with ThreadPoolExecutor(max_workers=8) as executor:
        for gt, area in executor.map(_area_of, [(args.data_dir, g) for g in unique]):
            stats[gt] = area
            done += 1
            if done % 10000 == 0:
                print(f"  обработано {done}/{len(unique)}")

    cache_path = Path(args.output)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(stats, f)

    n_empty = sum(1 for v in stats.values() if v is not None and v == 0.0)
    n_bad = sum(1 for v in stats.values() if v is None)
    print(f"пустых масок: {n_empty}, нечитаемых: {n_bad}")
    print(f"кэш сохранён в {cache_path}")


if __name__ == "__main__":
    main()
