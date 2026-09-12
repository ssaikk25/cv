"""Разведочный анализ обучающей выборки (EDA).

Использует только numpy, PIL и matplotlib, поэтому запускается любым Python
с этими библиотеками. Сохраняет графики в outputs/figures.

Запуск из корня проекта: python src/eda.py
"""

import csv
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
TRAIN_CSV = ROOT / "train_stage1" / "stage1" / "train.csv"
TRAIN_DIR = ROOT / "train_stage1"
FIG_DIR = ROOT / "outputs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

N_SIZE_SAMPLE = 1500   # сколько изображений открыть для оценки размеров
N_MASK_SAMPLE = 2500   # сколько масок открыть для оценки доли правки


def main():
    rows = []
    with open(TRAIN_CSV, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append({
                "chng": r["chng_img_path"].strip(),
                "gt": r["gt_path"].strip(),
                "src": (r.get("orgl_img_path") or "").strip(),
            })

    n = len(rows)
    distinct_gt = len({r["gt"] for r in rows})
    distinct_src = len({r["src"] for r in rows if r["src"]})
    empty_src = sum(1 for r in rows if not r["src"])

    print(f"строк: {n}")
    print(f"уникальных масок (gt): {distinct_gt}  (в среднем {n / distinct_gt:.2f} строк на маску)")
    print(f"уникальных оригиналов (src): {distinct_src}")
    print(f"строк без оригинала: {empty_src} ({empty_src / n:.1%})")

    # Размеры изменённых изображений по случайной выборке.
    rng = random.Random(42)
    chng_sample = rng.sample([r["chng"] for r in rows], N_SIZE_SAMPLE)
    sizes = []
    for rel in chng_sample:
        try:
            with Image.open(TRAIN_DIR / rel) as im:
                sizes.append(im.size)  # (ширина, высота)
        except OSError:
            continue

    widths = np.asarray([s[0] for s in sizes])
    heights = np.asarray([s[1] for s in sizes])
    print("\nразмеры изменённых изображений (выборка):")
    print(f"  высота: min={heights.min()} p50={int(np.median(heights))} max={heights.max()}")
    print(f"  ширина: min={widths.min()} p50={int(np.median(widths))} max={widths.max()}")

    # Доля площади маски и наличие полностью пустых масок.
    mask_sample = rng.sample([r["gt"] for r in rows], N_MASK_SAMPLE)
    areas = []
    zero_masks = 0
    for rel in mask_sample:
        try:
            with Image.open(TRAIN_DIR / rel) as im:
                m = np.asarray(im.convert("L"))
        except OSError:
            continue
        area = float((m > 128).mean())
        areas.append(area)
        if area == 0.0:
            zero_masks += 1

    areas = np.asarray(areas)
    print(f"\nдоля площади правки в маске (выборка {len(areas)} масок):")
    print(f"  min={areas.min():.4f} p25={np.percentile(areas, 25):.4f} "
          f"p50={np.percentile(areas, 50):.4f} p75={np.percentile(areas, 75):.4f} max={areas.max():.4f}")
    print(f"  полностью пустых масок в выборке: {zero_masks}")
    print(f"  масок с площадью < 1% (мелкие правки): {(areas < 0.01).mean():.1%}")

    # Графики для наглядности.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].hist(np.log10(areas + 1e-6), bins=40, color="#4c72b0", alpha=0.85)
    axes[0].set_xlabel("log10(доля площади правки)")
    axes[0].set_ylabel("число масок")
    axes[0].set_title("Распределение доли изменённой области")

    axes[1].scatter(widths, heights, s=6, alpha=0.4, color="#dd8452")
    axes[1].set_xlabel("ширина, px")
    axes[1].set_ylabel("высота, px")
    axes[1].set_title("Размеры изменённых изображений")

    fig.tight_layout()
    out = FIG_DIR / "eda_overview.png"
    fig.savefig(out, dpi=120)
    print(f"\nграфики сохранены в {out}")


if __name__ == "__main__":
    main()
