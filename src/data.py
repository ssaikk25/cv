"""Чтение данных, разбиение без утечки и датасет для обучения.

Ключевая особенность разметки: одна и та же маска может встречаться в
нескольких строках train.csv (одну маску используют до пяти разных изменённых
изображений). Поэтому обычный случайный сплит даст утечку валидации, и мы
разбиваем выборку по группам масок целиком.
"""

import csv
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def read_train_csv(csv_path):
    """Читает train.csv и возвращает список строк с путями к изображениям и маске."""
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append({
                "chng": row["chng_img_path"].strip(),
                "gt": row["gt_path"].strip(),
                "src": (row.get("orgl_img_path") or "").strip(),
            })
    return rows


def read_test_csv(csv_path):
    """Читает test.csv и возвращает список путей к тестовым изображениям."""
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row["img_path"].strip())
    return rows


def group_split(rows, val_ratio, seed):
    """Разбивает строки так, чтобы все копии одной маски попали в одну часть.

    Группа строки — это путь к маске gt_path. Одинаковые маски всегда оказываются
    либо в обучении, либо в валидации, что исключает утечку.
    """
    rng = random.Random(seed)
    groups = sorted({r["gt"] for r in rows})
    rng.shuffle(groups)
    n_val = max(1, int(len(groups) * val_ratio))
    val_groups = set(groups[:n_val])

    train = [r for r in rows if r["gt"] not in val_groups]
    val = [r for r in rows if r["gt"] in val_groups]
    return train, val


def load_mask_stats(cache_path):
    """Загружает кэш доли площади правки для каждой маски (см. src/prepare.py)."""
    with open(cache_path, encoding="utf-8") as f:
        return json.load(f)


def split_rows(rows, mask_stats, val_ratio, seed):
    """Делит строки на позитивы и негативы и делает групповой сплит по маске.

    Позитивы — строки, где маска непустая. Негативы — строки с пустой маской
    (это "чистые" изображения, прошедшие тот же пайплайн редактирования, но без
    фактических изменений). Обе группы разбиваются по gt_path, чтобы одинаковые
    маски не попадали одновременно в обучение и валидацию.
    """
    def is_positive(r):
        area = mask_stats.get(r["gt"])
        return area is None or area > 0.0

    pos_rows = [r for r in rows if is_positive(r)]
    neg_rows = [r for r in rows if not is_positive(r)]

    train_pos, val_pos = group_split(pos_rows, val_ratio, seed)
    train_neg, val_neg = group_split(neg_rows, val_ratio, seed)
    return train_pos, val_pos, train_neg, val_neg


class SegDataset(Dataset):
    """Датасет сегментации: изображение и маска.

    Чистые примеры (оригиналы без правок) задаются через mask=None и получают
    нулевую маску — так модель учится не давать ложных срабатываний, что важно
    для составляющей FPR в метрике AIC.
    """

    def __init__(self, samples, data_dir, img_size, train=True):
        self.samples = samples
        self.data_dir = Path(data_dir)
        self.img_size = img_size
        self.augment = train

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _load_rgb(path: Path) -> np.ndarray:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Не удалось прочитать изображение: {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    @staticmethod
    def _load_mask(path: Path) -> np.ndarray:
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Не удалось прочитать маску: {path}")
        # Возвращаем uint8 без бинаризации: порог применим после уменьшения,
        # чтобы не создавать большие float32-массивы в полном разрешении.
        return mask

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = self._load_rgb(self.data_dir / sample["img"])

        if sample["mask"] is None:
            mask = np.zeros(img.shape[:2], dtype=np.uint8)
        else:
            mask = self._load_mask(self.data_dir / sample["mask"])

        img, mask = self._preprocess(img, mask)

        return {
            "image": torch.from_numpy(img),
            "mask": torch.from_numpy(mask).unsqueeze(0),
            "is_neg": float(sample["mask"] is None),
        }

    def _preprocess(self, img, mask):
        img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 128).astype(np.float32)

        if self.augment:
            # Горизонтальный флип — безопасная геометрическая аугментация
            # для сегментации, маска переворачивается вместе с изображением.
            if random.random() < 0.5:
                img = np.ascontiguousarray(img[:, ::-1])
                mask = np.ascontiguousarray(mask[:, ::-1])
            # Повороты на 90 градусов. np.rot90 возвращает view с отрицательными
            # шагами, который torch.from_numpy не принимает, поэтому копируем в
            # непрерывный массив через ascontiguousarray.
            if random.random() < 0.5:
                k = random.randint(0, 3)
                img = np.ascontiguousarray(np.rot90(img, k))
                mask = np.ascontiguousarray(np.rot90(mask, k))

        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)
        return img, mask
