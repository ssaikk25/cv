"""Метрика конкурса AIC Score и вспомогательные функции.

AIC = 2 * Dice_pos * (1 - FPR_neg) / (Dice_pos + (1 - FPR_neg)),

где Dice_pos — средний Dice по изменённым изображениям, а FPR_neg — доля
чистых изображений, на которых предсказанная маска занимает не меньше
1% площади кадра. Это гармоническое среднее Dice и (1 - FPR): решение
сильно штрафуется, если хотя бы одна из двух составляющих провалена.
"""

import numpy as np

# Порог площади кадра, начиная с которого изображение считается ложной тревогой.
FPR_AREA_THRESHOLD = 0.01


def dice_score(pred: np.ndarray, gt: np.ndarray) -> float:
    """Dice между двумя бинарными масками со значениями 0 и 1."""
    pred = np.asarray(pred).astype(bool)
    gt = np.asarray(gt).astype(bool)
    inter = np.logical_and(pred, gt).sum()
    denom = pred.sum() + gt.sum()
    return float(2.0 * inter / (denom + 1e-6))


def fpr_negative(areas: np.ndarray) -> float:
    """Доля чистых изображений с площадью предсказания >= 1% кадра."""
    return float((np.asarray(areas) >= FPR_AREA_THRESHOLD).mean())


def aic_score(dice_pos: float, fpr_neg: float) -> float:
    """Итоговая метрика AIC как гармоническое среднее Dice и (1 - FPR)."""
    one_minus_fpr = 1.0 - fpr_neg
    return float(2.0 * dice_pos * one_minus_fpr / (dice_pos + one_minus_fpr + 1e-12))


def dice_from_hist(hist_all, hist_pos, n_pos, threshold):
    """Dice одного изображения по гистограммам вероятностей.

    hist_all — гистограмма (256 бинов) вероятностей по всем пикселям,
    hist_pos — по пикселям переднего плана маски, n_pos — их количество.
    Такой способ не хранит в памяти полную карту вероятностей.
    """
    tbin = int(threshold * 255)
    pred_count = float(np.asarray(hist_all[tbin:]).sum())
    inter = float(np.asarray(hist_pos[tbin:]).sum())
    return 2.0 * inter / (pred_count + n_pos + 1e-6)


def area_from_hist(hist_all, total_pixels, threshold):
    """Доля площади предсказанной маски при заданном пороге."""
    tbin = int(threshold * 255)
    return float(np.asarray(hist_all[tbin:]).sum()) / total_pixels
