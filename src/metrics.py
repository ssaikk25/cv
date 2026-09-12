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


def select_threshold(probs_pos, masks_pos, probs_neg, thresholds):
    """Подбирает порог бинаризации, максимизирующий AIC.

    probs_pos и masks_pos — списки массивов вероятностей и соответствующих
    истинных масок для изменённых изображений. probs_neg — список массивов
    вероятностей для чистых изображений. Все массивы numpy с одинаковым
    разрешением. Возвращает (порог, aic, dice_pos, fpr_neg).
    """
    best = (0.5, -1.0, 0.0, 1.0)
    for threshold in thresholds:
        dices = [dice_score(p >= threshold, m) for p, m in zip(probs_pos, masks_pos)]
        dice_pos = float(np.mean(dices)) if dices else 0.0

        areas = [(p >= threshold).sum() / p.size for p in probs_neg]
        fpr = fpr_negative(np.asarray(areas))
        score = aic_score(dice_pos, fpr)

        if score > best[1]:
            best = (threshold, score, dice_pos, fpr)

    return best
