"""Обучение модели и подбор порога бинаризации по метрике AIC.

Запуск из корня проекта: python -m src.train --epochs 8
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from . import config
from .data import SegDataset, load_mask_stats, read_train_csv, split_rows
from .metrics import aic_score, area_from_hist, dice_from_hist, fpr_negative
from .model import ManipulationUnet, check_flops


def parse_args():
    parser = argparse.ArgumentParser(description="Обучение модели сегментации области манипуляции")
    parser.add_argument("--data-dir", type=str, default=str(config.TRAIN_DATA_DIR))
    parser.add_argument("--train-csv", type=str, default=str(config.TRAIN_CSV))
    parser.add_argument("--mask-stats", type=str,
                        default=str(config.OUTPUTS_DIR / "mask_stats.json"))
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--encoder", type=str, default="resnet34")
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--neg-ratio", type=float, default=0.1,
                        help="Доля чистых примеров относительно обучающих позитивов")
    parser.add_argument("--dice-w", type=float, default=1.0,
                        help="Вес Dice-составляющей в loss (0 = только BCE)")
    parser.add_argument("--val-neg", type=int, default=1000,
                        help="Число чистых оригиналов в валидации для оценки FPR")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0,
                        help="Ограничение числа строк для быстрой проверки пайплайна")
    parser.add_argument("--checkpoint-dir", type=str, default=str(config.CHECKPOINT_DIR))
    parser.add_argument("--name", type=str, default="manip_unet_resnet34")
    parser.add_argument("--resume", action="store_true",
                        help="Возобновить обучение из {name}_last.pth, если он есть")
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Soft Dice loss: 1 - Dice. Напрямую оптимизирует метрику сегментации.

    BCE штрафует каждый пиксель независимо, а Dice — пересечение областей целиком,
    поэтому их сумма обычно даёт более чёткие маски, чем один BCE.
    """
    probs = torch.sigmoid(logits)
    num = 2.0 * (probs * targets).sum() + eps
    den = probs.sum() + targets.sum() + eps
    return 1.0 - num / den


def train_epoch(model, loader, criterion, optimizer, scaler, device, dice_w=0.0):
    model.train()
    total = 0.0
    for batch in tqdm(loader, desc="train", leave=False):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
            logits = model(images)
            loss = criterion(logits, masks)
            if dice_w > 0.0:
                loss = loss + dice_w * dice_loss(logits, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total += loss.item()
    return total / max(len(loader), 1)


@torch.no_grad()
def collect_predictions(model, loader, device):
    """Прогоняет валидацию и собирает компактные гистограммы вероятностей.

    Вместо полных карт храним по две гистограммы (256 бинов) на изображение:
    распределение вероятностей по всем пикселям и по пикселям переднего плана.
    Этого достаточно, чтобы точно посчитать Dice и площадь маски при любом
    пороге, не удерживая в памяти сотни мегабайт карт вероятностей.
    """
    model.eval()
    hist_all, hist_pos, n_pos, total, is_neg = [], [], [], [], []
    for batch in tqdm(loader, desc="eval", leave=False):
        images = batch["image"].to(device)
        logits = model(images)
        probs = torch.sigmoid(logits).cpu().numpy()[:, 0]
        masks = batch["mask"].cpu().numpy()[:, 0]
        flags = batch["is_neg"].cpu().numpy().astype(bool)

        # Квантуем вероятности в 256 бинов, чтобы гистограммы были компактными.
        quantized = np.clip((probs * 255.0), 0, 255).astype(np.uint8)

        for i in range(probs.shape[0]):
            flat = quantized[i].ravel()
            fg = (masks[i] > 0.5).ravel()
            hist_all.append(np.bincount(flat, minlength=256).astype(np.float64))
            hist_pos.append(np.bincount(flat[fg], minlength=256).astype(np.float64))
            n_pos.append(float(fg.sum()))
            total.append(float(fg.size))
            is_neg.append(bool(flags[i]))
    return hist_all, hist_pos, n_pos, total, is_neg


def evaluate(hist_all, hist_pos, n_pos, total, is_neg, threshold):
    """Считает Dice_pos, FPR_neg и AIC при заданном пороге по гистограммам."""
    pos_idx = [i for i, n in enumerate(is_neg) if not n]
    neg_idx = [i for i, n in enumerate(is_neg) if n]

    dices = [dice_from_hist(hist_all[i], hist_pos[i], n_pos[i], threshold) for i in pos_idx]
    dice_pos = float(np.mean(dices)) if dices else 0.0

    areas = [area_from_hist(hist_all[i], total[i], threshold) for i in neg_idx]
    fpr = fpr_negative(np.asarray(areas)) if neg_idx else 0.0
    return dice_pos, fpr, aic_score(dice_pos, fpr)


def select_threshold(hist_all, hist_pos, n_pos, total, is_neg, thresholds):
    """Подбирает порог бинаризации, максимизирующий AIC на валидации."""
    pos_idx = [i for i, n in enumerate(is_neg) if not n]
    neg_idx = [i for i, n in enumerate(is_neg) if n]

    best = (0.5, -1.0, 0.0, 1.0)
    for threshold in thresholds:
        dices = [dice_from_hist(hist_all[i], hist_pos[i], n_pos[i], threshold) for i in pos_idx]
        dice_pos = float(np.mean(dices)) if dices else 0.0

        areas = [area_from_hist(hist_all[i], total[i], threshold) for i in neg_idx]
        fpr = fpr_negative(np.asarray(areas)) if neg_idx else 0.0
        score = aic_score(dice_pos, fpr)

        if score > best[1]:
            best = (threshold, score, dice_pos, fpr)
    return best


def main():
    args = parse_args()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    rows = read_train_csv(args.train_csv)
    if args.limit:
        rows = random.Random(args.seed).sample(rows, min(args.limit, len(rows)))

    mask_stats = load_mask_stats(args.mask_stats)
    train_pos, val_pos, train_neg_mask, val_neg_mask = split_rows(
        rows, mask_stats, args.val_ratio, args.seed
    )
    print(f"строк: {len(rows)} | train_pos={len(train_pos)} val_pos={len(val_pos)} | "
          f"train_neg_mask={len(train_neg_mask)} val_neg_mask={len(val_neg_mask)}")

    # Дополнительные чистые примеры берём из оригиналов (src): в них нет правок,
    # поэтому истинная маска нулевая. Это второй источник негативов для FPR.
    src_paths = sorted({r["src"] for r in rows if r["src"]})
    random.Random(args.seed).shuffle(src_paths)
    n_val_neg_src = min(args.val_neg, len(src_paths) // 2)
    val_neg_src = src_paths[:n_val_neg_src]
    n_train_neg_src = min(len(src_paths) - n_val_neg_src, int(len(train_pos) * args.neg_ratio))
    train_neg_src = src_paths[n_val_neg_src:n_val_neg_src + n_train_neg_src]
    print(f"чистых оригиналов: train_neg_src={len(train_neg_src)} val_neg_src={len(val_neg_src)}")

    train_samples = [{"img": r["chng"], "mask": r["gt"]} for r in train_pos]
    train_samples += [{"img": r["chng"], "mask": None} for r in train_neg_mask]
    train_samples += [{"img": p, "mask": None} for p in train_neg_src]

    val_samples = [{"img": r["chng"], "mask": r["gt"]} for r in val_pos]
    val_samples += [{"img": r["chng"], "mask": None} for r in val_neg_mask]
    val_samples += [{"img": p, "mask": None} for p in val_neg_src]

    train_ds = SegDataset(train_samples, args.data_dir, args.img_size, train=True)
    val_ds = SegDataset(val_samples, args.data_dir, args.img_size, train=False)

    # pin_memory выключен: на машинах с небольшим объёмом ОЗУ pinned-буферы
    # CUDA заметно увеличивают commit-заряд и могут приводить к нехватке памяти.
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=False, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=False,
    )

    model = ManipulationUnet(args.encoder).to(device)
    check_flops(model, args.img_size, device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    last_path = ckpt_dir / f"{args.name}_last.pth"
    best_path = ckpt_dir / f"{args.name}_best.pth"

    # Возобновление: если просили и есть чекпоинт, стартуем с его эпохи.
    start_epoch = 0
    if args.resume and last_path.exists():
        state = torch.load(last_path, map_location=device, weights_only=True)
        model.load_state_dict(state["model"])
        start_epoch = int(state.get("epoch", 0))
        print(f"возобновление с эпохи {start_epoch} ({last_path.name})")

    # Фиксируем параметры запуска для воспроизводимости.
    run_config = vars(args).copy()
    run_config["device"] = device
    with open(ckpt_dir / f"{args.name}_config.json", "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2, ensure_ascii=False)

    best_aic = -1.0
    for epoch in range(start_epoch + 1, args.epochs + 1):
        tr_loss = train_epoch(model, train_loader, criterion, optimizer, scaler, device, args.dice_w)
        scheduler.step()

        hist_all, hist_pos, n_pos, total, is_neg = collect_predictions(model, val_loader, device)
        dice_pos, fpr_neg, aic = evaluate(hist_all, hist_pos, n_pos, total, is_neg, threshold=0.5)

        lr = scheduler.get_last_lr()[0]
        print(f"epoch {epoch}/{args.epochs} | loss={tr_loss:.4f} | "
              f"val_dice={dice_pos:.4f} fpr={fpr_neg:.4f} aic@0.5={aic:.4f} | lr={lr:.2e}")

        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "encoder": args.encoder,
            "img_size": args.img_size,
            "val_dice": dice_pos,
            "val_fpr": fpr_neg,
            "val_aic": aic,
            "threshold": None,
        }
        torch.save(state, last_path)
        if aic > best_aic:
            best_aic = aic
            torch.save(state, best_path)
            print(f"  сохранён лучший чекпоинт -> {best_path.name} (aic={aic:.4f})")

    # После обучения подбираем порог бинаризации по метрике AIC. Порог считаем
    # на лучшем чекпоинте, чтобы он точно соответствовал модели для инференса.
    best_state = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(best_state["model"])
    hist_all, hist_pos, n_pos, total, is_neg = collect_predictions(model, val_loader, device)
    best_thr, best_aic, best_dice, best_fpr = select_threshold(
        hist_all, hist_pos, n_pos, total, is_neg, np.arange(0.10, 0.96, 0.05)
    )

    threshold_info = {
        "threshold": float(best_thr),
        "aic": float(best_aic),
        "dice": float(best_dice),
        "fpr": float(best_fpr),
    }
    with open(ckpt_dir / f"{args.name}_threshold.json", "w", encoding="utf-8") as f:
        json.dump(threshold_info, f, indent=2, ensure_ascii=False)

    print(f"подбор порога: thr={best_thr:.2f} aic={best_aic:.4f} "
          f"dice={best_dice:.4f} fpr={best_fpr:.4f}")
    print("обучение завершено")


if __name__ == "__main__":
    main()
