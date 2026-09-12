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
from .metrics import aic_score, dice_score, fpr_negative, select_threshold
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
    parser.add_argument("--val-neg", type=int, default=1000,
                        help="Число чистых изображений в валидации для оценки FPR")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0,
                        help="Ограничение числа строк для быстрой проверки пайплайна")
    parser.add_argument("--checkpoint-dir", type=str, default=str(config.CHECKPOINT_DIR))
    parser.add_argument("--name", type=str, default="manip_unet_resnet34")
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    total = 0.0
    for batch in tqdm(loader, desc="train", leave=False):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
            logits = model(images)
            loss = criterion(logits, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total += loss.item()
    return total / max(len(loader), 1)


@torch.no_grad()
def collect_predictions(model, loader, device):
    """Прогоняет валидацию и возвращает вероятности, маски и флаг чистоты."""
    model.eval()
    probs, masks, is_neg = [], [], []
    for batch in tqdm(loader, desc="eval", leave=False):
        images = batch["image"].to(device)
        logits = model(images)
        p = torch.sigmoid(logits).cpu().numpy()[:, 0]
        m = batch["mask"].cpu().numpy()[:, 0]
        n = batch["is_neg"].cpu().numpy().astype(bool)

        probs.extend(p[i] for i in range(len(p)))
        masks.extend(m[i] for i in range(len(m)))
        is_neg.extend(bool(n[i]) for i in range(len(n)))
    return probs, masks, is_neg


def evaluate(probs, masks, is_neg, threshold):
    """Считает Dice_pos, FPR_neg и AIC при заданном пороге."""
    pos_probs = [p for p, n in zip(probs, is_neg) if not n]
    pos_masks = [m for m, n in zip(masks, is_neg) if not n]
    neg_probs = [p for p, n in zip(probs, is_neg) if n]

    dices = [dice_score(p >= threshold, m) for p, m in zip(pos_probs, pos_masks)]
    dice_pos = float(np.mean(dices)) if dices else 0.0

    areas = [(p >= threshold).sum() / p.size for p in neg_probs]
    fpr = fpr_negative(np.asarray(areas)) if neg_probs else 0.0
    return dice_pos, fpr, aic_score(dice_pos, fpr)


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

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
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

    # Фиксируем параметры запуска для воспроизводимости.
    run_config = vars(args).copy()
    run_config["device"] = device
    with open(ckpt_dir / f"{args.name}_config.json", "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2, ensure_ascii=False)

    best_aic = -1.0
    for epoch in range(1, args.epochs + 1):
        tr_loss = train_epoch(model, train_loader, criterion, optimizer, scaler, device)
        scheduler.step()

        probs, masks, is_neg = collect_predictions(model, val_loader, device)
        dice_pos, fpr_neg, aic = evaluate(probs, masks, is_neg, threshold=0.5)

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

    # После обучения подбираем порог бинаризации по метрике AIC на валидации.
    probs, masks, is_neg = collect_predictions(model, val_loader, device)
    pos_probs = [p for p, n in zip(probs, is_neg) if not n]
    pos_masks = [m for m, n in zip(masks, is_neg) if not n]
    neg_probs = [p for p, n in zip(probs, is_neg) if n]

    thresholds = np.arange(0.10, 0.96, 0.05)
    best_thr, best_aic, best_dice, best_fpr = select_threshold(
        pos_probs, pos_masks, neg_probs, thresholds
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
