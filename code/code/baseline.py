import argparse
import csv
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import segmentation_models_pytorch as smp

# TODO: change to your data directory
DATA_DIR = Path('')

SCRIPT_DIR = Path(__file__).resolve().parent
TRAIN_CSV = 'stage1/train.csv'
TEST_CSV = 'stage1/test.csv'


def resolve_path(path: str) -> Path:
    """Пути к данным — относительно DATA_DIR."""
    p = Path(path.strip().replace('\\', '/'))
    return p if p.is_absolute() else DATA_DIR / p


def resolve_output_path(path: str) -> Path:
    """Артефакты (checkpoints, predictions, submission) — относительно папки скрипта."""
    p = Path(path.strip().replace('\\', '/'))
    return p if p.is_absolute() else SCRIPT_DIR / p


def save_train_config(args, ckpt_dir: Path):
    config = {k: getattr(args, k) for k in vars(args)}
    config['data_dir'] = str(DATA_DIR)
    config['script_dir'] = str(SCRIPT_DIR)
    path = ckpt_dir / 'config_train.json'
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f'config saved -> {path}')


def load_rgb(path: Path) -> np.ndarray:
    img = np.asarray(Image.open(path).convert('RGB'))
    return img


def load_mask_binary(path: Path) -> np.ndarray:
    m = np.asarray(Image.open(path))
    if m.ndim == 3:
        m = m[:, :, 0]
    return (m > 128).astype(np.float32)


def preprocess(img: np.ndarray, mask, size: int, train: bool):
    h, w = img.shape[:2]
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    if train and random.random() < 0.5:
        img = np.ascontiguousarray(img[:, ::-1])
        if mask is not None:
            mask = np.ascontiguousarray(mask[:, ::-1])
    if mask is not None:
        mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)
    img = img.astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)
    return img, mask, (h, w)


class TrainDataset(Dataset):
    def __init__(self, rows, img_size: int, train: bool):
        self.rows = rows
        self.img_size = img_size
        self.train = train

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        img = load_rgb(resolve_path(row['chng']))
        mask = load_mask_binary(resolve_path(row['gt']))
        img, mask, _ = preprocess(img, mask, self.img_size, self.train)
        return {
            'image': torch.from_numpy(img),
            'mask': torch.from_numpy(mask).unsqueeze(0),
        }


class TestDataset(Dataset):
    def __init__(self, rows, img_size: int):
        self.rows = rows
        self.img_size = img_size

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        chng_rel = row['chng']
        img = load_rgb(resolve_path(chng_rel))
        img, _, orig_hw = preprocess(img, None, self.img_size, train=False)
        return {
            'image': torch.from_numpy(img),
            'chng_path': chng_rel,
            'orig_h': orig_hw[0],
            'orig_w': orig_hw[1],
        }


def read_train_csv(csv_path: Path):
    rows = []
    with open(csv_path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            rows.append({
                'chng': row['chng_img_path'].strip(),
                'gt': row['gt_path'].strip(),
            })
    return rows


def read_test_csv(csv_path: Path):
    rows = []
    with open(csv_path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            img_path = row.get('img_path')
            if img_path is None:
                img_path = row.get('chng_img_path')
            if img_path is None:
                raise ValueError(f'test csv must contain "img_path" (or legacy "chng_img_path"), got: {row.keys()}')
            rows.append({'chng': img_path.strip()})
    return rows


def split_train_val(rows, val_ratio: float, seed: int):
    idx = list(range(len(rows)))
    rng = random.Random(seed)
    rng.shuffle(idx)
    n_val = max(1, int(len(rows) * val_ratio))
    val_set = set(idx[:n_val])
    train_rows = [rows[i] for i in idx if i not in val_set]
    val_rows = [rows[i] for i in idx if i in val_set]
    return train_rows, val_rows


def build_model(encoder: str = 'resnet34'):
    return smp.Unet(
        encoder_name=encoder,
        encoder_weights='imagenet',
        in_channels=3,
        classes=1,
        activation=None,
    )


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total = 0.0
    for batch in tqdm(loader, desc='train', leave=False):
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / max(len(loader), 1)


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total = 0.0
    for batch in tqdm(loader, desc='val', leave=False):
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        logits = model(images)
        total += criterion(logits, masks).item()
    return total / max(len(loader), 1)


def run_train(args):
    rows = read_train_csv(resolve_path(args.train_csv))
    train_rows, val_rows = split_train_val(rows, args.val_ratio, args.seed)
    print(f'train={len(train_rows)} val={len(val_rows)}')

    train_ds = TrainDataset(train_rows, args.img_size, train=True)
    val_ds = TrainDataset(val_rows, args.img_size, train=False)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(args.encoder).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val = float('inf')
    ckpt_dir = resolve_output_path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / args.checkpoint_name
    save_train_config(args, ckpt_dir)

    for epoch in range(1, args.epochs + 1):
        tr_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        va_loss = eval_epoch(model, val_loader, criterion, device)
        print(f'epoch {epoch}/{args.epochs} train_loss={tr_loss:.4f} val_loss={va_loss:.4f}')
        state = {
            'epoch': epoch,
            'model': model.state_dict(),
            'encoder': args.encoder,
            'img_size': args.img_size,
            'val_loss': va_loss,
        }
        torch.save(state, ckpt_path.with_suffix('.last.pth'))
        if va_loss < best_val:
            best_val = va_loss
            torch.save(state, ckpt_path.with_suffix('.best.pth'))
            print(f'  saved best -> {ckpt_path.with_suffix(".best.pth")}')


@torch.no_grad()
def run_predict(args):
    rows = read_test_csv(resolve_path(args.test_csv))
    test_ds = TestDataset(rows, args.img_size)
    loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(args.encoder).to(device)
    ckpt = torch.load(resolve_output_path(args.checkpoint), map_location=device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    pred_dir_name = Path(args.pred_dir).as_posix().rstrip('/')
    pred_dir = resolve_output_path(args.pred_dir)
    pred_dir.mkdir(parents=True, exist_ok=True)

    sub_rows = []
    for batch in tqdm(loader, desc='predict'):
        images = batch['image'].to(device)
        logits = model(images)
        probs = torch.sigmoid(logits).cpu().numpy()[:, 0]

        for i in range(probs.shape[0]):
            chng_rel = batch['chng_path'][i]
            oh = int(batch['orig_h'][i])
            ow = int(batch['orig_w'][i])
            pred = (probs[i] > 0.5).astype(np.uint8) * 255
            pred = cv2.resize(pred, (ow, oh), interpolation=cv2.INTER_NEAREST)

            out_name = Path(chng_rel).stem + '_pred.png'
            out_abs = pred_dir / out_name
            cv2.imwrite(str(out_abs), pred)
            sub_rows.append({
                'img_path': chng_rel,
                'prediction_path': f'{pred_dir_name}/{out_name}',
            })

    sub_path = resolve_output_path(args.submission_csv)
    with open(sub_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['img_path', 'prediction_path'])
        writer.writeheader()
        writer.writerows(sub_rows)
    print(f'submission: {sub_path} ({len(sub_rows)} rows)')


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='cmd', required=True)

    p_train = sub.add_parser('train')
    p_train.add_argument('--data-dir', type=str, default=None, help='override DATA_DIR')
    p_train.add_argument('--train-csv', type=str, default=TRAIN_CSV)
    p_train.add_argument('--val-ratio', type=float, default=0.1)
    p_train.add_argument('--seed', type=int, default=42)
    p_train.add_argument('--img-size', type=int, default=256)
    p_train.add_argument('--batch-size', type=int, default=32)
    p_train.add_argument('--epochs', type=int, default=15)
    p_train.add_argument('--lr', type=float, default=1e-4)
    p_train.add_argument('--encoder', type=str, default='resnet34')
    p_train.add_argument('--num-workers', type=int, default=4)
    p_train.add_argument('--checkpoint-dir', type=str, default='checkpoints')
    p_train.add_argument('--checkpoint-name', type=str, default='baseline_unet_resnet34')

    p_pred = sub.add_parser('predict')
    p_pred.add_argument('--data-dir', type=str, default=None, help='override DATA_DIR')
    p_pred.add_argument('--test-csv', type=str, default=TEST_CSV)
    p_pred.add_argument('--checkpoint', type=str,
                        default='checkpoints/baseline_unet_resnet34.best.pth')
    p_pred.add_argument('--img-size', type=int, default=256)
    p_pred.add_argument('--batch-size', type=int, default=32)
    p_pred.add_argument('--encoder', type=str, default='resnet34')
    p_pred.add_argument('--num-workers', type=int, default=4)
    p_pred.add_argument('--pred-dir', type=str, default='predictions')
    p_pred.add_argument('--submission-csv', type=str, default='submission.csv')

    args = parser.parse_args()
    global DATA_DIR
    if getattr(args, 'data_dir', None):
        DATA_DIR = Path(args.data_dir)

    if args.cmd == 'train':
        run_train(args)
    elif args.cmd == 'predict':
        run_predict(args)


if __name__ == '__main__':
    main()
