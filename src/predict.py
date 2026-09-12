"""Инференс на тестовой выборке и сборка посылки submission.zip.

Запуск из корня проекта:
    python -m src.predict --checkpoint outputs/checkpoints/<имя>_best.pth \
        --threshold-json outputs/checkpoints/<имя>_threshold.json
"""

import argparse
import csv
import json
import zipfile
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from . import config
from .data import read_test_csv
from .model import ManipulationUnet, check_flops


def parse_args():
    parser = argparse.ArgumentParser(description="Инференс и сборка посылки")
    parser.add_argument("--data-dir", type=str, default=str(config.TEST_DATA_DIR))
    parser.add_argument("--test-csv", type=str, default=str(config.TEST_CSV))
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--encoder", type=str, default="resnet34")
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--threshold-json", type=str, default=None,
                        help="Файл с подобранным порогом; перекрывает --threshold")
    parser.add_argument("--pred-dir", type=str, default=str(config.PREDICTIONS_DIR))
    parser.add_argument("--submission-csv", type=str,
                        default=str(config.OUTPUTS_DIR / "submission.csv"))
    parser.add_argument("--zip", type=str, default=str(config.OUTPUTS_DIR / "submission.zip"))
    return parser.parse_args()


class TestDataset(Dataset):
    """Тестовый датасет: возвращает тензор изображения и его исходный размер."""

    def __init__(self, paths, data_dir, img_size):
        self.paths = paths
        self.data_dir = Path(data_dir)
        self.img_size = img_size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        rel = self.paths[idx]
        img = cv2.imread(str(self.data_dir / rel), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Не удалось прочитать изображение: {self.data_dir / rel}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)
        return torch.from_numpy(img), h, w, rel


def write_submission_zip(submission_csv: Path, pred_dir: Path, zip_path: Path):
    """Упаковывает submission.csv и predictions/ в один архив."""
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(submission_csv, "submission.csv")
        for png in sorted(pred_dir.glob("*.png")):
            zf.write(png, f"predictions/{png.name}")
    print(f"посылка сохранена -> {zip_path}")


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    threshold = args.threshold
    if args.threshold_json:
        with open(args.threshold_json, encoding="utf-8") as f:
            threshold = json.load(f)["threshold"]
    print(f"порог бинаризации: {threshold:.3f}")

    paths = read_test_csv(args.test_csv)
    print(f"тестовых изображений: {len(paths)}")

    model = ManipulationUnet(args.encoder)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state["model"])
    model.to(device)
    model.eval()

    # Контролируем лимит FLOPs на случай, если img_size не совпадает с обучением.
    check_flops(model, args.img_size, device)

    dataset = TestDataset(paths, args.data_dir, args.img_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=False)

    pred_dir = Path(args.pred_dir)
    pred_dir.mkdir(parents=True, exist_ok=True)

    sub_rows = []
    for images, heights, widths, rels in tqdm(loader, desc="predict"):
        images = images.to(device)
        with torch.no_grad():
            logits = model(images)
        probs = torch.sigmoid(logits).cpu().numpy()[:, 0]

        for i in range(probs.shape[0]):
            h, w = int(heights[i]), int(widths[i])
            rel = rels[i]
            binary = (probs[i] >= threshold).astype(np.uint8) * 255
            # Возвращаем маску к исходному разрешению изображения.
            binary = cv2.resize(binary, (w, h), interpolation=cv2.INTER_NEAREST)

            out_name = Path(rel).stem + "_pred.png"
            cv2.imwrite(str(pred_dir / out_name), binary)
            sub_rows.append({
                "img_path": rel,
                "prediction_path": f"predictions/{out_name}",
            })

    submission_csv = Path(args.submission_csv)
    with open(submission_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["img_path", "prediction_path"])
        writer.writeheader()
        writer.writerows(sub_rows)
    print(f"submission.csv -> {submission_csv} ({len(sub_rows)} строк)")

    write_submission_zip(submission_csv, pred_dir, Path(args.zip))


if __name__ == "__main__":
    main()
