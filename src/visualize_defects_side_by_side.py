from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "src" / "custom_steel_detector_grid14_mejorado.pth"
TEST_ROOT = ROOT / "data" / "test"
OUTPUT_DIR = ROOT / "resultados" / "detector_mejorado"
OUTPUT_GRID = OUTPUT_DIR / "side_by_side_por_defecto.png"

CLASS_NAMES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, use_pool: bool = True) -> None:
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        ]
        if use_pool:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + identity
        return self.act(out)


class CustomSteelDetectorImproved(nn.Module):
    def __init__(self, s: int = 14, num_classes: int = 6) -> None:
        super().__init__()
        self.s = s
        self.num_classes = num_classes
        self.backbone = nn.Sequential(
            ConvBlock(3, 32, use_pool=True),
            ConvBlock(32, 64, use_pool=True),
            ConvBlock(64, 128, use_pool=True),
            ConvBlock(128, 256, use_pool=True),
            ConvBlock(256, 512, use_pool=False),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            ResidualBlock(512),
        )
        self.head = nn.Conv2d(512, 5 + num_classes, kernel_size=1, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.head(x)
        return x.permute(0, 2, 3, 1)


def _box_iou(box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter_area
    return 0.0 if union <= 0 else inter_area / union


def _nms_per_class(dets: List[dict], iou_thr: float = 0.30) -> List[dict]:
    kept: List[dict] = []
    by_class = {}
    for det in dets:
        by_class.setdefault(det["class_id"], []).append(det)

    for class_dets in by_class.values():
        class_dets = sorted(class_dets, key=lambda x: x["conf"], reverse=True)
        while class_dets:
            best = class_dets.pop(0)
            kept.append(best)
            remain = []
            for det in class_dets:
                if _box_iou(best["box"], det["box"]) < iou_thr:
                    remain.append(det)
            class_dets = remain

    return sorted(kept, key=lambda x: x["conf"], reverse=True)


def decode_predictions(
    pred_grid: torch.Tensor,
    threshold: float,
    s: int,
    image_w: int,
    image_h: int,
    nms_iou_thr: float = 0.30,
) -> List[dict]:
    dets: List[dict] = []
    pred_grid = pred_grid.detach().cpu()

    for i in range(s):
        for j in range(s):
            conf = torch.sigmoid(pred_grid[i, j, 0]).item()
            if not np.isfinite(conf) or conf < threshold:
                continue

            x_cell = torch.sigmoid(pred_grid[i, j, 1]).item()
            y_cell = torch.sigmoid(pred_grid[i, j, 2]).item()
            w_norm = torch.sigmoid(pred_grid[i, j, 3]).item()
            h_norm = torch.sigmoid(pred_grid[i, j, 4]).item()

            if not (np.isfinite(x_cell) and np.isfinite(y_cell) and np.isfinite(w_norm) and np.isfinite(h_norm)):
                continue

            class_logits = pred_grid[i, j, 5:]
            if not torch.isfinite(class_logits).all().item():
                continue

            class_id = int(torch.argmax(class_logits).item())
            x_center = (j + x_cell) / s
            y_center = (i + y_cell) / s

            x1 = int((x_center - w_norm / 2.0) * image_w)
            y1 = int((y_center - h_norm / 2.0) * image_h)
            x2 = int((x_center + w_norm / 2.0) * image_w)
            y2 = int((y_center + h_norm / 2.0) * image_h)

            x1 = max(0, min(image_w - 1, x1))
            y1 = max(0, min(image_h - 1, y1))
            x2 = max(0, min(image_w - 1, x2))
            y2 = max(0, min(image_h - 1, y2))
            if x2 <= x1 or y2 <= y1:
                continue

            dets.append({"conf": float(conf), "class_id": class_id, "box": (x1, y1, x2, y2)})

    return _nms_per_class(dets, iou_thr=nms_iou_thr)


def read_gt_boxes(image_path: Path, class_names: Sequence[str]) -> List[dict]:
    label_path = image_path.with_suffix(".txt")
    if not label_path.exists():
        return []

    try:
        with Image.open(image_path) as img:
            w, h = img.size
    except Exception:
        return []

    out: List[dict] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cls, xc, yc, bw, bh = map(float, parts)
        class_id = int(cls)
        if class_id < 0 or class_id >= len(class_names):
            continue

        x1 = int((xc - bw / 2.0) * w)
        y1 = int((yc - bh / 2.0) * h)
        x2 = int((xc + bw / 2.0) * w)
        y2 = int((yc + bh / 2.0) * h)

        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(0, min(w - 1, x2))
        y2 = max(0, min(h - 1, y2))
        if x2 <= x1 or y2 <= y1:
            continue

        out.append({"class_id": class_id, "box": (x1, y1, x2, y2)})

    return out


def _load_rgb_image(image_path: Path) -> np.ndarray:
    with Image.open(image_path) as img:
        return np.asarray(img.convert("RGB"))


def _to_pil_rgb(array_rgb: np.ndarray) -> Image.Image:
    return Image.fromarray(array_rgb.astype(np.uint8), mode="RGB")


def draw_boxes(img_rgb: np.ndarray, boxes: Sequence[dict], class_names: Sequence[str], color=(0, 255, 0)) -> np.ndarray:
    pil_img = _to_pil_rgb(img_rgb)
    draw = ImageDraw.Draw(pil_img)
    font = ImageFont.load_default()
    for box_info in boxes:
        x1, y1, x2, y2 = box_info["box"]
        class_id = int(box_info["class_id"])
        class_name = class_names[class_id] if class_id < len(class_names) else str(class_id)
        conf = box_info.get("conf")
        label = class_name if conf is None else f"{class_name}:{conf:.2f}"
        draw.rectangle([x1, y1, x2, y2], outline=tuple(color), width=2)
        text_y = max(0, y1 - 12)
        draw.text((x1, text_y), label, fill=tuple(color), font=font)
    return np.asarray(pil_img)


def _make_pair_image(left_rgb: np.ndarray, right_rgb: np.ndarray, title_left: str, title_right: str) -> Image.Image:
    left_img = _to_pil_rgb(left_rgb)
    right_img = _to_pil_rgb(right_rgb)
    gap = 24
    pad = 16
    title_h = 28

    width = left_img.width + right_img.width + gap + pad * 2
    height = max(left_img.height, right_img.height) + pad * 2 + title_h
    canvas = Image.new("RGB", (width, height), (245, 246, 248))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    x_left = pad
    y_img = pad + title_h
    x_right = pad + left_img.width + gap

    canvas.paste(left_img, (x_left, y_img))
    canvas.paste(right_img, (x_right, y_img))
    draw.text((x_left, pad), title_left, fill=(25, 25, 25), font=font)
    draw.text((x_right, pad), title_right, fill=(25, 25, 25), font=font)
    return canvas


def collect_one_image_per_class(test_root: Path, class_names: Sequence[str]) -> List[Tuple[str, Path]]:
    valid_ext = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    selected: List[Tuple[str, Path]] = []

    for class_name in class_names:
        class_dir = test_root / class_name
        candidates = []
        if class_dir.exists() and class_dir.is_dir():
            candidates = sorted([p for p in class_dir.rglob("*") if p.is_file() and p.suffix.lower() in valid_ext])
        if not candidates and test_root.exists():
            candidates = sorted(
                [p for p in test_root.rglob("*") if p.is_file() and p.suffix.lower() in valid_ext and class_name in p.parts]
            )

        if not candidates:
            print(f"[WARN] No se encontro una imagen para la clase: {class_name}")
            continue

        selected.append((class_name, candidates[0]))

    return selected


def load_model(weights: Path) -> Tuple[CustomSteelDetectorImproved, Sequence[str], int, int]:
    ckpt = torch.load(weights, map_location=DEVICE)
    class_names = ckpt.get("class_names", CLASS_NAMES)
    s = int(ckpt.get("s", 14))
    num_classes = int(ckpt.get("num_classes", len(class_names)))
    image_size = int(ckpt.get("image_size", 224))

    model = CustomSteelDetectorImproved(s=s, num_classes=num_classes).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, class_names, s, image_size


def predict_boxes(
    model: CustomSteelDetectorImproved,
    image_path: Path,
    threshold: float,
    nms_iou_thr: float,
    s: int,
    image_size: int,
) -> List[dict]:
    try:
        with Image.open(image_path) as img:
            raw = img.convert("RGB")
    except Exception:
        return []

    w, h = raw.size
    resized = raw.resize((image_size, image_size), resample=Image.Resampling.BILINEAR)
    resized = np.asarray(resized, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    resized = (resized - mean) / std

    x = torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        pred = model(x)[0]

    dets = decode_predictions(pred, threshold, s, image_size, image_size, nms_iou_thr=nms_iou_thr)
    sx = w / float(image_size)
    sy = h / float(image_size)

    preds: List[dict] = []
    for det in dets:
        x1, y1, x2, y2 = det["box"]
        preds.append(
            {
                "class_id": int(det["class_id"]),
                "conf": float(det["conf"]),
                "box": (int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy)),
            }
        )
    return preds


def build_report_grid(weights: Path = WEIGHTS, test_root: Path = TEST_ROOT) -> Path:
    if not weights.exists():
        raise FileNotFoundError(f"No se encuentra el checkpoint: {weights}")
    if not test_root.exists():
        raise FileNotFoundError(f"No se encuentra el directorio de test: {test_root}")

    model, class_names, s, image_size = load_model(weights)
    selected = collect_one_image_per_class(test_root, class_names)
    if not selected:
        raise RuntimeError("No se encontraron imágenes para visualizar.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pair_images: List[Tuple[str, Image.Image]] = []
    for row, (class_name, image_path) in enumerate(selected):
        raw = _load_rgb_image(image_path)
        if raw is None:
            continue

        gts = read_gt_boxes(image_path, class_names)
        preds = predict_boxes(model, image_path, threshold=0.64, nms_iou_thr=0.30, s=s, image_size=image_size)

        left = draw_boxes(raw, gts, class_names, color=(0, 220, 0))
        right = draw_boxes(raw, preds, class_names, color=(255, 200, 0))

        per_class_path = OUTPUT_DIR / f"{class_name}_side_by_side.png"
        pair_image = _make_pair_image(left, right, f"GT | {class_name} | {image_path.name}", f"Modelo | {class_name} | {len(preds)} detecciones")
        pair_image.save(per_class_path)
        pair_images.append((class_name, pair_image))
        print(f"[OK] Guardado: {per_class_path}")

    if not pair_images:
        raise RuntimeError("No se pudo construir ninguna imagen de salida.")

    spacing = 24
    grid_width = max(img.width for _, img in pair_images) + 32
    grid_height = sum(img.height for _, img in pair_images) + spacing * (len(pair_images) - 1) + 32
    grid = Image.new("RGB", (grid_width, grid_height), (255, 255, 255))
    y = 16
    for class_name, img in pair_images:
        x = (grid_width - img.width) // 2
        grid.paste(img, (x, y))
        y += img.height + spacing

    grid.save(OUTPUT_GRID)
    print(f"[OK] Guardado: {OUTPUT_GRID}")
    return OUTPUT_GRID


if __name__ == "__main__":
    build_report_grid()