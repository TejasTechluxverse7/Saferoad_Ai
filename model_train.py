"""
AegisRoad AI – Model Training Pipeline
========================================
Trains a YOLOv8 accident-detection model using the Roboflow
accident-detection dataset and saves the final weights as:

    models/aegis_model.pt          (primary output)
    runs/detect/aegis_model/weights/best.pt  (ultralytics default)

Usage:
    python model_train.py
    python model_train.py --epochs 50 --batch 8 --device cpu
    python model_train.py --epochs 25 --device 0   # GPU 0

Requirements:
    pip install ultralytics

Dataset:
    Roboflow: https://universe.roboflow.com/yolovideos/accident-detection-bcc2v
    The dataset YAML is fetched online; no local download needed.
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


# ─── Dataset URL ──────────────────────────────────────────────────────────────

DATASET_URL = (
    "https://universe.roboflow.com/yolovideos/accident-detection-bcc2v/dataset/1"
    "/download/yolov8"
)

# Alternative YAML paths to try (Roboflow may change export URLs over time)
YAML_CANDIDATES = [
    "https://universe.roboflow.com/yolovideos/accident-detection-bcc2v/data.yaml",
    "accident-detection-bcc2v-1/data.yaml",   # local fallback after export
]

BASE_MODEL       = "yolov8n.pt"       # nano – fast and lightweight
OUTPUT_DIR       = "models"
OUTPUT_NAME      = "aegis_model"
OUTPUT_PATH      = f"{OUTPUT_DIR}/aegis_model.pt"

DEFAULT_EPOCHS   = 25
DEFAULT_IMGSZ    = 640
DEFAULT_BATCH    = 16
DEFAULT_PATIENCE = 10                  # early stopping


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _find_best_weights(run_name: str) -> Path | None:
    """Search for best.pt produced by the training run."""
    candidates = [
        Path(f"runs/detect/{run_name}/weights/best.pt"),
        Path(f"runs/detect/{run_name}2/weights/best.pt"),
        Path(f"runs/detect/{run_name}3/weights/best.pt"),
    ]
    for c in candidates:
        if c.exists():
            return c
    # Wildcard search
    for p in Path("runs/detect").glob(f"{run_name}*/weights/best.pt"):
        return p
    return None


def _copy_weights(best_pt: Path) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    shutil.copy2(best_pt, OUTPUT_PATH)
    print(f"\n✅ Model saved → {OUTPUT_PATH}")
    print(f"   (copy of {best_pt})")


def _download_roboflow_dataset() -> str | None:
    """
    Try to download dataset via roboflow SDK.
    Returns path to data.yaml, or None if unavailable.
    """
    try:
        from roboflow import Roboflow  # type: ignore
    except ImportError:
        return None

    api_key = os.environ.get("ROBOFLOW_API_KEY", "")
    if not api_key:
        print("⚠️  ROBOFLOW_API_KEY not set. Skipping SDK download.")
        return None

    try:
        rf = Roboflow(api_key=api_key)
        project = rf.workspace("yolovideos").project("accident-detection-bcc2v")
        dataset = project.version(1).download("yolov8")
        yaml_path = Path(dataset.location) / "data.yaml"
        if yaml_path.exists():
            return str(yaml_path)
    except Exception as e:
        print(f"⚠️  Roboflow SDK download failed: {e}")
    return None


# ─── Main training function ───────────────────────────────────────────────────

def train_model(
    epochs: int   = DEFAULT_EPOCHS,
    imgsz: int    = DEFAULT_IMGSZ,
    batch: int    = DEFAULT_BATCH,
    device: str   = "0" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu",
    patience: int = DEFAULT_PATIENCE,
    resume: bool  = False,
) -> None:
    """
    Full training pipeline:
      1. Attempt Roboflow SDK dataset download (needs ROBOFLOW_API_KEY).
      2. Fall back to direct URL (online training, Ultralytics handles download).
      3. Train YOLOv8n on the dataset.
      4. Copy best.pt → models/aegis_model.pt.
    """
    from ultralytics import YOLO

    print("\n🛡️  AegisRoad AI – Model Training Pipeline")
    print("=" * 50)
    print(f"  Base model : {BASE_MODEL}")
    print(f"  Epochs     : {epochs}")
    print(f"  Image size : {imgsz}")
    print(f"  Batch size : {batch}")
    print(f"  Device     : {device}")
    print(f"  Output     : {OUTPUT_PATH}")
    print()

    # ── 1. Resolve dataset YAML path ──────────────────────────────
    yaml_path: str | None = None

    # Try local path first (already downloaded)
    for local in ["accident-detection-bcc2v-1/data.yaml",
                   "datasets/accident-detection/data.yaml"]:
        if Path(local).exists():
            yaml_path = local
            print(f"📁 Using local dataset: {local}")
            break

    # Try Roboflow SDK
    if yaml_path is None:
        yaml_path = _download_roboflow_dataset()
        if yaml_path:
            print(f"📥 Dataset downloaded via Roboflow SDK → {yaml_path}")

    # Fall back to URL
    if yaml_path is None:
        yaml_path = YAML_CANDIDATES[0]
        print(f"🌐 Using online dataset URL: {yaml_path}")
        print("   (Ultralytics will download and cache the dataset automatically)")

    # ── 2. Load base model ─────────────────────────────────────────
    print(f"\n⏳ Loading base model: {BASE_MODEL}")
    model = YOLO(BASE_MODEL)

    # ── 3. Train ───────────────────────────────────────────────────
    print(f"\n🚀 Starting training for {epochs} epochs…\n")
    results = model.train(
        data     = yaml_path,
        epochs   = epochs,
        imgsz    = imgsz,
        batch    = batch,
        device   = device,
        name     = OUTPUT_NAME,
        patience = patience,
        resume   = resume,
        # Performance options
        cache    = True,
        workers  = 4,
        # Augmentation (robustness for rain/night/fog)
        hsv_h    = 0.015,
        hsv_s    = 0.7,
        hsv_v    = 0.4,
        fliplr   = 0.5,
        mosaic   = 1.0,
        degrees  = 5.0,
    )

    print("\n✅ Training complete!")
    print(f"   Best mAP50: {getattr(results, 'maps', {}).get(0.5, 'N/A')}")

    # ── 4. Copy weights ────────────────────────────────────────────
    best_pt = _find_best_weights(OUTPUT_NAME)
    if best_pt:
        _copy_weights(best_pt)
    else:
        print("⚠️  Could not locate best.pt. Check runs/detect/ manually.")
        print(f"   Manually copy your best.pt to {OUTPUT_PATH}")

    print(f"\n🏁 Run inference with:")
    print(f"   python saferoad_main.py --source sample_videos/video1.mp4\n")


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="AegisRoad AI – YOLOv8 Training Pipeline"
    )
    ap.add_argument("--epochs",   type=int,  default=DEFAULT_EPOCHS)
    ap.add_argument("--imgsz",    type=int,  default=DEFAULT_IMGSZ)
    ap.add_argument("--batch",    type=int,  default=DEFAULT_BATCH)
    ap.add_argument("--device",   type=str,  default="cpu",
                    help="'cpu', '0', '0,1' (GPU index), or 'mps' (Apple Silicon)")
    ap.add_argument("--patience", type=int,  default=DEFAULT_PATIENCE,
                    help="Early stopping patience (epochs)")
    ap.add_argument("--resume",   action="store_true",
                    help="Resume interrupted training")
    args = ap.parse_args()

    train_model(
        epochs   = args.epochs,
        imgsz    = args.imgsz,
        batch    = args.batch,
        device   = args.device,
        patience = args.patience,
        resume   = args.resume,
    )
