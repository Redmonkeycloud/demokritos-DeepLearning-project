"""
Step 0: Setup folder structure and copy existing data files into workflows/iemocap_dl/.
Run once before any other DL script.
"""

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SRC_SPLITS = ROOT / "workflows" / "iemocap_80_20" / "features" / "splits" / "80_20"
SRC_FEATURES = ROOT / "workflows" / "iemocap_80_20" / "features" / "iemocap_features.csv"

DL_ROOT = ROOT / "workflows" / "iemocap_dl"

DIRS = [
    DL_ROOT / "features" / "splits" / "80_20",
    DL_ROOT / "spectrograms" / "train",
    DL_ROOT / "spectrograms" / "test",
    DL_ROOT / "models" / "mlp",
    DL_ROOT / "models" / "cnn",
    DL_ROOT / "models" / "cnn_lstm",
    DL_ROOT / "models" / "wav2vec2",
    DL_ROOT / "results" / "mlp",
    DL_ROOT / "results" / "cnn",
    DL_ROOT / "results" / "cnn_lstm",
    DL_ROOT / "results" / "wav2vec2",
    DL_ROOT / "results" / "noise_robustness",
]


def create_dirs():
    print("Creating folder structure...")
    for d in DIRS:
        d.mkdir(parents=True, exist_ok=True)
        print(f"  OK  {d.relative_to(ROOT)}")


def copy_data():
    print("\nCopying data files...")
    dst_splits = DL_ROOT / "features" / "splits" / "80_20"

    for fname in ("train.csv", "test.csv", "scaler.pkl"):
        src = SRC_SPLITS / fname
        dst = dst_splits / fname
        if not src.exists():
            print(f"  WARN  {fname} not found at {src}")
            continue
        if dst.exists():
            print(f"  SKIP  {fname} (already exists)")
        else:
            shutil.copy2(src, dst)
            print(f"  COPY  {fname}")

    dst_features = DL_ROOT / "features" / "iemocap_features.csv"
    if SRC_FEATURES.exists():
        if dst_features.exists():
            print(f"  SKIP  iemocap_features.csv (already exists)")
        else:
            shutil.copy2(SRC_FEATURES, dst_features)
            print(f"  COPY  iemocap_features.csv")
    else:
        print(f"  WARN  iemocap_features.csv not found at {SRC_FEATURES}")


def check_packages():
    print("\nChecking required packages...")
    required = {
        "torch": "torch",
        "transformers": "transformers",
        "audiomentations": "audiomentations",
        "sklearn": "scikit-learn",
        "librosa": "librosa",
        "pandas": "pandas",
        "numpy": "numpy",
    }
    missing = []
    for import_name, pkg_name in required.items():
        try:
            __import__(import_name)
            print(f"  OK  {pkg_name}")
        except ImportError:
            print(f"  MISSING  {pkg_name}  ->  pip install {pkg_name}")
            missing.append(pkg_name)

    return missing


def check_gpu():
    print("\nChecking GPU...")
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            print(f"  GPU available: {name}")
        else:
            print("  No CUDA GPU detected — training will run on CPU (wav2vec2 will be slow)")
    except ImportError:
        print("  torch not installed, skipping GPU check")


def print_summary():
    print("\n" + "=" * 60)
    train_csv = DL_ROOT / "features" / "splits" / "80_20" / "train.csv"
    test_csv = DL_ROOT / "features" / "splits" / "80_20" / "test.csv"
    features_csv = DL_ROOT / "features" / "iemocap_features.csv"

    if train_csv.exists():
        import pandas as pd
        train = pd.read_csv(train_csv)
        test = pd.read_csv(test_csv)
        print(f"Train samples : {len(train)}")
        print(f"Test  samples : {len(test)}")
        if "emotion" in train.columns:
            print(f"Train class dist:\n{train['emotion'].value_counts().to_string()}")

    if features_csv.exists():
        import pandas as pd
        feats = pd.read_csv(features_csv)
        print(f"\niemocap_features.csv : {feats.shape[0]} rows x {feats.shape[1]} cols")

    print("=" * 60)
    print("Step 0 complete. Next: run dl_01_extract_spectrograms.py")


if __name__ == "__main__":
    create_dirs()
    copy_data()
    missing = check_packages()
    check_gpu()
    print_summary()

    if missing:
        print(f"\nInstall missing packages: pip install {' '.join(missing)}")
        sys.exit(1)
