"""
Step 1B: Mel-Spectrogram extraction for CNN / CNN-LSTM training.

Reads train.csv + test.csv, loads each WAV via librosa, computes a
log-mel spectrogram, pads/truncates to exactly 300 frames, and saves
each sample as a .npy file of shape (1, 128, 300).

Outputs:
  workflows/iemocap_dl/spectrograms/train/<name>.npy
  workflows/iemocap_dl/spectrograms/test/<name>.npy
  workflows/iemocap_dl/features/splits/80_20/train_manifest.csv
  workflows/iemocap_dl/features/splits/80_20/test_manifest.csv
"""

import argparse
import numpy as np
import pandas as pd
import librosa
from pathlib import Path
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT       = Path(__file__).resolve().parent.parent
DL_ROOT    = ROOT / "workflows" / "iemocap_dl"
SPLITS_DIR = DL_ROOT / "features" / "splits" / "80_20"
SPEC_TRAIN = DL_ROOT / "spectrograms" / "train"
SPEC_TEST  = DL_ROOT / "spectrograms" / "test"

# Default: where the WAV files live on the original dev machine.
# Override with --dataset-root if your path is different.
DEFAULT_DATASET_ROOT = r"D:\Users\User\Desktop\demokritos-ml-project"

# ---------------------------------------------------------------------------
# Spectrogram parameters (from PLAN.md)
# ---------------------------------------------------------------------------
SR          = 16_000   # target sample rate
N_MELS      = 128
HOP_LENGTH  = 160      # 10 ms
WIN_LENGTH  = 400      # 25 ms
N_FFT       = 1024
N_FRAMES    = 300      # 3 seconds @ 10 ms/frame


def remap_path(docker_path: str, dataset_root: Path) -> Path:
    """Convert /workspace/... path from CSV to local path."""
    rel = docker_path.replace("/workspace/", "").replace("/", "\\")
    return dataset_root / rel


def wav_to_logmel(wav_path: Path) -> np.ndarray:
    """Load WAV, compute log-mel spectrogram, return shape (1, 128, 300)."""
    y, _ = librosa.load(str(wav_path), sr=SR, mono=True)

    S = librosa.feature.melspectrogram(
        y=y,
        sr=SR,
        n_mels=N_MELS,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        n_fft=N_FFT,
    )
    S_db = librosa.power_to_db(S, ref=np.max)  # shape: (128, T)

    # Pad or truncate time axis to N_FRAMES
    T = S_db.shape[1]
    if T < N_FRAMES:
        pad = N_FRAMES - T
        S_db = np.pad(S_db, ((0, 0), (0, pad)), mode="constant", constant_values=S_db.min())
    else:
        S_db = S_db[:, :N_FRAMES]

    return S_db[np.newaxis, :, :]  # (1, 128, 300)


def process_split(csv_path: Path, spec_dir: Path, split_name: str, dataset_root: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    records = []
    failed = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc=split_name):
        wav_path = remap_path(row["file_path"], dataset_root)
        stem = wav_path.stem
        npy_path = spec_dir / f"{stem}.npy"

        if not wav_path.exists():
            failed.append(str(wav_path))
            continue

        if not npy_path.exists():
            spec = wav_to_logmel(wav_path)
            np.save(str(npy_path), spec)

        records.append({
            "npy_path": str(npy_path),
            "file_path": row["file_path"],
            "label": row["label"],
        })

    if failed:
        print(f"\n  WARNING: {len(failed)} files not found in {split_name}.")
        for p in failed[:5]:
            print(f"    {p}")
        if len(failed) > 5:
            print(f"    ... and {len(failed) - 5} more")

    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        default=DEFAULT_DATASET_ROOT,
        help="Root folder that contains datasets/iemocap/Session* (default: %(default)s)",
    )
    args = parser.parse_args()
    dataset_root = Path(args.dataset_root)

    print("=== Step 1B: Mel-Spectrogram Extraction ===\n")
    print(f"  Dataset root : {dataset_root}\n")

    train_manifest = process_split(
        SPLITS_DIR / "train.csv", SPEC_TRAIN, "train", dataset_root
    )
    test_manifest = process_split(
        SPLITS_DIR / "test.csv", SPEC_TEST, "test", dataset_root
    )

    train_manifest.to_csv(SPLITS_DIR / "train_manifest.csv", index=False)
    test_manifest.to_csv(SPLITS_DIR / "test_manifest.csv", index=False)

    print(f"\nDone.")
    print(f"  Train spectrograms : {len(train_manifest)} / 4424")
    print(f"  Test  spectrograms : {len(test_manifest)} / 1107")
    print(f"  Saved to           : {DL_ROOT / 'spectrograms'}")
    print(f"  Manifests          : {SPLITS_DIR}")

    # Sanity check: load one file and verify shape
    if len(train_manifest) > 0:
        sample = np.load(train_manifest.iloc[0]["npy_path"])
        print(f"\nSanity check — shape of first .npy: {sample.shape}  (expected (1, 128, 300))")
        print(f"  min={sample.min():.2f}  max={sample.max():.2f}  dtype={sample.dtype}")


if __name__ == "__main__":
    main()
