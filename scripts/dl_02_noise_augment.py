"""
Step 2: Noise Augmentation — TRAIN SET ONLY.

DATA LEAKAGE: test set is NEVER touched here.

5 noise conditions applied to all 4424 train samples:
  gauss_20  Gaussian noise  SNR = 20 dB
  gauss_10  Gaussian noise  SNR = 10 dB
  gauss_5   Gaussian noise  SNR =  5 dB
  room_03   Reverberation   RT60 = 0.3 s
  room_06   Reverberation   RT60 = 0.6 s

Produces (4424 x 5 = 22 120 augmented samples):
  workflows/iemocap_dl/spectrograms/train_augmented/<stem>_<cond>.npy
  workflows/iemocap_dl/features/splits/80_20/train_augmented_features.csv
  workflows/iemocap_dl/features/splits/80_20/train_augmented_manifest.csv
"""

import argparse
import numpy as np
import pandas as pd
import librosa
from pathlib import Path
from tqdm import tqdm
from pyAudioAnalysis import ShortTermFeatures

np.random.seed(42)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT       = Path(__file__).resolve().parent.parent
DL_ROOT    = ROOT / "workflows" / "iemocap_dl"
SPLITS_DIR = DL_ROOT / "features" / "splits" / "80_20"
SPEC_AUG   = DL_ROOT / "spectrograms" / "train_augmented"

DEFAULT_DATASET_ROOT = r"D:\Users\User\Desktop\demokritos-ml-project"

# ---------------------------------------------------------------------------
# Audio / spectrogram parameters (identical to Step 1B)
# ---------------------------------------------------------------------------
SR         = 16_000
ST_WIN     = 0.050   # 50 ms  — same as original feature extraction
ST_STEP    = 0.025   # 25 ms
N_MELS     = 128
HOP_LENGTH = 160
WIN_LENGTH = 400
N_FFT      = 1024
N_FRAMES   = 300


# ---------------------------------------------------------------------------
# Augmentation functions
# ---------------------------------------------------------------------------
def add_gaussian_snr(audio: np.ndarray, snr_db: float) -> np.ndarray:
    signal_power = np.mean(audio ** 2)
    if signal_power < 1e-10:
        return audio
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.random.randn(len(audio)) * np.sqrt(noise_power)
    return np.clip(audio + noise, -1.0, 1.0).astype(np.float32)


def apply_reverb(audio: np.ndarray, sr: int, rt60: float) -> np.ndarray:
    """Convolve with a synthetic exponential-decay room impulse response."""
    n_rir = int(rt60 * sr)
    t = np.arange(n_rir) / sr
    rir = np.exp(-6.908 * t / rt60) * np.random.randn(n_rir)
    rir /= (np.linalg.norm(rir) + 1e-8)
    reverbed = np.convolve(audio, rir)[: len(audio)]
    # Preserve original RMS level
    rms_orig = np.sqrt(np.mean(audio ** 2) + 1e-10)
    rms_rev  = np.sqrt(np.mean(reverbed ** 2) + 1e-10)
    reverbed = reverbed * (rms_orig / rms_rev)
    return np.clip(reverbed, -1.0, 1.0).astype(np.float32)


CONDITIONS = {
    "gauss_20": lambda a: add_gaussian_snr(a, snr_db=20),
    "gauss_10": lambda a: add_gaussian_snr(a, snr_db=10),
    "gauss_5":  lambda a: add_gaussian_snr(a, snr_db=5),
    "room_03":  lambda a: apply_reverb(a, SR, rt60=0.3),
    "room_06":  lambda a: apply_reverb(a, SR, rt60=0.6),
}


# ---------------------------------------------------------------------------
# Feature & spectrogram extraction
# ---------------------------------------------------------------------------
def extract_features(audio: np.ndarray):
    win_samp  = int(ST_WIN * SR)
    step_samp = int(ST_STEP * SR)
    F, f_names = ShortTermFeatures.feature_extraction(audio, SR, win_samp, step_samp)
    feat = np.concatenate([
        np.mean(F, axis=1), np.min(F, axis=1),
        np.max(F, axis=1),  np.std(F, axis=1),
    ])
    col_names = (
        [f"{n}_mean" for n in f_names] + [f"{n}_min" for n in f_names] +
        [f"{n}_max"  for n in f_names] + [f"{n}_std" for n in f_names]
    )
    return feat, col_names


def wav_to_logmel(audio: np.ndarray) -> np.ndarray:
    S    = librosa.feature.melspectrogram(
        y=audio, sr=SR, n_mels=N_MELS,
        hop_length=HOP_LENGTH, win_length=WIN_LENGTH, n_fft=N_FFT,
    )
    S_db = librosa.power_to_db(S, ref=np.max)
    T    = S_db.shape[1]
    if T < N_FRAMES:
        S_db = np.pad(S_db, ((0, 0), (0, N_FRAMES - T)),
                      mode="constant", constant_values=S_db.min())
    else:
        S_db = S_db[:, :N_FRAMES]
    return S_db[np.newaxis, :, :].astype(np.float32)   # (1, 128, 300)


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------
def remap_path(docker_path: str, dataset_root: Path) -> Path:
    rel = docker_path.replace("/workspace/", "").replace("/", "\\")
    return dataset_root / rel


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root", default=DEFAULT_DATASET_ROOT,
        help="Root containing datasets/iemocap/Session* (default: %(default)s)",
    )
    args   = parser.parse_args()
    dataset_root = Path(args.dataset_root)

    SPEC_AUG.mkdir(parents=True, exist_ok=True)

    train_csv = SPLITS_DIR / "train.csv"
    df        = pd.read_csv(train_csv)

    feature_records  = []
    manifest_records = []
    col_names        = None
    failed           = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="augmenting"):
        wav_path = remap_path(row["file_path"], dataset_root)
        if not wav_path.exists():
            failed.append(str(wav_path))
            continue

        audio, _ = librosa.load(str(wav_path), sr=SR, mono=True)

        for cond_name, augment_fn in CONDITIONS.items():
            aug_audio = augment_fn(audio)
            stem      = f"{wav_path.stem}_{cond_name}"

            # --- spectrogram ---
            npy_path = SPEC_AUG / f"{stem}.npy"
            if not npy_path.exists():
                np.save(str(npy_path), wav_to_logmel(aug_audio))

            manifest_records.append({
                "npy_path":  str(npy_path),
                "file_path": row["file_path"],
                "label":     row["label"],
                "condition": cond_name,
            })

            # --- 272D features ---
            feat, names = extract_features(aug_audio)
            if col_names is None:
                col_names = names

            rec = dict(zip(col_names, feat))
            rec["label"]     = row["label"]
            rec["file_path"] = row["file_path"]
            rec["dataset"]   = "iemocap"
            rec["condition"] = cond_name
            feature_records.append(rec)

    if failed:
        print(f"\nWARNING: {len(failed)} WAV files not found, skipped.")

    # Save features CSV
    df_feat = pd.DataFrame(feature_records)
    feat_out = SPLITS_DIR / "train_augmented_features.csv"
    df_feat.to_csv(feat_out, index=False)

    # Save manifest CSV
    df_manifest = pd.DataFrame(manifest_records)
    manifest_out = SPLITS_DIR / "train_augmented_manifest.csv"
    df_manifest.to_csv(manifest_out, index=False)

    print(f"\nDone.")
    print(f"  Augmented samples  : {len(manifest_records)}  ({len(df)} x {len(CONDITIONS)} conditions)")
    print(f"  Spectrograms       : {SPEC_AUG}")
    print(f"  Features CSV       : {feat_out}")
    print(f"  Manifest CSV       : {manifest_out}")

    if len(manifest_records) > 0:
        sample = np.load(manifest_records[0]["npy_path"])
        print(f"\nSanity check — shape: {sample.shape}  min={sample.min():.1f}  max={sample.max():.1f}")


if __name__ == "__main__":
    main()
