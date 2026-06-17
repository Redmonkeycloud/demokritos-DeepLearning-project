"""
Step 0B: Speaker-independent (Leave-One-Speaker-Out) split for IEMOCAP.

Reads iemocap_features.csv, extracts speaker_id from each utterance filename
using the IEMOCAP naming convention (SesXXM / SesXXF), holds out one speaker
as the eval set, and writes:

  workflows/iemocap_dl/features/splits/loso/
    train.csv          (scaled 272D features + label + file_path + speaker_id)
    eval.csv           (held-out speaker, same columns)
    scaler.pkl         (StandardScaler fit on train only)

The scaler is fit on train-only features to prevent data leakage.

Usage:
  python scripts/dl_00b_speaker_split.py
  python scripts/dl_00b_speaker_split.py --eval_speaker Ses04F
"""

import argparse
import re
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT        = Path(__file__).resolve().parent.parent
DL_ROOT     = ROOT / "workflows" / "iemocap_dl"
FEATURES_CSV = DL_ROOT / "features" / "iemocap_features.csv"
LOSO_DIR    = DL_ROOT / "features" / "splits" / "loso"

# Columns that are NOT numeric features
META_COLS = {"label", "file_path", "dataset"}

# IEMOCAP label : integer mapping (must match all training scripts)
LABEL2IDX = {"angry": 0, "happy": 1, "neutral": 2, "sad": 3}

# All 10 IEMOCAP speakers (5 sessions × 2 sides)
ALL_SPEAKERS = [
    "Ses01F", "Ses01M",
    "Ses02F", "Ses02M",
    "Ses03F", "Ses03M",
    "Ses04F", "Ses04M",
    "Ses05F", "Ses05M",
]

DEFAULT_EVAL_SPEAKER = "Ses05M"

# ---------------------------------------------------------------------------
# Speaker extraction
# ---------------------------------------------------------------------------
_SPEAKER_RE = re.compile(r"(Ses\d{2}[MF])")


def extract_speaker(file_path: str) -> str:
    """
    Parse speaker ID from a file_path such as:
      /workspace/datasets/iemocap/Session1/.../Ses01F_script02_1_F000.wav
    Returns e.g. "Ses01F".  Raises ValueError if pattern not found.
    """
    basename = Path(file_path).stem          # e.g. "Ses01F_script02_1_F000"
    m = _SPEAKER_RE.match(basename)
    if m is None:
        raise ValueError(f"Cannot parse speaker from: {file_path!r}")
    return m.group(1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Build LOSO split (leave-one-speaker-out) from iemocap_features.csv"
    )
    parser.add_argument(
        "--eval_speaker",
        default=DEFAULT_EVAL_SPEAKER,
        choices=ALL_SPEAKERS,
        help=f"Speaker to hold out as eval set (default: {DEFAULT_EVAL_SPEAKER})",
    )
    parser.add_argument(
        "--features_csv",
        default=str(FEATURES_CSV),
        help="Path to iemocap_features.csv (default: %(default)s)",
    )
    parser.add_argument(
        "--out_dir",
        default=str(LOSO_DIR),
        help="Output directory for train.csv / eval.csv / scaler.pkl (default: %(default)s)",
    )
    args = parser.parse_args()

    eval_speaker = args.eval_speaker
    features_path = Path(args.features_csv)
    out_dir = Path(args.out_dir)

    # ------------------------------------------------------------------
    # 1. Load raw features
    # ------------------------------------------------------------------
    if not features_path.exists():
        raise FileNotFoundError(f"Features CSV not found: {features_path}")

    print(f"Loading  {features_path} ...")
    df = pd.read_csv(features_path)
    print(f"  Shape  : {df.shape[0]} rows x {df.shape[1]} cols")

    # Validate label values
    known_labels = set(LABEL2IDX.keys())
    found_labels = set(df["label"].unique())
    unknown = found_labels - known_labels
    if unknown:
        raise ValueError(f"Unexpected labels in CSV: {unknown}. Expected: {known_labels}")

    # ------------------------------------------------------------------
    # 2. Extract speaker_id per row
    # ------------------------------------------------------------------
    print("Extracting speaker IDs ...")
    df["speaker_id"] = df["file_path"].apply(extract_speaker)

    found_speakers = sorted(df["speaker_id"].unique())
    print(f"  Speakers found : {found_speakers}")
    if eval_speaker not in found_speakers:
        raise ValueError(
            f"Eval speaker '{eval_speaker}' has no utterances in the CSV. "
            f"Found: {found_speakers}"
        )

    # ------------------------------------------------------------------
    # 3. Split
    # ------------------------------------------------------------------
    mask_eval = df["speaker_id"] == eval_speaker
    df_eval  = df[mask_eval].copy().reset_index(drop=True)
    df_train = df[~mask_eval].copy().reset_index(drop=True)

    print(f"\nSplit  (eval_speaker = {eval_speaker})")
    print(f"  Train : {len(df_train)} utterances  ({df_train['speaker_id'].nunique()} speakers)")
    print(f"  Eval  : {len(df_eval)} utterances  (1 speaker)")

    # ------------------------------------------------------------------
    # 4. Fit scaler on TRAIN features only
    # ------------------------------------------------------------------
    feat_cols = [c for c in df.columns if c not in META_COLS and c != "speaker_id"]

    scaler = StandardScaler()
    X_train = df_train[feat_cols].values.astype(np.float64)
    X_eval  = df_eval[feat_cols].values.astype(np.float64)

    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_eval_scaled  = scaler.transform(X_eval).astype(np.float32)

    # Rebuild DataFrames with scaled features
    df_train_out = pd.DataFrame(X_train_scaled, columns=feat_cols)
    df_train_out["label"]      = df_train["label"].values
    df_train_out["file_path"]  = df_train["file_path"].values
    df_train_out["dataset"]    = df_train["dataset"].values
    df_train_out["speaker_id"] = df_train["speaker_id"].values

    df_eval_out = pd.DataFrame(X_eval_scaled, columns=feat_cols)
    df_eval_out["label"]      = df_eval["label"].values
    df_eval_out["file_path"]  = df_eval["file_path"].values
    df_eval_out["dataset"]    = df_eval["dataset"].values
    df_eval_out["speaker_id"] = df_eval["speaker_id"].values

    # ------------------------------------------------------------------
    # 5. Sanity checks
    # ------------------------------------------------------------------
    # Ensure zero leakage: no eval speaker in train
    assert len(set(df_train_out["speaker_id"]) & {eval_speaker}) == 0, \
        "BUG: eval speaker found in train set!"

    # Ensure eval set has only the held-out speaker
    assert df_eval_out["speaker_id"].nunique() == 1, \
        "BUG: eval set contains more than one speaker!"

    # ------------------------------------------------------------------
    # 6. Save
    # ------------------------------------------------------------------
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path  = out_dir / "train.csv"
    eval_path   = out_dir / "eval.csv"
    scaler_path = out_dir / "scaler.pkl"

    df_train_out.to_csv(train_path, index=False)
    df_eval_out.to_csv(eval_path, index=False)
    joblib.dump(scaler, scaler_path)

    print(f"\nSaved to  {out_dir}")
    print(f"  train.csv   : {len(df_train_out)} rows")
    print(f"  eval.csv    : {len(df_eval_out)} rows")
    print(f"  scaler.pkl")

    # ------------------------------------------------------------------
    # 7. Summary report
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("PER-SPEAKER UTTERANCE COUNTS")
    print("=" * 60)
    counts = df.groupby("speaker_id").size().sort_index()
    for spk, n in counts.items():
        tag = "  <-- EVAL" if spk == eval_speaker else ""
        print(f"  {spk}: {n:4d} utterances{tag}")

    print("\n" + "=" * 60)
    print("CLASS DISTRIBUTION")
    print("=" * 60)
    print(f"\nTRAIN ({len(df_train_out)} total):")
    train_dist = df_train_out["label"].value_counts().sort_index()
    for lbl, cnt in train_dist.items():
        print(f"  {lbl:8s}: {cnt:4d}  ({100*cnt/len(df_train_out):5.1f}%)")

    print(f"\nEVAL  ({len(df_eval_out)} total):")
    eval_dist = df_eval_out["label"].value_counts().sort_index()
    for lbl, cnt in eval_dist.items():
        print(f"  {lbl:8s}: {cnt:4d}  ({100*cnt/len(df_eval_out):5.1f}%)")

    print("\n" + "=" * 60)
    print(f"TOTALS: {len(df_train_out)} train + {len(df_eval_out)} eval = "
          f"{len(df_train_out) + len(df_eval_out)} (original: {len(df)})")
    print(f"Eval speaker  : {eval_speaker}")
    print(f"Train speakers: {sorted(df_train_out['speaker_id'].unique())}")
    print("=" * 60)


if __name__ == "__main__":
    main()
