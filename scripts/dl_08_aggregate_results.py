"""
Step 8: Aggregate all LOSO results into a single master comparison CSV.

Reads:
  results/mlp/mlp_summary.csv
  results/cnn/cnn_summary.csv
  results/cnn_lstm/cnn_lstm_summary.csv
  results/noise_robustness/robustness_summary.csv   (optional — for noise metrics)

Writes:
  results/aggregate/master_comparison.csv    — one row per model family (best config)
  results/aggregate/master_comparison.txt    — pretty-printed table for quick review
"""

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT      = Path(__file__).resolve().parent.parent
DL_ROOT   = ROOT / "workflows" / "iemocap_dl"
RESULT_DIR = DL_ROOT / "results" / "aggregate"


def best_row(csv_path: Path, sort_col: str = "uar") -> pd.Series:
    df = pd.read_csv(csv_path)
    return df.sort_values(sort_col, ascending=False).iloc[0]


def load_wav2vec2_metrics() -> dict:
    """Load metrics from the wav2vec2 results dir (JSON or summary CSV)."""
    w2v_dir = DL_ROOT / "results" / "wav2vec2"
    # Prefer summary CSV if it exists
    summary = w2v_dir / "wav2vec2_summary.csv"
    if summary.exists():
        df = pd.read_csv(summary)
        row = df.sort_values("uar", ascending=False).iloc[0]
        return row.to_dict()
    # Fall back to any metrics JSON
    jsons = sorted(w2v_dir.glob("*_metrics.json"))
    if jsons:
        best = None
        best_uar = -1.0
        for j in jsons:
            with open(j) as f:
                d = json.load(f)
            if d.get("uar", 0) > best_uar:
                best_uar = d["uar"]
                best = d
                best["config"] = j.stem.replace("_metrics", "")
        return best or {}
    return {}


def load_robustness(model_family: str, rob_csv: Path) -> dict:
    """Extract clean_UAR + best noised UAR + best denoised UAR for a model family."""
    if not rob_csv.exists():
        return {}
    df = pd.read_csv(rob_csv)
    row = df[df["model"].str.lower() == model_family.lower()]
    if row.empty:
        return {}
    row = row.iloc[0]
    noised_cols   = [c for c in df.columns if c.endswith("_noised_UAR")]
    denoised_cols = [c for c in df.columns if c.endswith("_denoised_UAR")]
    return {
        "clean_UAR":         round(float(row.get("clean_UAR", float("nan"))), 4),
        "best_noised_UAR":   round(float(row[noised_cols].max()),  4) if noised_cols   else float("nan"),
        "worst_noised_UAR":  round(float(row[noised_cols].min()),  4) if noised_cols   else float("nan"),
        "best_denoised_UAR": round(float(row[denoised_cols].max()), 4) if denoised_cols else float("nan"),
    }


def main():
    parser = argparse.ArgumentParser(description="Step 8 — Aggregate results")
    parser.add_argument("--split-dir", default="loso",
                        help="Split subfolder (informational only; default: loso)")
    args = parser.parse_args()

    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    mlp_csv  = DL_ROOT / "results" / "mlp"  / "mlp_summary.csv"
    cnn_csv  = DL_ROOT / "results" / "cnn"  / "cnn_summary.csv"
    cl_csv   = DL_ROOT / "results" / "cnn_lstm" / "cnn_lstm_summary.csv"
    rob_csv  = DL_ROOT / "results" / "noise_robustness" / "robustness_summary.csv"

    rows = []

    # ── MLP ──────────────────────────────────────────────────────────────────
    if mlp_csv.exists():
        b = best_row(mlp_csv)
        rob = load_robustness("MLP", rob_csv)
        rows.append({
            "family":       "MLP",
            "best_config":  b["config"],
            "n_params_k":   "~100–500",
            "accuracy":     round(float(b["accuracy"]),    4),
            "weighted_f1":  round(float(b["weighted_f1"]), 4),
            "macro_f1":     round(float(b["macro_f1"]),    4),
            "UAR_clean":    round(float(b["uar"]),          4),
            **{f"rob_{k}": v for k, v in rob.items()},
        })
        print(f"MLP    best={b['config']}  UAR={b['uar']:.4f}")
    else:
        print("WARNING: mlp_summary.csv not found — skipping MLP row")

    # ── CNN ──────────────────────────────────────────────────────────────────
    if cnn_csv.exists():
        b = best_row(cnn_csv)
        rob = load_robustness("CNN", rob_csv)
        rows.append({
            "family":       "CNN",
            "best_config":  b["config"],
            "n_params_k":   "~200–800",
            "accuracy":     round(float(b["accuracy"]),    4),
            "weighted_f1":  round(float(b["weighted_f1"]), 4),
            "macro_f1":     round(float(b["macro_f1"]),    4),
            "UAR_clean":    round(float(b["uar"]),          4),
            **{f"rob_{k}": v for k, v in rob.items()},
        })
        print(f"CNN    best={b['config']}  UAR={b['uar']:.4f}")
    else:
        print("WARNING: cnn_summary.csv not found — skipping CNN row")

    # ── CNN-LSTM ─────────────────────────────────────────────────────────────
    if cl_csv.exists():
        b = best_row(cl_csv)
        rob = load_robustness("CNN-LSTM", rob_csv)
        rows.append({
            "family":       "CNN-LSTM",
            "best_config":  b["config"],
            "n_params_k":   "~500–2000",
            "accuracy":     round(float(b["accuracy"]),    4),
            "weighted_f1":  round(float(b["weighted_f1"]), 4),
            "macro_f1":     round(float(b["macro_f1"]),    4),
            "UAR_clean":    round(float(b["uar"]),          4),
            **{f"rob_{k}": v for k, v in rob.items()},
        })
        print(f"CNNLSTM best={b['config']}  UAR={b['uar']:.4f}")
    else:
        print("WARNING: cnn_lstm_summary.csv not found — skipping CNN-LSTM row")

    # ── wav2vec2 ─────────────────────────────────────────────────────────────
    w2v = load_wav2vec2_metrics()
    if w2v:
        rob = load_robustness("wav2vec2", rob_csv)
        rows.append({
            "family":       "wav2vec2",
            "best_config":  w2v.get("config", "wav2vec2_finetuned"),
            "n_params_k":   "~95000",
            "accuracy":     round(float(w2v.get("accuracy",    float("nan"))), 4),
            "weighted_f1":  round(float(w2v.get("weighted_f1", float("nan"))), 4),
            "macro_f1":     round(float(w2v.get("macro_f1",    float("nan"))), 4),
            "UAR_clean":    round(float(w2v.get("uar",         float("nan"))), 4),
            **{f"rob_{k}": v for k, v in rob.items()},
        })
        print(f"wav2vec2 best={w2v.get('config')}  UAR={w2v.get('uar', '?'):.4f}"
              if isinstance(w2v.get("uar"), float) else f"wav2vec2 loaded")
    else:
        print("WARNING: wav2vec2 metrics not found — skipping wav2vec2 row")

    if not rows:
        print("ERROR: No result rows collected. Nothing to aggregate.")
        raise SystemExit(1)

    df = pd.DataFrame(rows)
    out_csv = RESULT_DIR / "master_comparison.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")

    # Pretty-print
    txt_path = RESULT_DIR / "master_comparison.txt"
    with open(txt_path, "w", encoding="utf-8") as fh:
        header = f"{'Family':<12} {'Best config':<20} {'UAR':>7} {'Acc':>7} {'W-F1':>7} {'M-F1':>7}"
        fh.write(header + "\n")
        fh.write("-" * len(header) + "\n")
        for _, row in df.iterrows():
            fh.write(
                f"{row['family']:<12} {row['best_config']:<20} "
                f"{row['UAR_clean']:>7.4f} {row['accuracy']:>7.4f} "
                f"{row['weighted_f1']:>7.4f} {row['macro_f1']:>7.4f}\n"
            )
    print(f"Saved: {txt_path}")

    print("\n" + "=" * 60)
    print("MASTER COMPARISON (sorted by UAR desc)")
    print("=" * 60)
    display_cols = ["family", "best_config", "UAR_clean", "accuracy", "weighted_f1", "macro_f1"]
    rob_display  = [c for c in df.columns if c.startswith("rob_")]
    print(df.sort_values("UAR_clean", ascending=False)[display_cols + rob_display].to_string(index=False))
    best = df.sort_values("UAR_clean", ascending=False).iloc[0]
    print(f"\nOverall best: {best['family']} / {best['best_config']}  UAR={best['UAR_clean']:.4f}")


if __name__ == "__main__":
    main()
