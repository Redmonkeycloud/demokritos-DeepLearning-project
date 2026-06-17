"""
Step 3: MLP training on 272D hand-crafted features.
6 configs trained sequentially. Results saved per config + summary CSV.

By default reads from splits/loso/ (speaker-independent eval set).
Pass --split-dir 80_20 to use the legacy random split.

Validation split (10%, stratified) is carved from the TRAIN set only —
the eval/test set is the held-out speaker and is never used during training.
"""

import argparse
import json
import sys
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT      = Path(__file__).resolve().parent.parent
DL_ROOT   = ROOT / "workflows" / "iemocap_dl"
MODEL_DIR = DL_ROOT / "models" / "mlp"
RESULT_DIR = DL_ROOT / "results" / "mlp"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LABEL2IDX  = {"angry": 0, "happy": 1, "neutral": 2, "sad": 3}
IDX2LABEL  = {v: k for k, v in LABEL2IDX.items()}
NON_FEAT   = {"label", "file_path", "dataset", "condition", "speaker_id"}

BATCH_SIZE = 64
MAX_EPOCHS = 100
PATIENCE   = 10
VAL_FRAC   = 0.10
SEED       = 42

# ---------------------------------------------------------------------------
# Hyperparameter grid (from PLAN.md)
# ---------------------------------------------------------------------------
CONFIGS = {
    "mlp_1": dict(n_layers=2, hidden=256, dropout=0.3, lr=1e-3, augmented=False),  # baseline
    "mlp_2": dict(n_layers=3, hidden=256, dropout=0.3, lr=1e-3, augmented=False),  # +depth
    "mlp_3": dict(n_layers=3, hidden=512, dropout=0.3, lr=1e-3, augmented=False),  # +width
    "mlp_4": dict(n_layers=3, hidden=512, dropout=0.5, lr=1e-3, augmented=False),  # +dropout
    "mlp_5": dict(n_layers=4, hidden=512, dropout=0.3, lr=5e-4, augmented=False),  # deeper+slowLR
    "mlp_6": dict(n_layers=3, hidden=256, dropout=0.3, lr=1e-3, augmented=True),   # MLP-2 + aug
}


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden: int, n_layers: int, dropout: float, n_classes: int = 4):
        super().__init__()
        layers = []
        in_dim = input_dim
        for _ in range(n_layers):
            layers += [
                nn.Linear(in_dim, hidden),
                nn.BatchNorm1d(hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_dim = hidden
        layers.append(nn.Linear(hidden, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Data loading  (splits_dir injected at call time via global SPLITS_DIR)
# ---------------------------------------------------------------------------
def load_split(cfg: dict):
    df_train = pd.read_csv(SPLITS_DIR / "train.csv")
    feat_cols = [c for c in df_train.columns if c not in NON_FEAT]

    X_clean = df_train[feat_cols].values.astype(np.float32)
    y_clean = df_train["label"].map(LABEL2IDX).values

    if cfg["augmented"]:
        aug_path = SPLITS_DIR / "train_augmented_features.csv"
        if not aug_path.exists():
            print(f"ERROR: {aug_path} not found. Run dl_02_noise_augment.py first.")
            sys.exit(1)
        scaler   = joblib.load(SPLITS_DIR / "scaler.pkl")
        df_aug   = pd.read_csv(aug_path)
        aug_cols = [c for c in df_aug.columns if c not in NON_FEAT]
        X_aug    = scaler.transform(df_aug[aug_cols].values).astype(np.float32)
        y_aug    = df_aug["label"].map(LABEL2IDX).values
        X_all    = np.concatenate([X_clean, X_aug], axis=0)
        y_all    = np.concatenate([y_clean, y_aug], axis=0)
    else:
        X_all, y_all = X_clean, y_clean

    # Eval set: held-out speaker (loso) or random 20% (80_20)
    eval_csv = "eval.csv" if (SPLITS_DIR / "eval.csv").exists() else "test.csv"
    df_eval  = pd.read_csv(SPLITS_DIR / eval_csv)
    X_eval   = df_eval[feat_cols].values.astype(np.float32)
    y_eval   = df_eval["label"].map(LABEL2IDX).values

    # Validation split carved from TRAIN only — eval speaker never seen here
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_all, y_all, test_size=VAL_FRAC, stratify=y_all, random_state=SEED
    )
    return X_tr, y_tr, X_val, y_val, X_eval, y_eval, feat_cols


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_loader(X: np.ndarray, y: np.ndarray, shuffle: bool = True) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y).long())
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)


def compute_metrics(y_true, y_pred) -> dict:
    return {
        "accuracy":    float(accuracy_score(y_true, y_pred)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "macro_f1":    float(f1_score(y_true, y_pred, average="macro",    zero_division=0)),
        "uar":         float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def save_confusion_matrix(y_true, y_pred, name: str):
    cm     = confusion_matrix(y_true, y_pred)
    labels = [IDX2LABEL[i] for i in range(4)]
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{name.upper()} — Confusion Matrix")
    plt.tight_layout()
    plt.savefig(RESULT_DIR / f"{name}_confusion.png", dpi=120)
    plt.close()


def save_history(history: dict, name: str):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(epochs, history["train_loss"], label="Train")
    ax1.plot(epochs, history["val_loss"],   label="Val")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.set_title("Loss"); ax1.legend()
    ax2.plot(epochs, history["val_f1"])
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Weighted F1"); ax2.set_title("Validation F1")
    plt.suptitle(name.upper())
    plt.tight_layout()
    plt.savefig(RESULT_DIR / f"{name}_history.png", dpi=120)
    plt.close()


# ---------------------------------------------------------------------------
# Training loop for one config
# ---------------------------------------------------------------------------
def train_config(name: str, cfg: dict, device: torch.device) -> dict:
    print(f"\n{'='*55}")
    print(f"  {name.upper()}  |  layers={cfg['n_layers']}  hidden={cfg['hidden']}  "
          f"dropout={cfg['dropout']}  lr={cfg['lr']}  aug={cfg['augmented']}")
    print(f"{'='*55}")

    X_tr, y_tr, X_val, y_val, X_eval, y_eval, feat_cols = load_split(cfg)
    print(f"  Train={len(X_tr)}  Val={len(X_val)}  Eval={len(X_eval)}  Features={len(feat_cols)}")

    model     = MLP(len(feat_cols), cfg["hidden"], cfg["n_layers"], cfg["dropout"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)
    criterion = nn.CrossEntropyLoss()

    tr_loader  = make_loader(X_tr,  y_tr,  shuffle=True)
    val_loader = make_loader(X_val, y_val, shuffle=False)

    history          = {"train_loss": [], "val_loss": [], "val_f1": []}
    best_val_f1      = -1.0
    patience_counter = 0
    best_state       = None

    for epoch in range(MAX_EPOCHS):
        # --- train ---
        model.train()
        running_loss = 0.0
        for Xb, yb in tr_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(Xb)
        scheduler.step()
        train_loss = running_loss / len(X_tr)

        # --- validate ---
        model.eval()
        val_loss, preds, trues = 0.0, [], []
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                logits   = model(Xb)
                val_loss += criterion(logits, yb).item() * len(Xb)
                preds.extend(logits.argmax(1).cpu().tolist())
                trues.extend(yb.cpu().tolist())
        val_loss /= len(X_val)
        val_f1    = f1_score(trues, preds, average="weighted", zero_division=0)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_f1"].append(val_f1)

        if val_f1 > best_val_f1:
            best_val_f1      = val_f1
            patience_counter = 0
            best_state       = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            print(f"  Ep {epoch+1:3d} | train_loss={train_loss:.4f}  "
                  f"val_loss={val_loss:.4f}  val_f1={val_f1:.4f}  pat={patience_counter}")

        if patience_counter >= PATIENCE:
            print(f"  Early stopping at epoch {epoch + 1}")
            break

    # --- eval set evaluation (held-out speaker / test set) ---
    model.load_state_dict(best_state)
    torch.save(best_state, MODEL_DIR / f"{name}_best.pt")

    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for Xb, yb in make_loader(X_eval, y_eval, shuffle=False):
            preds.extend(model(Xb.to(device)).argmax(1).cpu().tolist())
            trues.extend(yb.tolist())

    metrics = compute_metrics(trues, preds)
    metrics["best_val_f1"]    = float(best_val_f1)
    metrics["epochs_trained"] = len(history["train_loss"])
    metrics["n_layers"]       = cfg["n_layers"]
    metrics["hidden"]         = cfg["hidden"]
    metrics["dropout"]        = cfg["dropout"]
    metrics["lr"]             = cfg["lr"]
    metrics["augmented"]      = cfg["augmented"]

    print(f"\n  EVAL  accuracy={metrics['accuracy']:.4f}  "
          f"w-f1={metrics['weighted_f1']:.4f}  "
          f"macro-f1={metrics['macro_f1']:.4f}  "
          f"UAR={metrics['uar']:.4f}")

    with open(RESULT_DIR / f"{name}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    save_confusion_matrix(trues, preds, name)
    save_history(history, name)

    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split-dir",
        default="loso",
        help="Subfolder under features/splits/ to read train.csv / eval.csv from "
             "(default: loso). Use '80_20' for the legacy random split.",
    )
    args = parser.parse_args()

    global SPLITS_DIR
    SPLITS_DIR = DL_ROOT / "features" / "splits" / args.split_dir

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device     : {device}")
    print(f"Split dir  : {SPLITS_DIR}\n")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    summary = []
    for name, cfg in CONFIGS.items():
        metrics = train_config(name, cfg, device)
        summary.append({"config": name, **metrics})

    df_summary = pd.DataFrame(summary)
    df_summary.to_csv(RESULT_DIR / "mlp_summary.csv", index=False)

    print("\n" + "=" * 55)
    print("MLP SUMMARY")
    print("=" * 55)
    cols = ["config", "accuracy", "weighted_f1", "macro_f1", "uar", "epochs_trained"]
    print(df_summary[cols].to_string(index=False))
    print(f"\nBest config by UAR: {df_summary.loc[df_summary['uar'].idxmax(), 'config']}")


if __name__ == "__main__":
    main()
