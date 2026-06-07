"""
Step 4: CNN training on Mel-Spectrograms (1, 128, 300).
6 configs trained sequentially. Results saved per config + summary CSV.
"""

import json
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
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
ROOT       = Path(__file__).resolve().parent.parent
DL_ROOT    = ROOT / "workflows" / "iemocap_dl"
SPLITS_DIR = DL_ROOT / "features" / "splits" / "80_20"
MODEL_DIR  = DL_ROOT / "models" / "cnn"
RESULT_DIR = DL_ROOT / "results" / "cnn"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LABEL2IDX = {"angry": 0, "happy": 1, "neutral": 2, "sad": 3}
IDX2LABEL = {v: k for k, v in LABEL2IDX.items()}

BATCH_SIZE = 32
MAX_EPOCHS = 100
PATIENCE   = 10
VAL_FRAC   = 0.10
SEED       = 42

# ---------------------------------------------------------------------------
# Hyperparameter grid (from PLAN.md)
# ---------------------------------------------------------------------------
CONFIGS = {
    "cnn_1": dict(filters=[32, 64],          dropout=0.3, lr=1e-3, augmented=False),  # baseline
    "cnn_2": dict(filters=[32, 64, 128],     dropout=0.3, lr=1e-3, augmented=False),  # +depth
    "cnn_3": dict(filters=[64, 128, 256],    dropout=0.3, lr=1e-3, augmented=False),  # +width
    "cnn_4": dict(filters=[32, 64, 128],     dropout=0.5, lr=1e-3, augmented=False),  # +dropout
    "cnn_5": dict(filters=[32, 64, 128, 256],dropout=0.3, lr=5e-4, augmented=False),  # deeper+slowLR
    "cnn_6": dict(filters=[32, 64, 128],     dropout=0.3, lr=1e-3, augmented=True),   # CNN-2 + aug
}


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class SpectrogramDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.paths  = df["npy_path"].tolist()
        self.labels = df["label"].map(LABEL2IDX).tolist()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        spec = np.load(self.paths[idx]).astype(np.float32)   # (1, 128, 300)
        return torch.from_numpy(spec), self.labels[idx]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class CNN(nn.Module):
    def __init__(self, filters: list, dropout: float, n_classes: int = 4):
        super().__init__()
        blocks = []
        in_ch  = 1
        for out_ch in filters:
            blocks += [
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
            ]
            in_ch = out_ch
        self.cnn        = nn.Sequential(*blocks)
        self.pool       = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(filters[-1], 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.pool(self.cnn(x)))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_split(cfg: dict):
    df_train = pd.read_csv(SPLITS_DIR / "train_manifest.csv")

    if cfg["augmented"]:
        aug_path = SPLITS_DIR / "train_augmented_manifest.csv"
        if not aug_path.exists():
            print(f"ERROR: {aug_path} not found. Run dl_02_noise_augment.py first.")
            sys.exit(1)
        df_aug   = pd.read_csv(aug_path)[["npy_path", "label"]]
        df_train = pd.concat([df_train[["npy_path", "label"]], df_aug], ignore_index=True)

    df_test = pd.read_csv(SPLITS_DIR / "test_manifest.csv")

    labels_all = df_train["label"].map(LABEL2IDX).values
    idx        = np.arange(len(df_train))
    idx_tr, idx_val = train_test_split(
        idx, test_size=VAL_FRAC, stratify=labels_all, random_state=SEED
    )

    df_tr  = df_train.iloc[idx_tr].reset_index(drop=True)
    df_val = df_train.iloc[idx_val].reset_index(drop=True)

    return df_tr, df_val, df_test


def make_loader(df: pd.DataFrame, shuffle: bool = True) -> DataLoader:
    ds = SpectrogramDataset(df)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle,
                      num_workers=0, pin_memory=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
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
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"{name.upper()} — Confusion Matrix")
    plt.tight_layout()
    plt.savefig(RESULT_DIR / f"{name}_confusion.png", dpi=120)
    plt.close()


def save_history(history: dict, name: str):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(epochs, history["train_loss"], label="Train")
    ax1.plot(epochs, history["val_loss"],   label="Val")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.set_title("Loss"); ax1.legend()
    ax2.plot(epochs, history["val_f1"])
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Weighted F1")
    ax2.set_title("Validation F1")
    plt.suptitle(name.upper())
    plt.tight_layout()
    plt.savefig(RESULT_DIR / f"{name}_history.png", dpi=120)
    plt.close()


# ---------------------------------------------------------------------------
# Training loop for one config
# ---------------------------------------------------------------------------
def train_config(name: str, cfg: dict, device: torch.device) -> dict:
    print(f"\n{'='*55}")
    print(f"  {name.upper()}  |  filters={cfg['filters']}  "
          f"dropout={cfg['dropout']}  lr={cfg['lr']}  aug={cfg['augmented']}")
    print(f"{'='*55}")

    df_tr, df_val, df_test = load_split(cfg)
    print(f"  Train={len(df_tr)}  Val={len(df_val)}  Test={len(df_test)}")

    model     = CNN(cfg["filters"], cfg["dropout"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)
    criterion = nn.CrossEntropyLoss()

    tr_loader  = make_loader(df_tr,   shuffle=True)
    val_loader = make_loader(df_val,  shuffle=False)
    test_loader= make_loader(df_test, shuffle=False)

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
        train_loss = running_loss / len(df_tr)

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
        val_loss /= len(df_val)
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

        if (epoch + 1) % 5 == 0:
            print(f"  Ep {epoch+1:3d} | train_loss={train_loss:.4f}  "
                  f"val_loss={val_loss:.4f}  val_f1={val_f1:.4f}  pat={patience_counter}")

        if patience_counter >= PATIENCE:
            print(f"  Early stopping at epoch {epoch + 1}")
            break

    # --- test evaluation ---
    model.load_state_dict(best_state)
    torch.save(best_state, MODEL_DIR / f"{name}_best.pt")

    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for Xb, yb in test_loader:
            preds.extend(model(Xb.to(device)).argmax(1).cpu().tolist())
            trues.extend(yb.tolist())

    metrics = compute_metrics(trues, preds)
    metrics["best_val_f1"]    = float(best_val_f1)
    metrics["epochs_trained"] = len(history["train_loss"])
    metrics["filters"]        = cfg["filters"]
    metrics["dropout"]        = cfg["dropout"]
    metrics["lr"]             = cfg["lr"]
    metrics["augmented"]      = cfg["augmented"]

    print(f"\n  TEST  accuracy={metrics['accuracy']:.4f}  "
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
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    summary = []
    for name, cfg in CONFIGS.items():
        metrics = train_config(name, cfg, device)
        summary.append({"config": name, **metrics})

    df_summary = pd.DataFrame(summary)
    df_summary.to_csv(RESULT_DIR / "cnn_summary.csv", index=False)

    print("\n" + "=" * 55)
    print("CNN SUMMARY")
    print("=" * 55)
    cols = ["config", "accuracy", "weighted_f1", "macro_f1", "uar", "epochs_trained"]
    print(df_summary[cols].to_string(index=False))
    print(f"\nBest config by UAR: {df_summary.loc[df_summary['uar'].idxmax(), 'config']}")


if __name__ == "__main__":
    main()
