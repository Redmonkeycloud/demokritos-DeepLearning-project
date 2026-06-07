"""
Step 6: wav2vec2 fine-tuning for Speech Emotion Recognition.
Model : facebook/wav2vec2-base (pretrained on LibriSpeech 960h)
Head  : learnable attention pooling -> Linear(768->256) -> Linear(256->4)

Training strategy (2 phases):
  Phase 1 β€” Frozen   (epochs 1β€“FREEZE_EPOCHS): wav2vec2 fully frozen, train head only
  Phase 2 β€” Finetune (epoch FREEZE_EPOCHS+1 onward): unfreeze last UNFREEZE_LAYERS
             transformer layers + feature_projection with layerwise LR decay

Post-training analysis:
  - Layer probing  : linear probe per layer (0β€“12) on test set -> UAR curve
  - Attention viz  : attention weight bar plots per emotion class
"""

import argparse
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import librosa
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
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
MODEL_DIR  = DL_ROOT / "models" / "wav2vec2"
RESULT_DIR = DL_ROOT / "results" / "wav2vec2"

DEFAULT_DATASET_ROOT = r"D:\Users\User\Desktop\demokritos-ml-project"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LABEL2IDX = {"angry": 0, "happy": 1, "neutral": 2, "sad": 3}
IDX2LABEL = {v: k for k, v in LABEL2IDX.items()}

MODEL_NAME     = "facebook/wav2vec2-base"
SR             = 16_000
MAX_LENGTH_SEC = 8               # truncate/pad audio to 8 seconds
MAX_LENGTH     = MAX_LENGTH_SEC * SR

BATCH_SIZE     = 16              # smaller batch β€” model is large
MAX_EPOCHS     = 40
PATIENCE       = 7
VAL_FRAC       = 0.10
SEED           = 42

FREEZE_EPOCHS    = 5             # Phase 1: head-only
UNFREEZE_LAYERS  = 4             # Phase 2: last N transformer layers
LR_HEAD          = 1e-3          # head learning rate
LR_FINETUNE      = 5e-5          # base fine-tune LR (top layer)
LAYERWISE_DECAY  = 0.9           # per-layer LR multiplier going down


# ---------------------------------------------------------------------------
# Path remapping (Docker -> local)
# ---------------------------------------------------------------------------
def remap_path(docker_path: str, dataset_root: Path) -> Path:
    rel = docker_path.replace("/workspace/", "").replace("/", "\\")
    return dataset_root / rel


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class IEMOCAPWavDataset(Dataset):
    def __init__(self, df: pd.DataFrame, feature_extractor, dataset_root: Path):
        self.rows       = df.reset_index(drop=True)
        self.extractor  = feature_extractor
        self.root       = dataset_root

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row      = self.rows.iloc[idx]
        wav_path = remap_path(row["file_path"], self.root)
        audio, _ = librosa.load(str(wav_path), sr=SR, mono=True)

        inputs = self.extractor(
            audio,
            sampling_rate=SR,
            max_length=MAX_LENGTH,
            truncation=True,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_values   = inputs["input_values"].squeeze(0)    # (MAX_LENGTH,)
        attention_mask = inputs["attention_mask"].squeeze(0)  # (MAX_LENGTH,)
        label          = LABEL2IDX[row["label"]]
        return input_values, attention_mask, label


def make_loader(df, extractor, dataset_root, shuffle=True):
    ds = IEMOCAPWavDataset(df, extractor, dataset_root)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle,
                      num_workers=0, pin_memory=False)


# ---------------------------------------------------------------------------
# Attention pooling (single-head, interpretable weights)
# ---------------------------------------------------------------------------
class AttentionPooling(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.score = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor):
        # x: (B, T, H)
        scores  = self.score(x).squeeze(-1)             # (B, T)
        weights = torch.softmax(scores, dim=1)           # (B, T)
        context = (weights.unsqueeze(-1) * x).sum(dim=1) # (B, H)
        return context, weights


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class Wav2Vec2SER(nn.Module):
    def __init__(self, model_name: str, dropout: float = 0.3, n_classes: int = 4):
        super().__init__()
        self.wav2vec2   = Wav2Vec2Model.from_pretrained(model_name)
        hidden          = self.wav2vec2.config.hidden_size  # 768
        self.pool       = AttentionPooling(hidden)
        self.classifier = nn.Sequential(
            nn.Linear(hidden, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, n_classes),
        )

    def forward(self, input_values, attention_mask=None):
        out    = self.wav2vec2(input_values, attention_mask=attention_mask)
        hidden = out.last_hidden_state          # (B, T, 768)
        ctx, _ = self.pool(hidden)              # (B, 768)
        return self.classifier(ctx)

    def get_attention_weights(self, input_values, attention_mask=None):
        """Return (logits, attention_weights) for visualization."""
        out    = self.wav2vec2(input_values, attention_mask=attention_mask)
        hidden = out.last_hidden_state
        ctx, weights = self.pool(hidden)
        return self.classifier(ctx), weights


# ---------------------------------------------------------------------------
# Freeze / unfreeze helpers
# ---------------------------------------------------------------------------
def freeze_wav2vec2(model: Wav2Vec2SER):
    for p in model.wav2vec2.parameters():
        p.requires_grad = False


def unfreeze_last_layers(model: Wav2Vec2SER, n_layers: int):
    """Unfreeze last n_layers transformer encoder layers + feature_projection."""
    total = len(model.wav2vec2.encoder.layers)
    for i in range(total - n_layers, total):
        for p in model.wav2vec2.encoder.layers[i].parameters():
            p.requires_grad = True
    for p in model.wav2vec2.feature_projection.parameters():
        p.requires_grad = True


def get_finetune_params(model: Wav2Vec2SER):
    """Parameter groups with layerwise LR decay for fine-tuning phase."""
    total  = len(model.wav2vec2.encoder.layers)
    groups = []

    # Head (always highest LR)
    head_params = list(model.pool.parameters()) + list(model.classifier.parameters())
    groups.append({"params": head_params, "lr": LR_HEAD})

    # Transformer layers (top -> bottom, decaying LR)
    for i in range(total - 1, total - UNFREEZE_LAYERS - 1, -1):
        layer_lr = LR_FINETUNE * (LAYERWISE_DECAY ** (total - 1 - i))
        groups.append({
            "params": list(model.wav2vec2.encoder.layers[i].parameters()),
            "lr": layer_lr,
        })

    # Feature projection (lowest LR)
    fp_lr = LR_FINETUNE * (LAYERWISE_DECAY ** UNFREEZE_LAYERS)
    groups.append({
        "params": list(model.wav2vec2.feature_projection.parameters()),
        "lr": fp_lr,
    })

    return groups


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------
def compute_metrics(y_true, y_pred) -> dict:
    return {
        "accuracy":    float(accuracy_score(y_true, y_pred)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "macro_f1":    float(f1_score(y_true, y_pred, average="macro",    zero_division=0)),
        "uar":         float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def evaluate(model, loader, device):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for iv, am, labels in loader:
            iv, am = iv.to(device), am.to(device)
            logits = model(iv, am)
            preds.extend(logits.argmax(1).cpu().tolist())
            trues.extend(labels.tolist())
    return compute_metrics(trues, preds), trues, preds


# ---------------------------------------------------------------------------
# Post-training: Layer Probing
# ---------------------------------------------------------------------------
def layer_probing(model: Wav2Vec2SER, train_loader, test_loader, device):
    """Train a logistic regression probe on each hidden layer's mean-pooled output."""
    print("\n-- Layer Probing --")
    model.eval()
    n_layers = model.wav2vec2.config.num_hidden_layers  # 12

    def extract_all(loader):
        hidden_per_layer = [[] for _ in range(n_layers + 1)]
        all_labels = []
        with torch.no_grad():
            for iv, am, labels in loader:
                iv, am = iv.to(device), am.to(device)
                out = model.wav2vec2(iv, am, output_hidden_states=True)
                for i, hs in enumerate(out.hidden_states):  # tuple of (n+1) tensors
                    hidden_per_layer[i].append(hs.mean(dim=1).cpu().numpy())
                all_labels.extend(labels.numpy())
        return [np.concatenate(h) for h in hidden_per_layer], np.array(all_labels)

    print("  Extracting train features...")
    X_tr_layers, y_tr = extract_all(train_loader)
    print("  Extracting test features...")
    X_te_layers, y_te = extract_all(test_loader)

    probe_results = {}
    for i in range(n_layers + 1):
        clf = LogisticRegression(max_iter=500, random_state=SEED, C=1.0)
        clf.fit(X_tr_layers[i], y_tr)
        preds = clf.predict(X_te_layers[i])
        uar   = float(recall_score(y_te, preds, average="macro", zero_division=0))
        probe_results[i] = uar
        print(f"  Layer {i:2d}: UAR = {uar:.4f}")

    # Plot
    layers = list(probe_results.keys())
    uars   = list(probe_results.values())
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(layers, uars, marker="o", color="#a855f7", linewidth=2)
    ax.set_xlabel("Layer index (0 = feature projection output)")
    ax.set_ylabel("UAR (test set)")
    ax.set_title("wav2vec2 Layer Probing β€” Emotion Information per Layer")
    ax.set_xticks(layers)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULT_DIR / "layer_probing.png", dpi=120)
    plt.close()

    with open(RESULT_DIR / "layer_probing.json", "w") as f:
        json.dump(probe_results, f, indent=2)

    print(f"  Best layer: {max(probe_results, key=probe_results.get)} "
          f"(UAR={max(probe_results.values()):.4f})")
    return probe_results


# ---------------------------------------------------------------------------
# Post-training: Attention Visualization
# ---------------------------------------------------------------------------
def visualize_attention(model: Wav2Vec2SER, test_loader, device):
    """Plot attention weights for 2 samples per emotion class."""
    print("\n-- Attention Visualization --")
    model.eval()

    samples = {k: [] for k in range(4)}
    with torch.no_grad():
        for iv, am, labels in test_loader:
            for i in range(len(labels)):
                lbl = labels[i].item()
                if len(samples[lbl]) < 2:
                    samples[lbl].append((
                        iv[i:i+1].to(device),
                        am[i:i+1].to(device),
                    ))
            if all(len(v) >= 2 for v in samples.values()):
                break

    fig, axes = plt.subplots(4, 2, figsize=(14, 10))
    for cls_idx, cls_name in IDX2LABEL.items():
        for j, (iv, am) in enumerate(samples[cls_idx]):
            with torch.no_grad():
                _, weights = model.get_attention_weights(iv, am)
            w = weights.squeeze().cpu().numpy()
            ax = axes[cls_idx, j]
            ax.bar(range(len(w)), w, color="#a855f7", alpha=0.7, width=1.0)
            ax.set_title(f"{cls_name} β€” sample {j+1}", fontsize=10)
            ax.set_xlabel("Time frame (~20ms each)")
            ax.set_ylabel("Attention weight")

    plt.suptitle("wav2vec2 Attention Weights per Emotion Class", fontsize=13)
    plt.tight_layout()
    plt.savefig(RESULT_DIR / "attention_visualization.png", dpi=120)
    plt.close()
    print("  Saved attention_visualization.png")


# ---------------------------------------------------------------------------
# Save confusion matrix + history
# ---------------------------------------------------------------------------
def save_confusion_matrix(y_true, y_pred, name: str):
    cm     = confusion_matrix(y_true, y_pred)
    labels = [IDX2LABEL[i] for i in range(4)]
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Purples",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"wav2vec2 β€” Confusion Matrix ({name})")
    plt.tight_layout()
    plt.savefig(RESULT_DIR / f"confusion_{name}.png", dpi=120)
    plt.close()


def save_history(history: dict):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(epochs, history["train_loss"], label="Train")
    ax1.plot(epochs, history["val_loss"],   label="Val")
    ax1.axvline(FREEZE_EPOCHS + 0.5, color="gray", linestyle="--", alpha=0.6,
                label=f"Unfreeze (ep {FREEZE_EPOCHS+1})")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.set_title("Loss"); ax1.legend()
    ax2.plot(epochs, history["val_f1"])
    ax2.axvline(FREEZE_EPOCHS + 0.5, color="gray", linestyle="--", alpha=0.6)
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Weighted F1"); ax2.set_title("Validation F1")
    plt.suptitle("wav2vec2 Training History")
    plt.tight_layout()
    plt.savefig(RESULT_DIR / "training_history.png", dpi=120)
    plt.close()


# ---------------------------------------------------------------------------
# Main training
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    args = parser.parse_args()
    dataset_root = Path(args.dataset_root)

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    # β”€β”€ Feature extractor β”€β”€
    extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)

    # β”€β”€ Data splits β”€β”€
    df_train = pd.read_csv(SPLITS_DIR / "train.csv")
    df_test  = pd.read_csv(SPLITS_DIR / "test.csv")

    labels_all = df_train["label"].map(LABEL2IDX).values
    idx        = np.arange(len(df_train))
    idx_tr, idx_val = train_test_split(
        idx, test_size=VAL_FRAC, stratify=labels_all, random_state=SEED
    )
    df_tr  = df_train.iloc[idx_tr].reset_index(drop=True)
    df_val = df_train.iloc[idx_val].reset_index(drop=True)

    print(f"Train={len(df_tr)}  Val={len(df_val)}  Test={len(df_test)}")

    tr_loader   = make_loader(df_tr,  extractor, dataset_root, shuffle=True)
    val_loader  = make_loader(df_val, extractor, dataset_root, shuffle=False)
    test_loader = make_loader(df_test, extractor, dataset_root, shuffle=False)

    # β”€β”€ Model β”€β”€
    print(f"\nLoading {MODEL_NAME}...")
    model = Wav2Vec2SER(MODEL_NAME).to(device)

    # -- Phase 1: Freeze wav2vec2, train head only --
    print(f"\n-- Phase 1: Frozen (epochs 1-{FREEZE_EPOCHS}) --")
    freeze_wav2vec2(model)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR_HEAD, weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=FREEZE_EPOCHS)
    criterion = nn.CrossEntropyLoss()

    history          = {"train_loss": [], "val_loss": [], "val_f1": []}
    best_val_f1      = -1.0
    patience_counter = 0
    best_state       = None

    for epoch in range(FREEZE_EPOCHS):
        model.train()
        running_loss = 0.0
        for iv, am, yb in tr_loader:
            iv, am, yb = iv.to(device), am.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(iv, am), yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(yb)
        scheduler.step()
        train_loss = running_loss / len(df_tr)

        model.eval()
        val_loss, preds, trues = 0.0, [], []
        with torch.no_grad():
            for iv, am, yb in val_loader:
                iv, am, yb = iv.to(device), am.to(device), yb.to(device)
                logits    = model(iv, am)
                val_loss += criterion(logits, yb).item() * len(yb)
                preds.extend(logits.argmax(1).cpu().tolist())
                trues.extend(yb.cpu().tolist())
        val_loss /= len(df_val)
        val_f1    = f1_score(trues, preds, average="weighted", zero_division=0)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_f1"].append(val_f1)

        print(f"  Ep {epoch+1:2d} | train_loss={train_loss:.4f}  "
              f"val_loss={val_loss:.4f}  val_f1={val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state  = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Save frozen-phase checkpoint for comparison
    torch.save(best_state, MODEL_DIR / "wav2vec2_frozen_best.pt")
    frozen_metrics, _, frozen_preds = evaluate(model, test_loader, device)
    print(f"\n  [Frozen] TEST UAR={frozen_metrics['uar']:.4f}  "
          f"Acc={frozen_metrics['accuracy']:.4f}")

    # -- Phase 2: Unfreeze last layers + fine-tune --
    print(f"\n-- Phase 2: Fine-tuning (last {UNFREEZE_LAYERS} layers) --")
    unfreeze_last_layers(model, UNFREEZE_LAYERS)

    optimizer = torch.optim.AdamW(get_finetune_params(model), weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=(MAX_EPOCHS - FREEZE_EPOCHS)
    )

    best_val_f1      = -1.0
    patience_counter = 0
    best_state       = None

    for epoch in range(FREEZE_EPOCHS, MAX_EPOCHS):
        model.train()
        running_loss = 0.0
        for iv, am, yb in tr_loader:
            iv, am, yb = iv.to(device), am.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(iv, am), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item() * len(yb)
        scheduler.step()
        train_loss = running_loss / len(df_tr)

        model.eval()
        val_loss, preds, trues = 0.0, [], []
        with torch.no_grad():
            for iv, am, yb in val_loader:
                iv, am, yb = iv.to(device), am.to(device), yb.to(device)
                logits    = model(iv, am)
                val_loss += criterion(logits, yb).item() * len(yb)
                preds.extend(logits.argmax(1).cpu().tolist())
                trues.extend(yb.cpu().tolist())
        val_loss /= len(df_val)
        val_f1    = f1_score(trues, preds, average="weighted", zero_division=0)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_f1"].append(val_f1)

        print(f"  Ep {epoch+1:2d} | train_loss={train_loss:.4f}  "
              f"val_loss={val_loss:.4f}  val_f1={val_f1:.4f}  pat={patience_counter}")

        if val_f1 > best_val_f1:
            best_val_f1      = val_f1
            patience_counter = 0
            best_state       = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print(f"  Early stopping at epoch {epoch + 1}")
            break

    # β”€β”€ Final evaluation β”€β”€
    model.load_state_dict(best_state)
    torch.save(best_state, MODEL_DIR / "wav2vec2_finetuned_best.pt")

    metrics, y_true, y_pred = evaluate(model, test_loader, device)
    metrics["best_val_f1"]    = float(best_val_f1)
    metrics["epochs_trained"] = len(history["train_loss"])
    metrics["frozen_uar"]     = frozen_metrics["uar"]
    metrics["frozen_acc"]     = frozen_metrics["accuracy"]

    print(f"\n{'='*55}")
    print(f"  [Fine-tuned] TEST  "
          f"accuracy={metrics['accuracy']:.4f}  "
          f"w-f1={metrics['weighted_f1']:.4f}  "
          f"UAR={metrics['uar']:.4f}")
    print(f"  [Frozen baseline]  "
          f"accuracy={frozen_metrics['accuracy']:.4f}  "
          f"UAR={frozen_metrics['uar']:.4f}")
    print(f"  Fine-tune gain: +{metrics['uar']-metrics['frozen_uar']:.4f} UAR")

    with open(RESULT_DIR / "wav2vec2_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    save_confusion_matrix(y_true, y_pred, "finetuned")
    save_history(history)

    # β”€β”€ Post-training analysis β”€β”€
    tr_loader_noshuf = make_loader(df_tr, extractor, dataset_root, shuffle=False)
    layer_probing(model, tr_loader_noshuf, test_loader, device)
    visualize_attention(model, test_loader, device)

    print("\nDone. Results in:", RESULT_DIR)


if __name__ == "__main__":
    main()

