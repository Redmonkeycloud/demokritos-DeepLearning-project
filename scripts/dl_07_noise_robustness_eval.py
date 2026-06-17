"""
Step 7: Noise Robustness Evaluation with In-Memory Spectral-Subtraction Denoising

Evaluates the best checkpoint from each model family (MLP, CNN, CNN-LSTM, wav2vec2)
under three conditions per noise type:

  1. clean           — no modification, no denoising (baseline)
  2. <cond>_noised   — noise added on-the-fly from raw WAV
  3. <cond>_denoised — noise added, then spectral-subtraction denoiser applied

Pipeline:  clean_wav → add_noise → [denoise] → extract_features → model

Noise conditions: gauss_20, gauss_10, gauss_5, gauss_0, room_03, room_06

Selection strategy:
  - All checkpoints that exist on disk are evaluated on the clean set first.
  - The config with the highest clean UAR is chosen as each family's representative.
  - Only the representative is then run on all noised / denoised conditions.

Outputs  (results/noise_robustness/):
  robustness_summary.csv  — one row per model family; clean + noised/denoised per condition
  robustness_curves.png   — Gaussian + Reverb sub-plots; 3 lines per family (clean/noised/denoised)
  denoising_gain.png      — bar chart: UAR gain (denoised − noised) per model per condition
  robustness_heatmap.png  — 3-panel heatmap (Clean | Noised | Denoised)
"""

from __future__ import annotations

import argparse
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import librosa
import joblib
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import recall_score, f1_score
from pyAudioAnalysis import ShortTermFeatures
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT       = Path(__file__).resolve().parent.parent
DL_ROOT    = ROOT / "workflows" / "iemocap_dl"
MODEL_ROOT = DL_ROOT / "models"
RESULT_DIR = DL_ROOT / "results" / "noise_robustness"

DEFAULT_DATASET_ROOT = r"D:\Users\User\Desktop\demokritos-ml-project"
DEFAULT_SPLIT_DIR    = "loso"

# ---------------------------------------------------------------------------
# Label mapping
# ---------------------------------------------------------------------------
LABEL2IDX = {"angry": 0, "happy": 1, "neutral": 2, "sad": 3}
IDX2LABEL  = {v: k for k, v in LABEL2IDX.items()}
NON_FEAT   = {"label", "file_path", "dataset", "condition", "speaker_id"}

# ---------------------------------------------------------------------------
# Audio constants — must match training scripts exactly
# ---------------------------------------------------------------------------
SR          = 16_000
N_MELS      = 128
HOP_LENGTH  = 160        # 10 ms / frame
WIN_LENGTH  = 400        # 25 ms window
N_FFT_SPEC  = 1024
N_FRAMES    = 300        # 3 s at 10 ms / frame
MAX_WAV2VEC = 8 * SR     # 8 s maximum for wav2vec2

ST_WIN  = 0.050          # 50 ms  — pyAudioAnalysis window
ST_STEP = 0.025          # 25 ms

# ---------------------------------------------------------------------------
# Model configs — mirror the hyperparameter grids in training scripts
# ---------------------------------------------------------------------------
MLP_CONFIGS: dict[str, dict] = {
    "mlp_1": dict(n_layers=2, hidden=256, dropout=0.3),
    "mlp_2": dict(n_layers=3, hidden=256, dropout=0.3),
    "mlp_3": dict(n_layers=3, hidden=512, dropout=0.3),
    "mlp_4": dict(n_layers=3, hidden=512, dropout=0.5),
    "mlp_5": dict(n_layers=4, hidden=512, dropout=0.3),
    "mlp_6": dict(n_layers=3, hidden=256, dropout=0.3),
}

CNN_CONFIGS: dict[str, dict] = {
    "cnn_1":  dict(filters=[32, 64],           dropout=0.3, model_type="standard"),
    "cnn_2":  dict(filters=[32, 64, 128],      dropout=0.3, model_type="standard"),
    "cnn_3":  dict(filters=[64, 128, 256],     dropout=0.3, model_type="standard"),
    "cnn_4":  dict(filters=[32, 64, 128],      dropout=0.5, model_type="standard"),
    "cnn_5":  dict(filters=[32, 64, 128, 256], dropout=0.3, model_type="standard"),
    "cnn_6":  dict(filters=[32, 64, 128],      dropout=0.3, model_type="standard"),
    "cnn_7":  dict(filters=[32, 64, 128],      dropout=0.3, model_type="gap"),
    "cnn_8":  dict(filters=[32, 64, 128],      dropout=0.3, model_type="residual"),
    "cnn_9":  dict(filters=[32, 64, 128],      dropout=0.3, model_type="freq_aware"),
    "cnn_10": dict(filters=[32, 64, 128],      dropout=0.3, model_type="deep_head"),
}

CL_CONFIGS: dict[str, dict] = {
    "cl_1": dict(cnn_blocks=2, lstm_hidden=128, lstm_layers=1, bidirectional=True,  dropout=0.3),
    "cl_2": dict(cnn_blocks=2, lstm_hidden=256, lstm_layers=1, bidirectional=True,  dropout=0.3),
    "cl_3": dict(cnn_blocks=2, lstm_hidden=128, lstm_layers=2, bidirectional=True,  dropout=0.3),
    "cl_4": dict(cnn_blocks=3, lstm_hidden=128, lstm_layers=1, bidirectional=True,  dropout=0.3),
    "cl_5": dict(cnn_blocks=2, lstm_hidden=128, lstm_layers=1, bidirectional=False, dropout=0.3),
    "cl_6": dict(cnn_blocks=2, lstm_hidden=128, lstm_layers=1, bidirectional=True,  dropout=0.3),
}

WAV2VEC2_CHECKPOINTS = [
    "wav2vec2_finetuned_best.pt",
    "wav2vec2_frozen_best.pt",
]
WAV2VEC2_MODEL_NAME = "facebook/wav2vec2-base"

CNN_FILTER_BANK = [32, 64, 128, 256]   # first N entries selected by cnn_blocks

# ============================================================================
#  Noise conditions
# ============================================================================

def add_gaussian_snr(audio: np.ndarray, snr_db: float) -> np.ndarray:
    signal_power = np.mean(audio ** 2)
    if signal_power < 1e-10:
        return audio
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.random.randn(len(audio)) * np.sqrt(noise_power)
    return np.clip(audio + noise, -1.0, 1.0).astype(np.float32)


def apply_reverb(audio: np.ndarray, sr: int, rt60: float) -> np.ndarray:
    n_rir    = int(rt60 * sr)
    t        = np.arange(n_rir) / sr
    rir      = np.exp(-6.908 * t / rt60) * np.random.randn(n_rir)
    rir     /= (np.linalg.norm(rir) + 1e-8)
    reverbed = np.convolve(audio, rir)[: len(audio)]
    rms_orig = np.sqrt(np.mean(audio ** 2) + 1e-10)
    rms_rev  = np.sqrt(np.mean(reverbed ** 2) + 1e-10)
    reverbed = reverbed * (rms_orig / rms_rev)
    return np.clip(reverbed, -1.0, 1.0).astype(np.float32)


CONDITIONS: dict[str, callable] = {
    "gauss_20": lambda a: add_gaussian_snr(a, 20.0),
    "gauss_10": lambda a: add_gaussian_snr(a, 10.0),
    "gauss_5":  lambda a: add_gaussian_snr(a,  5.0),
    "gauss_0":  lambda a: add_gaussian_snr(a,  0.0),
    "room_03":  lambda a: apply_reverb(a, SR, 0.3),
    "room_06":  lambda a: apply_reverb(a, SR, 0.6),
}

GAUSSIAN_CONDS = ["gauss_20", "gauss_10", "gauss_5", "gauss_0"]
REVERB_CONDS   = ["room_03", "room_06"]
ALL_CONDS      = list(CONDITIONS.keys())

# ============================================================================
#  Spectral-subtraction denoiser (in-memory, no files written)
# ============================================================================

def spectral_subtraction_denoise(
    wav: np.ndarray,
    sr: int,  # noqa: ARG001  (kept for API compatibility)
    noise_est_frames: int = 10,
    alpha: float = 2.0,
) -> np.ndarray:
    """
    Simple spectral-subtraction denoiser.

    Estimate noise PSD from the first `noise_est_frames` STFT frames,
    subtract alpha * noise_PSD from signal PSD, floor at 0 to avoid
    musical noise, reconstruct via ISTFT using original phase.
    """
    orig_len   = len(wav)
    n_fft      = 512
    hop_length = 128
    stft       = librosa.stft(wav, n_fft=n_fft, hop_length=hop_length)
    magnitude  = np.abs(stft)
    phase      = np.angle(stft)
    noise_psd  = np.mean(magnitude[:, :noise_est_frames], axis=1, keepdims=True)
    mag_den    = np.maximum(magnitude - alpha * noise_psd, 0.0)
    stft_den   = mag_den * np.exp(1j * phase)
    wav_den    = librosa.istft(stft_den, hop_length=hop_length, length=orig_len)
    return wav_den.astype(np.float32)

# ============================================================================
#  Feature extraction helpers
# ============================================================================

def wav_to_logmel(audio: np.ndarray) -> np.ndarray:
    """Compute log-mel spectrogram from waveform → shape (1, 128, 300)."""
    S    = librosa.feature.melspectrogram(
        y=audio, sr=SR, n_mels=N_MELS,
        hop_length=HOP_LENGTH, win_length=WIN_LENGTH, n_fft=N_FFT_SPEC,
    )
    S_db = librosa.power_to_db(S, ref=np.max)
    T    = S_db.shape[1]
    if T < N_FRAMES:
        S_db = np.pad(S_db, ((0, 0), (0, N_FRAMES - T)),
                      mode="constant", constant_values=S_db.min())
    else:
        S_db = S_db[:, :N_FRAMES]
    return S_db[np.newaxis, :, :].astype(np.float32)  # (1, 128, 300)


def extract_272d(audio: np.ndarray) -> np.ndarray:
    """Extract 272D pyAudioAnalysis features (unscaled) from waveform."""
    win_samp  = int(ST_WIN  * SR)
    step_samp = int(ST_STEP * SR)
    F, _      = ShortTermFeatures.feature_extraction(audio, SR, win_samp, step_samp)
    return np.concatenate([
        np.mean(F, axis=1), np.min(F, axis=1),
        np.max(F, axis=1),  np.std(F, axis=1),
    ]).astype(np.float32)

# ============================================================================
#  Path helper
# ============================================================================

def remap_path(docker_path: str, dataset_root: Path) -> Path:
    rel = docker_path.replace("/workspace/", "").replace("/", "\\")
    return dataset_root / rel

# ============================================================================
#  Metrics
# ============================================================================

def compute_metrics(y_true: list, y_pred: list) -> dict:
    return {
        "uar": float(recall_score(y_true, y_pred, average="macro",    zero_division=0)),
        "wf1": float(f1_score    (y_true, y_pred, average="weighted", zero_division=0)),
    }

# ============================================================================
#  Model architectures  (verbatim mirrors of training scripts)
# ============================================================================

# ---- MLP ------------------------------------------------------------------ #

class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden: int, n_layers: int,
                 dropout: float, n_classes: int = 4):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = input_dim
        for _ in range(n_layers):
            layers += [nn.Linear(in_dim, hidden), nn.BatchNorm1d(hidden),
                       nn.ReLU(), nn.Dropout(dropout)]
            in_dim = hidden
        layers.append(nn.Linear(hidden, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---- CNN ------------------------------------------------------------------ #

def _conv_blocks(filters: list) -> nn.Sequential:
    layers: list[nn.Module] = []
    in_ch = 1
    for out_ch in filters:
        layers += [nn.Conv2d(in_ch, out_ch, 3, padding=1),
                   nn.BatchNorm2d(out_ch), nn.ReLU(), nn.MaxPool2d(2, 2)]
        in_ch = out_ch
    return nn.Sequential(*layers)


class CNN(nn.Module):
    def __init__(self, filters: list, dropout: float, n_classes: int = 4):
        super().__init__()
        self.cnn        = _conv_blocks(filters)
        self.pool       = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(filters[-1], 256), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(256, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.pool(self.cnn(x)))


class GlobalAvgPoolCNN(nn.Module):
    def __init__(self, filters: list, dropout: float, n_classes: int = 4):
        super().__init__()
        self.cnn        = _conv_blocks(filters)
        self.classifier = nn.Sequential(
            nn.Linear(filters[-1], 256), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(256, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.cnn(x).mean(dim=[2, 3]))


class _ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.main     = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch), nn.ReLU(), nn.MaxPool2d(2, 2),
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False), nn.MaxPool2d(2, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x) + self.shortcut(x)


class ResidualCNN(nn.Module):
    def __init__(self, filters: list, dropout: float, n_classes: int = 4):
        super().__init__()
        blocks: list[nn.Module] = []
        in_ch = 1
        for out_ch in filters:
            blocks.append(_ResBlock(in_ch, out_ch))
            in_ch = out_ch
        self.cnn        = nn.Sequential(*blocks)
        self.pool       = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(filters[-1], 256), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(256, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.pool(self.cnn(x)))


class FreqAwareCNN(nn.Module):
    FREQ_BINS = 4

    def __init__(self, filters: list, dropout: float, n_classes: int = 4):
        super().__init__()
        self.cnn        = _conv_blocks(filters)
        self.pool       = nn.AdaptiveAvgPool2d((self.FREQ_BINS, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(filters[-1] * self.FREQ_BINS, 256), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(256, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.pool(self.cnn(x)))


class DeepHeadCNN(nn.Module):
    def __init__(self, filters: list, dropout: float, n_classes: int = 4):
        super().__init__()
        self.cnn        = _conv_blocks(filters)
        self.pool       = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(filters[-1], 512), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(dropout), nn.Linear(256, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.pool(self.cnn(x)))


_CNN_REGISTRY: dict[str, type] = {
    "standard":  CNN,
    "gap":        GlobalAvgPoolCNN,
    "residual":   ResidualCNN,
    "freq_aware": FreqAwareCNN,
    "deep_head":  DeepHeadCNN,
}


def build_cnn(cfg: dict) -> nn.Module:
    return _CNN_REGISTRY[cfg["model_type"]](cfg["filters"], cfg["dropout"])


# ---- CNN-LSTM ------------------------------------------------------------- #

class AttentionPool(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scores  = self.attn(x).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        return (weights.unsqueeze(-1) * x).sum(dim=1)


class CNNLSTM(nn.Module):
    def __init__(self, cnn_blocks: int, lstm_hidden: int, lstm_layers: int,
                 bidirectional: bool, dropout: float, n_classes: int = 4):
        super().__init__()
        filters = CNN_FILTER_BANK[:cnn_blocks]
        cnn_layers: list[nn.Module] = []
        in_ch = 1
        for out_ch in filters:
            cnn_layers += [nn.Conv2d(in_ch, out_ch, 3, padding=1),
                           nn.BatchNorm2d(out_ch), nn.ReLU(), nn.MaxPool2d(2, 2)]
            in_ch = out_ch
        self.cnn  = nn.Sequential(*cnn_layers)
        self.lstm = nn.LSTM(
            input_size=filters[-1], hidden_size=lstm_hidden,
            num_layers=lstm_layers, batch_first=True, bidirectional=bidirectional,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        lstm_out        = lstm_hidden * (2 if bidirectional else 1)
        self.attention  = AttentionPool(lstm_out)
        self.classifier = nn.Sequential(
            nn.Linear(lstm_out, lstm_out // 2), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(lstm_out // 2, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.cnn(x)
        feat = feat.mean(dim=2)       # avg over frequency axis
        feat = feat.permute(0, 2, 1)  # (B, T', C)
        out, _ = self.lstm(feat)
        ctx = self.attention(out)
        return self.classifier(ctx)


# ---- wav2vec2 ------------------------------------------------------------- #

class _AttentionPoolingW2V(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.score = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor):
        scores  = self.score(x).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        context = (weights.unsqueeze(-1) * x).sum(dim=1)
        return context, weights


class Wav2Vec2SER(nn.Module):
    def __init__(self, model_name: str, dropout: float = 0.3, n_classes: int = 4):
        super().__init__()
        from transformers import Wav2Vec2Model
        self.wav2vec2   = Wav2Vec2Model.from_pretrained(model_name)
        hidden          = self.wav2vec2.config.hidden_size   # 768
        self.pool       = _AttentionPoolingW2V(hidden)
        self.classifier = nn.Sequential(
            nn.Linear(hidden, 256), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(256, n_classes),
        )

    def forward(self, input_values: torch.Tensor,
                attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        out    = self.wav2vec2(input_values, attention_mask=attention_mask)
        ctx, _ = self.pool(out.last_hidden_state)
        return self.classifier(ctx)

# ============================================================================
#  Waveform loading
# ============================================================================

def load_eval_waveforms(df_eval: pd.DataFrame, dataset_root: Path
                        ) -> tuple[list[np.ndarray], list[int], list[int]]:
    """
    Load raw WAVs for every row in df_eval.
    Returns (wavs, labels, valid_row_indices) — rows where the WAV was found.
    """
    wavs:   list[np.ndarray] = []
    labels: list[int]        = []
    valid:  list[int]        = []
    missing = 0

    for idx, row in tqdm(df_eval.iterrows(), total=len(df_eval), desc="Loading WAVs"):
        wav_path = remap_path(row["file_path"], dataset_root)
        if not wav_path.exists():
            missing += 1
            continue
        audio, _ = librosa.load(str(wav_path), sr=SR, mono=True)
        wavs.append(audio.astype(np.float32))
        labels.append(LABEL2IDX[row["label"]])
        valid.append(idx)

    if missing:
        print(f"  WARNING: {missing} WAV files not found — skipped.")
    print(f"  Loaded {len(wavs)} / {len(df_eval)} eval samples.")
    return wavs, labels, valid

# ============================================================================
#  Generic batch inference
# ============================================================================

def _infer_mlp(model: MLP, X: np.ndarray, device: torch.device,
               batch_size: int = 256) -> list[int]:
    model.eval()
    preds: list[int] = []
    for i in range(0, len(X), batch_size):
        Xb = torch.from_numpy(X[i:i + batch_size]).float().to(device)
        with torch.no_grad():
            preds.extend(model(Xb).argmax(1).cpu().tolist())
    return preds


def _infer_spec(model: nn.Module, specs: list[np.ndarray], device: torch.device,
                batch_size: int = 32) -> list[int]:
    model.eval()
    preds: list[int] = []
    for i in range(0, len(specs), batch_size):
        Xb = torch.from_numpy(np.stack(specs[i:i + batch_size])).to(device)
        with torch.no_grad():
            preds.extend(model(Xb).argmax(1).cpu().tolist())
    return preds


def _infer_wav2vec2(model: Wav2Vec2SER, extractor,
                    wavs: list[np.ndarray], device: torch.device,
                    batch_size: int = 8) -> list[int]:
    model.eval()
    preds: list[int] = []
    for i in range(0, len(wavs), batch_size):
        batch = [w.tolist() for w in wavs[i:i + batch_size]]
        inputs = extractor(
            batch, sampling_rate=SR, max_length=MAX_WAV2VEC,
            truncation=True, padding="max_length",
            return_attention_mask=True, return_tensors="pt",
        )
        iv = inputs["input_values"].to(device)
        am = inputs["attention_mask"].to(device)
        with torch.no_grad():
            preds.extend(model(iv, am).argmax(1).cpu().tolist())
    return preds

# ============================================================================
#  Condition evaluation  (clean / noised / denoised)
# ============================================================================

def eval_condition(
    family:  str,
    model:   nn.Module,
    wavs:    list[np.ndarray],
    labels:  list[int],
    device:  torch.device,
    cond_fn: callable | None = None,
    denoise: bool = False,
    scaler=None,
    extractor=None,
) -> dict:
    """
    Apply cond_fn (noise) and optionally denoise, then run model inference.

    cond_fn=None → clean pass (no noise, no denoising).
    """
    np.random.seed(42)   # reproducible noise

    processed: list[np.ndarray] = []
    for wav in wavs:
        if cond_fn is not None:
            wav = cond_fn(wav)
        if denoise:
            wav = spectral_subtraction_denoise(wav, SR)
        processed.append(wav)

    if family == "mlp":
        feats = np.stack([extract_272d(w) for w in processed])
        X = scaler.transform(feats).astype(np.float32)
        preds = _infer_mlp(model, X, device)

    elif family in ("cnn", "cnn_lstm"):
        specs = [wav_to_logmel(w) for w in processed]
        preds = _infer_spec(model, specs, device)

    elif family == "wav2vec2":
        preds = _infer_wav2vec2(model, extractor, processed, device)

    else:
        raise ValueError(f"Unknown family: {family!r}")

    return compute_metrics(labels, preds)

# ============================================================================
#  Best-config selection helpers
# ============================================================================

def _find_best_mlp(
    wavs: list[np.ndarray], labels: list[int],
    device: torch.device, input_dim: int, scaler,
) -> tuple[str, dict, nn.Module]:
    """Evaluate all MLP checkpoints on clean data; return (best_name, cfg, model)."""
    model_dir = MODEL_ROOT / "mlp"
    best_name, best_uar, best_model, best_cfg = None, -1.0, None, None

    # Pre-compute clean scaled features once — identical across configs.
    print("    Pre-computing 272D features for clean eval …")
    feats = np.stack([extract_272d(w) for w in tqdm(wavs, leave=False)])
    X_clean = scaler.transform(feats).astype(np.float32)

    for name, cfg in MLP_CONFIGS.items():
        ckpt = model_dir / f"{name}_best.pt"
        if not ckpt.exists():
            continue
        model = MLP(input_dim, cfg["hidden"], cfg["n_layers"], cfg["dropout"])
        model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
        model.to(device).eval()

        preds = _infer_mlp(model, X_clean, device)
        uar   = compute_metrics(labels, preds)["uar"]
        print(f"    {name}  clean UAR={uar:.4f}")

        if uar > best_uar:
            best_uar, best_name, best_model, best_cfg = uar, name, model, cfg

    return best_name, best_cfg, best_model


def _find_best_cnn(
    wavs: list[np.ndarray], labels: list[int], device: torch.device,
) -> tuple[str, dict, nn.Module]:
    model_dir = MODEL_ROOT / "cnn"
    best_name, best_uar, best_model, best_cfg = None, -1.0, None, None

    # Pre-compute clean mel-spectrograms once.
    print("    Pre-computing mel-spectrograms for clean eval …")
    specs_clean = [wav_to_logmel(w) for w in tqdm(wavs, leave=False)]

    for name, cfg in CNN_CONFIGS.items():
        ckpt = model_dir / f"{name}_best.pt"
        if not ckpt.exists():
            continue
        model = build_cnn(cfg)
        model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
        model.to(device).eval()

        preds = _infer_spec(model, specs_clean, device)
        uar   = compute_metrics(labels, preds)["uar"]
        print(f"    {name}  clean UAR={uar:.4f}")

        if uar > best_uar:
            best_uar, best_name, best_model, best_cfg = uar, name, model, cfg

    return best_name, best_cfg, best_model


def _find_best_cnn_lstm(
    wavs: list[np.ndarray], labels: list[int], device: torch.device,
) -> tuple[str, dict, nn.Module]:
    model_dir = MODEL_ROOT / "cnn_lstm"
    best_name, best_uar, best_model, best_cfg = None, -1.0, None, None

    # Reuse mel-spectrograms computed once (same params as CNN).
    print("    Pre-computing mel-spectrograms for clean eval …")
    specs_clean = [wav_to_logmel(w) for w in tqdm(wavs, leave=False)]

    for name, cfg in CL_CONFIGS.items():
        ckpt = model_dir / f"{name}_best.pt"
        if not ckpt.exists():
            continue
        model = CNNLSTM(**cfg)
        model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
        model.to(device).eval()

        preds = _infer_spec(model, specs_clean, device)
        uar   = compute_metrics(labels, preds)["uar"]
        print(f"    {name}  clean UAR={uar:.4f}")

        if uar > best_uar:
            best_uar, best_name, best_model, best_cfg = uar, name, model, cfg

    return best_name, best_cfg, best_model


def _find_best_wav2vec2(
    wavs: list[np.ndarray], labels: list[int], device: torch.device,
) -> tuple[str | None, nn.Module | None, object | None]:
    """Load wav2vec2 model and feature extractor; pick best checkpoint by clean UAR."""
    try:
        from transformers import Wav2Vec2FeatureExtractor
    except ImportError:
        print("  transformers not available — skipping wav2vec2.")
        return None, None, None

    model_dir  = MODEL_ROOT / "wav2vec2"
    extractor  = Wav2Vec2FeatureExtractor.from_pretrained(WAV2VEC2_MODEL_NAME)
    best_name, best_uar, best_model = None, -1.0, None

    for ckpt_file in WAV2VEC2_CHECKPOINTS:
        ckpt = model_dir / ckpt_file
        if not ckpt.exists():
            continue
        model = Wav2Vec2SER(WAV2VEC2_MODEL_NAME)
        state = torch.load(ckpt, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        model.to(device).eval()

        preds = _infer_wav2vec2(model, extractor, wavs, device)
        uar   = compute_metrics(labels, preds)["uar"]
        print(f"    {ckpt_file}  clean UAR={uar:.4f}")

        if uar > best_uar:
            best_uar, best_name, best_model = uar, ckpt_file, model

    return best_name, best_model, extractor

# ============================================================================
#  Full family evaluation (clean + all noise conditions × noised + denoised)
# ============================================================================

def eval_family(
    display_name: str,
    family:       str,
    best_model:   nn.Module,
    wavs:         list[np.ndarray],
    labels:       list[int],
    device:       torch.device,
    scaler=None,
    extractor=None,
) -> dict:
    """
    Returns:
    {
      "clean": {"uar": float, "wf1": float},
      "gauss_20": {"noised": {"uar": ..., "wf1": ...},
                   "denoised": {"uar": ..., "wf1": ...}},
      ...
    }
    """
    results: dict = {}

    print(f"  [{display_name}] clean …", end=" ", flush=True)
    results["clean"] = eval_condition(
        family, best_model, wavs, labels, device,
        cond_fn=None, denoise=False, scaler=scaler, extractor=extractor,
    )
    print(f"UAR={results['clean']['uar']:.4f}")

    for cond_name, cond_fn in CONDITIONS.items():
        results[cond_name] = {}

        print(f"  [{display_name}] {cond_name:10s} noised   …", end=" ", flush=True)
        results[cond_name]["noised"] = eval_condition(
            family, best_model, wavs, labels, device,
            cond_fn=cond_fn, denoise=False, scaler=scaler, extractor=extractor,
        )
        print(f"UAR={results[cond_name]['noised']['uar']:.4f}")

        print(f"  [{display_name}] {cond_name:10s} denoised …", end=" ", flush=True)
        results[cond_name]["denoised"] = eval_condition(
            family, best_model, wavs, labels, device,
            cond_fn=cond_fn, denoise=True, scaler=scaler, extractor=extractor,
        )
        print(f"UAR={results[cond_name]['denoised']['uar']:.4f}")

    return results

# ============================================================================
#  Summary CSV
# ============================================================================

def build_summary_csv(all_results: dict[str, dict]) -> pd.DataFrame:
    """
    Columns: model, best_config, clean_UAR, clean_WF1,
             {cond}_noised_UAR, {cond}_noised_WF1,
             {cond}_denoised_UAR, {cond}_denoised_WF1, ...
    """
    rows = []
    for display_name, (best_cfg_name, res) in all_results.items():
        row: dict = {"model": display_name, "best_config": best_cfg_name}
        row["clean_UAR"] = res["clean"]["uar"]
        row["clean_WF1"] = res["clean"]["wf1"]
        for cond in ALL_CONDS:
            row[f"{cond}_noised_UAR"]   = res[cond]["noised"]["uar"]
            row[f"{cond}_noised_WF1"]   = res[cond]["noised"]["wf1"]
            row[f"{cond}_denoised_UAR"] = res[cond]["denoised"]["uar"]
            row[f"{cond}_denoised_WF1"] = res[cond]["denoised"]["wf1"]
        rows.append(row)

    df = pd.DataFrame(rows)
    out = RESULT_DIR / "robustness_summary.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved: {out}")
    return df

# ============================================================================
#  Visualisations
# ============================================================================

FAMILY_COLORS = {
    "MLP":       "#4C72B0",
    "CNN":       "#DD8452",
    "CNN-LSTM":  "#55A868",
    "wav2vec2":  "#C44E52",
}
FAMILY_MARKERS = {
    "MLP": "o", "CNN": "s", "CNN-LSTM": "^", "wav2vec2": "D",
}


def _uar_series(res: dict, cond_names: list[str], tag: str) -> list[float]:
    """Extract noised or denoised UAR for a list of conditions."""
    return [res[c][tag]["uar"] for c in cond_names]


def plot_robustness_curves(all_results: dict[str, dict]) -> None:
    """
    Two sub-plots: Gaussian SNR (left) and Reverberation (right).
    For each model family: clean (horizontal dotted), noised (solid), denoised (dashed).
    """
    gauss_x      = [20, 10, 5, 0]      # SNR dB — severity increases left to right
    reverb_x     = [0.3, 0.6]          # RT60 s

    fig, (ax_g, ax_r) = plt.subplots(1, 2, figsize=(14, 5))

    for display_name, (_, res) in all_results.items():
        color  = FAMILY_COLORS.get(display_name, "grey")
        marker = FAMILY_MARKERS.get(display_name, "o")
        clean_uar = res["clean"]["uar"]

        # --- Gaussian ---
        gauss_noised   = _uar_series(res, GAUSSIAN_CONDS, "noised")
        gauss_denoised = _uar_series(res, GAUSSIAN_CONDS, "denoised")

        ax_g.axhline(clean_uar, color=color, linewidth=1.2, linestyle=":",
                     label=f"{display_name} clean")
        ax_g.plot(gauss_x, gauss_noised,   color=color, linewidth=1.8,
                  linestyle="-",  marker=marker, label=f"{display_name} noised")
        ax_g.plot(gauss_x, gauss_denoised, color=color, linewidth=1.8,
                  linestyle="--", marker=marker, label=f"{display_name} denoised")

        # --- Reverb ---
        reverb_noised   = _uar_series(res, REVERB_CONDS, "noised")
        reverb_denoised = _uar_series(res, REVERB_CONDS, "denoised")

        ax_r.axhline(clean_uar, color=color, linewidth=1.2, linestyle=":",
                     label=f"{display_name} clean")
        ax_r.plot(reverb_x, reverb_noised,   color=color, linewidth=1.8,
                  linestyle="-",  marker=marker, label=f"{display_name} noised")
        ax_r.plot(reverb_x, reverb_denoised, color=color, linewidth=1.8,
                  linestyle="--", marker=marker, label=f"{display_name} denoised")

    # Gaussian axis: severity increases as SNR decreases
    ax_g.invert_xaxis()
    ax_g.set_xlabel("SNR (dB)")
    ax_g.set_ylabel("UAR")
    ax_g.set_title("Gaussian Noise")
    ax_g.set_xticks(gauss_x)
    ax_g.set_xticklabels([f"{v} dB" for v in gauss_x])
    ax_g.set_ylim(0, 1)
    ax_g.grid(True, alpha=0.3)
    ax_g.legend(fontsize=7, ncol=2, loc="lower left")

    ax_r.set_xlabel("RT60 (s)")
    ax_r.set_title("Reverberation")
    ax_r.set_xticks(reverb_x)
    ax_r.set_xticklabels([f"{v} s" for v in reverb_x])
    ax_r.set_ylim(0, 1)
    ax_r.grid(True, alpha=0.3)
    ax_r.legend(fontsize=7, ncol=2, loc="lower right")

    # Shared legend patch:  solid=noised, dashed=denoised, dotted=clean
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color="black", linewidth=1.8, linestyle=":",  label="clean"),
        Line2D([0], [0], color="black", linewidth=1.8, linestyle="-",  label="noised"),
        Line2D([0], [0], color="black", linewidth=1.8, linestyle="--", label="denoised"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, fontsize=9,
               title="Line style", bbox_to_anchor=(0.5, 1.02))

    plt.suptitle("Noise Robustness Curves", y=1.06, fontsize=12)
    plt.tight_layout()
    out = RESULT_DIR / "robustness_curves.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def plot_denoising_gain(all_results: dict[str, dict]) -> None:
    """
    Bar chart: UAR gain = denoised_UAR − noised_UAR per model per condition.
    Positive = denoiser helped.
    """
    families = list(all_results.keys())
    n_fam    = len(families)
    n_cond   = len(ALL_CONDS)
    bar_w    = 0.8 / n_fam

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(n_cond)

    for i, display_name in enumerate(families):
        _, res = all_results[display_name]
        gains  = [res[c]["denoised"]["uar"] - res[c]["noised"]["uar"] for c in ALL_CONDS]
        offsets = (i - (n_fam - 1) / 2) * bar_w
        color   = FAMILY_COLORS.get(display_name, "grey")
        bars    = ax.bar(x + offsets, gains, width=bar_w, label=display_name, color=color,
                         edgecolor="white", linewidth=0.5)
        for bar, g in zip(bars, gains):
            if abs(g) > 0.005:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + (0.003 if g >= 0 else -0.012),
                        f"{g:+.3f}", ha="center", va="bottom", fontsize=6.5, rotation=45)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(ALL_CONDS, rotation=15, ha="right")
    ax.set_ylabel("UAR gain  (denoised − noised)")
    ax.set_title("Denoising Gain per Model per Condition")
    ax.legend(loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    out = RESULT_DIR / "denoising_gain.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def plot_heatmap(all_results: dict[str, dict]) -> None:
    """
    3-panel heatmap: Clean | Noised | Denoised
    Y-axis: model families.  X-axis: conditions (clean has 1 col, noised/denoised have 6).
    For the Clean panel the single value is tiled across all conditions for alignment.
    """
    families = list(all_results.keys())

    # Build matrices
    clean_mat  = np.zeros((len(families), len(ALL_CONDS)))
    noised_mat = np.zeros((len(families), len(ALL_CONDS)))
    den_mat    = np.zeros((len(families), len(ALL_CONDS)))

    for i, name in enumerate(families):
        _, res = all_results[name]
        clean_uar = res["clean"]["uar"]
        for j, cond in enumerate(ALL_CONDS):
            clean_mat[i, j]  = clean_uar                          # tiled for alignment
            noised_mat[i, j] = res[cond]["noised"]["uar"]
            den_mat[i, j]    = res[cond]["denoised"]["uar"]

    vmin = min(clean_mat.min(), noised_mat.min(), den_mat.min()) - 0.02
    vmax = max(clean_mat.max(), noised_mat.max(), den_mat.max()) + 0.02
    vmin, vmax = max(0.0, vmin), min(1.0, vmax)

    fig, axes = plt.subplots(1, 3, figsize=(16, max(3, len(families) * 0.9 + 1.5)),
                              constrained_layout=True)

    for ax, mat, title in zip(
        axes,
        [clean_mat, noised_mat, den_mat],
        ["Clean (baseline)", "Noised", "Denoised"],
    ):
        df_hm = pd.DataFrame(mat, index=families, columns=ALL_CONDS)
        sns.heatmap(
            df_hm, ax=ax, annot=True, fmt=".3f", cmap="RdYlGn",
            vmin=vmin, vmax=vmax, linewidths=0.4,
            cbar_kws={"shrink": 0.6},
        )
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Condition")
        ax.set_ylabel("Model" if ax == axes[0] else "")
        ax.tick_params(axis="x", rotation=20)
        ax.tick_params(axis="y", rotation=0)

    fig.suptitle("Noise Robustness Heatmap (UAR)", fontsize=13, fontweight="bold")
    out = RESULT_DIR / "robustness_heatmap.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")

# ============================================================================
#  Main
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 7 — Noise robustness + denoising evaluation"
    )
    parser.add_argument(
        "--dataset-root", default=DEFAULT_DATASET_ROOT,
        help="Root directory containing datasets/iemocap/Session* WAV files.",
    )
    parser.add_argument(
        "--split-dir", default=DEFAULT_SPLIT_DIR,
        help="Subfolder under features/splits/ to read eval.csv from (default: loso).",
    )
    parser.add_argument(
        "--skip-wav2vec2", action="store_true",
        help="Skip wav2vec2 evaluation (useful if transformers / GPU RAM unavailable).",
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    splits_dir   = DL_ROOT / "features" / "splits" / args.split_dir
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device     : {device}")
    print(f"Split dir  : {splits_dir}")
    print(f"Dataset    : {dataset_root}\n")

    # ------------------------------------------------------------------
    # Load eval set + raw WAVs
    # ------------------------------------------------------------------
    df_eval = pd.read_csv(splits_dir / "eval.csv")
    scaler  = joblib.load(splits_dir / "scaler.pkl")

    print("Loading eval WAVs …")
    wavs, labels, _ = load_eval_waveforms(df_eval, dataset_root)
    if not wavs:
        print("ERROR: No WAV files loaded. Check --dataset-root.")
        return

    feat_cols  = [c for c in df_eval.columns if c not in NON_FEAT]
    input_dim  = len(feat_cols)
    print(f"  Feature dim (MLP): {input_dim}\n")

    all_results: dict[str, tuple[str, dict]] = {}

    # ------------------------------------------------------------------
    # MLP
    # ------------------------------------------------------------------
    print("=" * 55)
    print("  MLP — finding best config …")
    best_name, best_cfg, best_model = _find_best_mlp(
        wavs, labels, device, input_dim, scaler
    )
    if best_model is not None:
        print(f"  → best: {best_name}\n")
        res = eval_family("MLP", "mlp", best_model, wavs, labels,
                          device, scaler=scaler)
        all_results["MLP"] = (best_name, res)
    else:
        print("  No MLP checkpoints found — skipping.\n")

    # ------------------------------------------------------------------
    # CNN
    # ------------------------------------------------------------------
    print("=" * 55)
    print("  CNN — finding best config …")
    best_name, best_cfg, best_model = _find_best_cnn(wavs, labels, device)
    if best_model is not None:
        print(f"  → best: {best_name}\n")
        res = eval_family("CNN", "cnn", best_model, wavs, labels, device)
        all_results["CNN"] = (best_name, res)
    else:
        print("  No CNN checkpoints found — skipping.\n")

    # ------------------------------------------------------------------
    # CNN-LSTM
    # ------------------------------------------------------------------
    print("=" * 55)
    print("  CNN-LSTM — finding best config …")
    best_name, best_cfg, best_model = _find_best_cnn_lstm(wavs, labels, device)
    if best_model is not None:
        print(f"  → best: {best_name}\n")
        res = eval_family("CNN-LSTM", "cnn_lstm", best_model, wavs, labels, device)
        all_results["CNN-LSTM"] = (best_name, res)
    else:
        print("  No CNN-LSTM checkpoints found — skipping.\n")

    # ------------------------------------------------------------------
    # wav2vec2
    # ------------------------------------------------------------------
    if not args.skip_wav2vec2:
        print("=" * 55)
        print("  wav2vec2 — loading model …")
        best_name, best_model, extractor = _find_best_wav2vec2(wavs, labels, device)
        if best_model is not None:
            print(f"  → best: {best_name}\n")
            res = eval_family("wav2vec2", "wav2vec2", best_model, wavs, labels,
                              device, extractor=extractor)
            all_results["wav2vec2"] = (best_name, res)
        else:
            print("  No wav2vec2 checkpoints found — skipping.\n")

    if not all_results:
        print("No results produced.  Exiting.")
        return

    # ------------------------------------------------------------------
    # Outputs
    # ------------------------------------------------------------------
    print("\n" + "=" * 55)
    print("  Building outputs …")

    df_summary = build_summary_csv(all_results)

    print("\n--- Robustness Summary (UAR) ---")
    print(df_summary[["model", "best_config", "clean_UAR"]].to_string(index=False))

    plot_robustness_curves(all_results)
    plot_denoising_gain(all_results)
    plot_heatmap(all_results)

    print("\nDone.  All outputs in:", RESULT_DIR)


if __name__ == "__main__":
    main()
