# Deep Learning SER — Implementation Plan

## Στόχος
Συστηματική σύγκριση 4 αρχιτεκτονικών deep learning για Speech Emotion Recognition (SER) στο IEMOCAP dataset, με έμφαση στο noise robustness και την κατανόηση της συμπεριφοράς κάθε μοντέλου.

---

## Βήμα 0 — Setup & Dependencies

**Script:** `dl_00_setup_workflow.py`

**Νέα packages (προσθήκη στο requirements.txt):**
- `transformers` (HuggingFace — για wav2vec2)
- `audiomentations` (για noise augmentation)
- `torch` + `torchaudio` ήδη υπάρχουν ✓

**Νέα folder structure:**
```
workflows/iemocap_dl/
  features/splits/80_20/   ← copy των υπαρχόντων train.csv + test.csv
  spectrograms/train/
  spectrograms/test/
  models/mlp/
  models/cnn/
  models/cnn_lstm/
  models/wav2vec2/
  results/mlp/
  results/cnn/
  results/cnn_lstm/
  results/wav2vec2/
  results/noise_robustness/
```

**Reuse από ML εξάμηνο:**
- `train.csv` + `test.csv` (80/20 split) ✓
- `iemocap_features.csv` (272D) ✓
- `cremad_features.csv` ✓
- Docker + base requirements.txt ✓

---

## Βήμα 1 — Data Preparation (παράλληλα)

### 1A — 272D Features (υπάρχουν ήδη)
Χρήση του υπάρχοντος `iemocap_features.csv`. Κανένα νέο script δεν χρειάζεται για τα clean features.

### 1B — Mel-Spectrogram Extraction
**Script:** `dl_01_extract_spectrograms.py`

- Input: WAV paths από `train.csv` + `test.csv`
- `torchaudio.load()` → Resample 16kHz
- `MelSpectrogram`: n_mels=128, hop_length=160 (10ms), win_length=400 (25ms), n_fft=1024
- Log scale (`amplitude_to_DB`)
- Pad/truncate → 3 δευτερόλεπτα = 300 time frames
- Shape τελικό: `(1, 128, 300)`
- Αποθήκευση ανά αρχείο ως `.npy`
- Output: `train_manifest.csv` + `test_manifest.csv`

### 1C — Raw Audio για wav2vec2
Δεν χρειάζεται pre-extraction. Το wav2vec2 Dataset class φορτώνει raw WAV on-the-fly, resample 16kHz, max length 10s, μέσω `AutoFeatureExtractor`.

---

## Βήμα 2 — Noise Augmentation (TRAIN ONLY)

**Script:** `dl_02_noise_augment.py`

> ⚠️ DATA LEAKAGE: Augmented δείγματα ΜΟΝΟ στο train set. Test set παραμένει clean και untouched.

**Τύποι augmentation (audiomentations):**
- `AddGaussianNoise`: SNR = 5dB, 10dB, 20dB
- `AddBackgroundNoise`: SNR = 10dB (MUSAN dataset)
- `RoomSimulator`: RT60 = 0.3s, 0.6s

**Output για MLP:**
- Augmented WAVs → `datasets/iemocap_augmented/`
- Εξαγωγή 272D features (pyAudioAnalysis, ίδιες παράμετροι)
- `train_augmented_features.csv`

**Output για CNN/CNN-LSTM:**
- Mel-spectrograms από augmented WAVs (.npy)
- `train_augmented_manifest.csv`

**Για wav2vec2:**
- On-the-fly augmentation στο Dataset `__getitem__`

**Για Evaluation (Βήμα 7):**
- Noisy test versions δημιουργούνται on-the-fly, ΔΕΝ αποθηκεύονται

---

## Βήμα 3 — Architecture 1: Hand-crafted Features + MLP

**Script:** `dl_03_train_mlp.py`

**Architecture:**
```
Input(272) → [Linear → BatchNorm → ReLU → Dropout] × N → Linear(4)
```

**Hyperparameter Grid (6 configs):**

| Config | Layers | Hidden | Dropout | LR    | Notes |
|--------|--------|--------|---------|-------|-------|
| MLP-1  | 2      | 256    | 0.3     | 1e-3  | baseline |
| MLP-2  | 3      | 256    | 0.3     | 1e-3  | +depth |
| MLP-3  | 3      | 512    | 0.3     | 1e-3  | +width |
| MLP-4  | 3      | 512    | 0.5     | 1e-3  | +dropout |
| MLP-5  | 4      | 512    | 0.3     | 5e-4  | deeper+slowLR |
| MLP-6★ | 3      | 256    | 0.3     | 1e-3  | + augmented data |

★ MLP-6 = MLP-2 αρχιτεκτονική + augmented train set

**Training:**
- Loss: CrossEntropyLoss
- Optimizer: Adam
- Scheduler: CosineAnnealingLR
- Early stopping: patience=10 (val F1)
- Validation: 10% του train (stratified)
- Metrics: Accuracy, Weighted F1, Macro F1, UAR, Confusion Matrix

---

## Βήμα 4 — Architecture 2: Mel-spectrogram + CNN

**Script:** `dl_04_train_cnn.py`

**Architecture:**
```
Input(1, 128, 300)
→ [Conv2d(in, out, 3×3) → BatchNorm → ReLU → MaxPool(2×2)] × N
→ AdaptiveAvgPool → Flatten
→ Linear → Dropout → Linear(4)
```

**Hyperparameter Grid (6 configs):**

| Config | Blocks | Filters          | Dropout | LR    | Notes |
|--------|--------|------------------|---------|-------|-------|
| CNN-1  | 2      | [32, 64]         | 0.3     | 1e-3  | baseline |
| CNN-2  | 3      | [32, 64, 128]    | 0.3     | 1e-3  | +depth |
| CNN-3  | 3      | [64, 128, 256]   | 0.3     | 1e-3  | +width |
| CNN-4  | 3      | [32, 64, 128]    | 0.5     | 1e-3  | +dropout |
| CNN-5  | 4      | [32,64,128,256]  | 0.3     | 5e-4  | deeper+slowLR |
| CNN-6★ | 3      | [32, 64, 128]    | 0.3     | 1e-3  | + augmented data |

★ CNN-6 = CNN-2 αρχιτεκτονική + augmented spectrograms

**Training:** Ίδιο setup με MLP (early stopping, cosine scheduler). Dataset class φορτώνει `.npy` on-the-fly.

---

## Βήμα 5 — Architecture 3: Mel-spectrogram + CNN-LSTM

**Script:** `dl_05_train_cnn_lstm.py`

**Architecture:**
```
Input(1, 128, T)
→ [Conv2d → BN → ReLU → MaxPool] × N     # CNN blocks
→ squeeze + permute → shape: (batch, T', C)  # time-first για LSTM
→ BiLSTM(input=C, hidden=H, layers=L)
→ Attention pooling over time steps
→ Linear(2H) → Dropout → Linear(4)
```

**Hyperparameter Grid (6 configs):**

| Config | CNN blocks | LSTM H | LSTM L | BiLSTM | Notes |
|--------|------------|--------|--------|--------|-------|
| CL-1   | 2          | 128    | 1      | True   | baseline |
| CL-2   | 2          | 256    | 1      | True   | +hidden |
| CL-3   | 2          | 128    | 2      | True   | +LSTM layers |
| CL-4   | 3          | 128    | 1      | True   | +CNN depth |
| CL-5   | 2          | 128    | 1      | False  | Uni-directional |
| CL-6★  | 2          | 128    | 1      | True   | + augmented data |

★ CL-6 = CL-1 + augmented spectrograms

**Report Analysis:** Σύγκριση CNN vs CNN-LSTM αναδεικνύει αν η χρονική μοντελοποίηση βοηθάει στο SER. Bi vs Uni-directional LSTM comparison.

---

## Βήμα 6 — Architecture 4: wav2vec2 + Attention Head (Upper Bound)

**Script:** `dl_06_train_wav2vec2.py`

**Architecture:**
```
facebook/wav2vec2-base (pretrained, HuggingFace)
  → feature extractor (CNN layers)
  → transformer (12 layers, hidden=768)
  → hidden states: (batch, T, 768)
  → Multi-head Self-Attention pooling → (batch, 768)
  → Linear(768 → 256) → ReLU → Dropout
  → Linear(256 → 4)
```

**Training Strategy:**
- Epochs 1-5: feature extractor frozen
- Epoch 6+: unfreeze, layerwise LR decay (transformer layers: μικρότερο LR)
- Optimizer: AdamW + weight decay
- ⚠️ Απαιτεί GPU — βαρύ μοντέλο

**Απαιτούμενη Ανάλυση (για report — ΟΧΙ απλό fine-tuning):**
1. **Attention weight visualization:** ποια time frames "κοιτάει" για κάθε emotion class
2. **Layer probing (layers 1-12):** train linear probe ανά layer → πότε "μαθαίνει" το emotion
3. **Frozen vs fine-tuned comparison:** πόσο βοηθάει το fine-tuning
4. **Αιτιολόγηση** γιατί υπερτερεί (ή όχι) έναντι CNN-LSTM

---

## Βήμα 7 — Noise Robustness Evaluation

**Script:** `dl_07_noise_robustness_eval.py`

**Input:** Best checkpoints από Βήματα 3-6 + clean test WAV files

**Noise conditions (on-the-fly, ΔΕΝ αποθηκεύονται):**
- Clean (baseline)
- Gaussian: SNR = 20dB, 10dB, 5dB, 0dB
- Background noise: SNR = 10dB
- Reverberation: RT60 = 0.3s, 0.6s

**Metrics ανά condition:** UAR, Weighted F1

**Output:**
- `robustness_curves.png` (x=SNR, y=UAR, 1 γραμμή/αρχιτεκτονική)
- `robustness_summary.csv`
- `per_model_degradation.png` (% drop from clean baseline)
- Heatmap: μοντέλο × noise condition

---

## Βήμα 8 — Aggregate Results & Comparison

**Script:** `dl_08_aggregate_results.py`

**Master Comparison Table:**

| Μοντέλο | Accuracy | W-F1 | Macro-F1 | UAR(clean) | UAR(10dB) | UAR(5dB) | #Params | Time |
|---------|----------|------|----------|------------|-----------|----------|---------|------|
| MLP-best | - | - | - | - | - | - | - | - |
| CNN-best | - | - | - | - | - | - | - | - |
| CNN-LSTM-best | - | - | - | - | - | - | - | - |
| wav2vec2 | - | - | - | - | - | - | - | - |

**Επιπλέον plots:**
- Hyperparameter sensitivity ανά αρχιτεκτονική (val F1 vs depth/size)
- Confusion matrices (best model ανά arch)
- Augmentation impact: clean-train vs aug-train

---

## Βήμα 9 — Report & Presentation

**Report (PROJECT_REPORT_DL.md / PDF):**
1. Εισαγωγή + στόχοι
2. Datasets (σύντομο — IEMOCAP ήδη στο ML report)
3. Αρχιτεκτονικές: περιγραφή + αιτιολόγηση
4. Noise Augmentation: strategy + data leakage prevention
5. Αποτελέσματα ανά architecture (hyperparameter sensitivity)
6. Noise Robustness analysis
7. wav2vec2 ανάλυση (attention, layer probing)
8. Σύγκριση + Συμπεράσματα

**Slides:** ~20 slides, αντίστοιχη δομή

---

## Σειρά Εκτέλεσης

```
Step 0 (setup)
    ↓
Step 1 (dl_01_extract_spectrograms.py)
    ↓
Step 2 (dl_02_noise_augment.py)
    ↓
Steps 3, 4, 5, 6 ⚡ παράλληλα
(dl_03 + dl_04 + dl_05 + dl_06)
    ↓
Step 7 (dl_07_noise_robustness_eval.py)
    ↓
Step 8 (dl_08_aggregate_results.py)
    ↓
Step 9 (Report + Slides)
```
