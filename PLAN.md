# Deep Learning SER — Implementation Plan

## Στόχος
Συστηματική σύγκριση 4 αρχιτεκτονικών deep learning για Speech Emotion Recognition (SER) στο IEMOCAP dataset, με έμφαση στο noise robustness και την κατανόηση της συμπεριφοράς κάθε μοντέλου.

---

## Βήμα 0 — Setup & Dependencies ✅ DONE

**Script:** `dl_00_setup_workflow.py`

**Packages που προστέθηκαν στο requirements.txt:**
- `transformers` (HuggingFace — για wav2vec2)
- `audiomentations` (για noise augmentation)
- `torch` ήδη υπήρχε ✓
- `torchaudio` ❌ αφαιρέθηκε — χρησιμοποιούμε `librosa` παντού (ίδια λειτουργικότητα, λιγότερες εξαρτήσεις)

**Folder structure που δημιουργήθηκε:**
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
- `train.csv` + `test.csv` (80/20 split, 4424 + 1107 δείγματα) ✓
- `iemocap_features.csv` (272D, 5531 rows) ✓
- WAV αρχεία: `D:\Users\User\Desktop\demokritos-ml-project\datasets\iemocap\`

> **Σημείωση:** Docker δεν χρησιμοποιείται — τρέχουμε locally (Python 3.14, torch+cpu 2.9.1).
> GPU (NVIDIA RTX 3060 12GB) θα ενεργοποιηθεί πριν το Βήμα 6 (wav2vec2).

---

## Βήμα 1 — Data Preparation ✅ DONE

### 1A — 272D Features (υπάρχουν ήδη)
Χρήση του υπάρχοντος `iemocap_features.csv`. Κανένα νέο script δεν χρειάζεται.

### 1B — Mel-Spectrogram Extraction ✅
**Script:** `dl_01_extract_spectrograms.py`

- Input: WAV paths από `train.csv` + `test.csv` (path remapping `/workspace` → local)
- `librosa.load()` → 16kHz (αντί torchaudio)
- `librosa.feature.melspectrogram()`: n_mels=128, hop_length=160 (10ms), win_length=400 (25ms), n_fft=1024
- `librosa.power_to_db()` (log scale)
- Pad/truncate → 300 time frames (3 δευτερόλεπτα)
- Shape τελικό: `(1, 128, 300)` — float32, range [-80, 0] dB
- Αποθήκευση ανά αρχείο ως `.npy`
- **Αποτέλεσμα:** 4424 train + 1107 test spectrograms, ~811 MB
- Output: `train_manifest.csv` + `test_manifest.csv`

### 1C — Raw Audio για wav2vec2
Δεν χρειάζεται pre-extraction. Το wav2vec2 Dataset class φορτώνει raw WAV on-the-fly.

---

## Βήμα 2 — Noise Augmentation (TRAIN ONLY) ✅ DONE

**Script:** `dl_02_noise_augment.py`

> ⚠️ DATA LEAKAGE: Augmented δείγματα ΜΟΝΟ στο train set. Test set παραμένει clean και untouched.

**Τύποι augmentation (5 conditions — χωρίς MUSAN):**

| Condition | Τύπος | Παράμετρος |
|-----------|-------|------------|
| gauss_20 | Gaussian noise | SNR = 20 dB |
| gauss_10 | Gaussian noise | SNR = 10 dB |
| gauss_5 | Gaussian noise | SNR = 5 dB |
| room_03 | Reverberation | RT60 = 0.3s |
| room_06 | Reverberation | RT60 = 0.6s |

> **Αλλαγή από plan:** Αφαιρέθηκε το `AddBackgroundNoise` (MUSAN dataset ~11GB).
> Αντικατάσταση reverberation: synthetic exponential-decay RIR (χωρίς pyroomacoustics).
> Augmented WAVs δεν αποθηκεύονται — επεξεργασία in-memory.

**Αποτέλεσμα:** 4424 × 5 = **22,120 augmented samples**

**Output για MLP:**
- Εξαγωγή 272D features (pyAudioAnalysis, ίδιες παράμετροι: ST_WIN=50ms, ST_STEP=25ms)
- `train_augmented_features.csv` (22,120 rows × 275 cols) — gitignored

**Output για CNN/CNN-LSTM:**
- Mel-spectrograms (.npy), shape (1, 128, 300) — gitignored (~3.3GB)
- `train_augmented_manifest.csv` — gitignored

**Για wav2vec2:** On-the-fly augmentation στο Dataset `__getitem__`

---

## Βήμα 3 — Architecture 1: Hand-crafted Features + MLP ✅ DONE

**Script:** `dl_03_train_mlp.py`

**Architecture:**
```
Input(272) → [Linear → BatchNorm → ReLU → Dropout] × N → Linear(4)
```

**Training setup:**
- Loss: CrossEntropyLoss | Optimizer: Adam | Scheduler: CosineAnnealingLR
- Early stopping: patience=10 (val weighted F1) | Val: 10% train (stratified)

### Αποτελέσματα — Run v1 (τελικό)

| Config | Layers | Hidden | Dropout | LR | Accuracy | W-F1 | UAR | Epochs |
|--------|--------|--------|---------|-----|----------|------|-----|--------|
| MLP-1 | 2 | 256 | 0.3 | 1e-3 | 62.8% | 0.627 | 0.626 | 18 |
| MLP-2 | 3 | 256 | 0.3 | 1e-3 | 61.2% | 0.609 | 0.630 | 19 |
| MLP-3 | 3 | 512 | 0.3 | 1e-3 | 59.9% | 0.599 | 0.602 | 34 |
| MLP-4 | 3 | 512 | 0.5 | 1e-3 | 62.5% | 0.623 | 0.629 | 20 |
| **MLP-5** ★ | **4** | **512** | **0.3** | **5e-4** | **61.9%** | **0.616** | **0.633** | 16 |
| MLP-6 aug | 3 | 256 | 0.3 | 1e-3 | 61.2% | 0.611 | 0.616 | 93 |

**★ Best: MLP-5** (UAR=0.633)

### Πειραματισμός — Run v2 (απορρίφθηκε)

Δοκιμάστηκαν 3 αλλαγές: **AdamW** (weight_decay=1e-4) + **class-weighted loss** + **patience=20**.
Αποτέλεσμα: χειρότερο σε όλα τα configs (best UAR 0.622 vs 0.633).

**Αιτία:** Το IEMOCAP 80/20 split είναι ήδη αρκετά balanced — τα class weights στρέβλωσαν την εκπαίδευση αντί να βοηθήσουν. Επαναφορά στο v1.

### Βασικά Συμπεράσματα MLP
- Το **πλάτος** (MLP-3, hidden=512) χωρίς αύξηση βάθους δεν βοηθάει
- Η **augmentation** (MLP-6) δεν βελτίωσε — 93 epochs για σύγκλιση, χωρίς κέρδος
- Το **βάθος + μικρό LR** (MLP-5) έδωσε το καλύτερο αποτέλεσμα

---

## Βήμα 4 — Architecture 2: Mel-spectrogram + CNN ✅ DONE

**Script:** `dl_04_train_cnn.py`

**Architecture:**
```
Input(1, 128, 300)
→ [Conv2d(3×3) → BatchNorm2d → ReLU → MaxPool(2×2)] × N
→ AdaptiveAvgPool2d(1,1) → Flatten
→ Linear(C→256) → ReLU → Dropout → Linear(256→4)
```

**Training:** batch_size=32, GPU (RTX 3060). Dataset class φορτώνει `.npy` on-the-fly.

### Αποτελέσματα

| Config | Blocks | Filters | Dropout | LR | Aug | Accuracy | W-F1 | UAR | Epochs |
|--------|--------|---------|---------|-----|-----|----------|------|-----|--------|
| CNN-1 | 2 | [32, 64] | 0.3 | 1e-3 | ✗ | 49.7% | 0.481 | 0.523 | 21 |
| CNN-2 | 3 | [32, 64, 128] | 0.3 | 1e-3 | ✗ | 55.6% | 0.554 | 0.558 | 31 |
| CNN-3 | 3 | [64, 128, 256] | 0.3 | 1e-3 | ✗ | 55.2% | 0.548 | 0.532 | 39 |
| CNN-4 | 3 | [32, 64, 128] | 0.5 | 1e-3 | ✗ | 54.7% | 0.535 | 0.560 | 25 |
| CNN-5 | 4 | [32,64,128,256] | 0.3 | 5e-4 | ✗ | 57.6% | 0.572 | 0.567 | 26 |
| **CNN-6** ★ | **3** | **[32, 64, 128]** | **0.3** | **1e-3** | **✓** | **60.5%** | **0.602** | **0.617** | **80** |

**★ Best: CNN-6** (UAR=0.617) — CNN-2 αρχιτεκτονική + augmented spectrograms

### Βασικά Συμπεράσματα CNN
- **Augmentation = το μεγαλύτερο κέρδος**: CNN-6 vs CNN-2 (ίδια arch): UAR 0.617 vs 0.558 (+5.9%)
- **Βάθος βοηθάει**: CNN-1→2→5: UAR 0.523→0.558→0.567
- **Πλάτος (CNN-3) δεν βοηθάει** χωρίς αρκετά δεδομένα (0.532 < 0.558)
- **MLP > CNN**: MLP-5 UAR=0.633 vs CNN-6 UAR=0.617 — τα hand-crafted features νικούν τα mel-spectrograms στο μέγεθος του IEMOCAP

---

## Βήμα 5 — Architecture 3: Mel-spectrogram + CNN-LSTM ✅ DONE

**Script:** `dl_05_train_cnn_lstm.py`

**Architecture:**
```
Input(1, 128, T)
→ [Conv2d → BN → ReLU → MaxPool] × N
→ mean over freq axis → (batch, T', C)
→ BiLSTM(input=C, hidden=H, layers=L)
→ Attention pooling over time steps
→ Linear(2H) → Dropout → Linear(4)
```

**Training:** batch_size=32, GPU (RTX 3060)

### Αποτελέσματα

| Config | CNN | LSTM H | LSTM L | BiLSTM | Aug | Accuracy | W-F1 | UAR | Epochs |
|--------|-----|--------|--------|--------|-----|----------|------|-----|--------|
| **CL-1** ★ | **2** | **128** | **1** | **✓** | **✗** | **55.9%** | **0.546** | **0.579** | **41** |
| CL-2 | 2 | 256 | 1 | ✓ | ✗ | 56.9% | 0.569 | 0.570 | 35 |
| CL-3 | 2 | 128 | 2 | ✓ | ✗ | 53.8% | 0.539 | 0.534 | 18 |
| CL-4 | 3 | 128 | 1 | ✓ | ✗ | 56.3% | 0.563 | 0.571 | 26 |
| CL-5 | 2 | 128 | 1 | ✗ | ✗ | 55.3% | 0.553 | 0.563 | 28 |
| CL-6 aug | 2 | 128 | 1 | ✓ | ✓ | 53.8% | 0.536 | 0.544 | 100 |

**★ Best: CL-1** (UAR=0.579) — το απλούστερο config!

### Βασικά Συμπεράσματα CNN-LSTM

- **Augmentation ΧΕΙΡΟΤΕΡΕΨΕ** (CL-6 UAR=0.544 vs CL-1 UAR=0.579, -3.5%) — αντίθεση με CNN. val_f1=0.93 του CL-6 αποκαλύπτει overfitting: augmented versions ίδιων utterances καταλήγουν και στο val, επομένως το μοντέλο "απομνημονεύει" και δεν γενικεύεται σε clean test data.
- **BiLSTM > UniLSTM** (CL-1 vs CL-5): UAR 0.579 vs 0.563 (+1.6%) — η αμφίδρομη ανάγνωση βοηθάει ελαφρά
- **2 LSTM layers (CL-3) χειρότερο**: UAR 0.534, μόνο 18 epochs — overfitting λόγω μεγαλύτερης πολυπλοκότητας
- **+CNN depth (CL-4)**: UAR 0.571 — οριακή διαφορά από baseline

### CNN-LSTM vs προηγούμενα μοντέλα

| Μοντέλο | Best UAR |
|---------|---------|
| MLP-5 | **0.633** |
| CNN-6 | 0.617 |
| CNN-LSTM CL-1 | 0.579 |

**Η χρονική μοντελοποίηση (LSTM) ΔΕΝ βοηθάει** — το CNN-LSTM είναι χειρότερο από το CNN. Η πολυπλοκότητα του LSTM απαιτεί περισσότερα δεδομένα. Το mel-spectrogram ήδη ενσωματώνει χρονική πληροφορία μέσω του AdaptiveAvgPool.

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
- Epoch 6+: unfreeze, layerwise LR decay
- Optimizer: AdamW + weight decay
- ⚠️ Απαιτεί GPU — πριν την εκτέλεση: `pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu126`

**Απαιτούμενη Ανάλυση (για report):**
1. Attention weight visualization
2. Layer probing (layers 1-12)
3. Frozen vs fine-tuned comparison

---

## Βήμα 7 — Noise Robustness Evaluation

**Script:** `dl_07_noise_robustness_eval.py`

**Input:** Best checkpoints από Βήματα 3-6 + clean test WAV files

**Noise conditions (on-the-fly, ΔΕΝ αποθηκεύονται):**
- Clean (baseline)
- Gaussian: SNR = 20dB, 10dB, 5dB, 0dB
- Reverberation: RT60 = 0.3s, 0.6s

**Metrics ανά condition:** UAR, Weighted F1

**Output:**
- `robustness_curves.png`
- `robustness_summary.csv`
- `per_model_degradation.png`
- Heatmap: μοντέλο × noise condition

---

## Βήμα 8 — Aggregate Results & Comparison

**Script:** `dl_08_aggregate_results.py`

**Master Comparison Table (partial):**

| Μοντέλο | Accuracy | W-F1 | UAR(clean) | UAR(10dB) | UAR(5dB) | #Params | Time |
|---------|----------|------|------------|-----------|----------|---------|------|
| MLP-5 | 61.9% | 0.616 | 0.633 | - | - | - | - |
| CNN-6 | 60.5% | 0.602 | 0.617 | - | - | - | - |
| CNN-LSTM CL-1 | 55.9% | 0.546 | 0.579 | - | - | - | - |
| wav2vec2 | - | - | - | - | - | - | - |

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
Step 0 (setup) ✅
    ↓
Step 1 (dl_01_extract_spectrograms.py) ✅
    ↓
Step 2 (dl_02_noise_augment.py) ✅
    ↓
Step 3 (dl_03_train_mlp.py) ✅  →  Best: MLP-5, UAR=0.633
    ↓
Step 4 (dl_04_train_cnn.py) ✅  →  Best: CNN-6, UAR=0.617
    ↓
Step 5 (dl_05_train_cnn_lstm.py) ✅  →  Best: CL-1, UAR=0.579
    ↓
Step 6 (dl_06_train_wav2vec2.py) ← GPU needed
    ↓
Step 7 (dl_07_noise_robustness_eval.py)
    ↓
Step 8 (dl_08_aggregate_results.py)
    ↓
Step 9 (Report + Slides)
```
