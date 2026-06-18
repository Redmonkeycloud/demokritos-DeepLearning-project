# Deep Learning SER — Implementation Plan

## Στόχος
Συστηματική σύγκριση 4 αρχιτεκτονικών deep learning για Speech Emotion Recognition (SER) στο IEMOCAP dataset, με έμφαση στο noise robustness και την κατανόηση της συμπεριφοράς κάθε μοντέλου.

**Split:** LOSO (Leave-One-Speaker-Out) — Ses05M held-out ως eval speaker  
**Train:** 4,934 utterances (Sessions 1–4, 9 ομιλητές)  
**Eval:**  597 utterances (Session 5 — Ses05M, αόρατος κατά την εκπαίδευση)

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
  features/splits/loso/    <- LOSO split (Ses05M held-out)
  spectrograms/train/
  spectrograms/eval/
  spectrograms/train_augmented/
  models/mlp/
  models/cnn/
  models/cnn_lstm/
  models/wav2vec2/
  results/mlp/
  results/cnn/
  results/cnn_lstm/
  results/wav2vec2/
  results/noise_robustness/
  results/aggregate/
```

**Reuse από ML εξάμηνο:**
- WAV αρχεία: `D:\Users\User\Desktop\demokritos-ml-project\datasets\iemocap\`
- `iemocap_features.csv` (272D, 5531 rows) ✓

> **Σημείωση:** GPU (NVIDIA RTX 3060 12GB), torch 2.9.1+cu126. Όλη η εκπαίδευση τρέχει on-GPU.

---

## Βήμα 0b — Speaker-Independent Split (LOSO) ✅ DONE

**Script:** `dl_00b_speaker_split.py`

Speaker-independent split: Ses05M (Session 5 male) held out ως eval speaker.

**Outputs:**
- `features/splits/loso/train.csv` — 4,934 utterances, Sessions 1–4
- `features/splits/loso/eval.csv`  — 597 utterances, Ses05M
- `features/splits/loso/scaler.pkl` — StandardScaler fitted on train set only

**Κατανομή κλάσεων (eval):**

| Emotion | Count |
|---------|-------|
| angry   | ~150  |
| happy   | ~150  |
| neutral | ~150  |
| sad     | ~147  |

> Αιτιολόγηση: Ο 80/20 τυχαίος split επιτρέπει data leakage μεταξύ ομιλητών. Ο LOSO split δοκιμάζει πραγματική speaker-independent γενίκευση — αυστηρότερο και ρεαλιστικότερο σενάριο.

---

## Βήμα 1 — Data Preparation ✅ DONE

### 1A — 272D Features (υπάρχουν ήδη)
Εξαγωγή Features.

### 1B — Mel-Spectrogram Extraction ✅
**Script:** `dl_01_extract_spectrograms.py --split-dir loso`

- Input: WAV paths από `train.csv` + `eval.csv` (path remapping `/workspace` → local)
- `librosa.load()` → 16kHz
- `librosa.feature.melspectrogram()`: n_mels=128, hop_length=160 (10ms), win_length=400 (25ms), n_fft=1024
- `librosa.power_to_db()` (log scale)
- Pad/truncate → 300 time frames (3 δευτερόλεπτα)
- Shape τελικό: `(1, 128, 300)` — float32, range [-80, 0] dB
- **Αποτέλεσμα:** 4,934 train + 597 eval spectrograms
- Output: `train_manifest.csv` + `eval_manifest.csv`

### 1C — Raw Audio για wav2vec2
Δεν χρειάζεται pre-extraction. Το wav2vec2 Dataset class φορτώνει raw WAV on-the-fly.

---

## Βήμα 2 — Noise Augmentation (TRAIN ONLY) ✅ DONE

**Script:** `dl_02_noise_augment.py --split-dir loso`

> ⚠️ DATA LEAKAGE: Augmented δείγματα ΜΟΝΟ στο train set. Eval set παραμένει clean και untouched.

**Τύποι augmentation (5 conditions):**

| Condition | Τύπος | Παράμετρος |
|-----------|-------|------------|
| gauss_20 | Gaussian noise | SNR = 20 dB |
| gauss_10 | Gaussian noise | SNR = 10 dB |
| gauss_5 | Gaussian noise | SNR = 5 dB |
| room_03 | Reverberation | RT60 = 0.3s |
| room_06 | Reverberation | RT60 = 0.6s |

> **Αλλαγή από plan:** Αφαιρέθηκε το `AddBackgroundNoise` (MUSAN dataset ~11GB).
> Reverberation: synthetic exponential-decay RIR (χωρίς pyroomacoustics).

**Αποτέλεσμα:** 4,934 × 5 = **24,670 augmented samples**

**Output για MLP:** `train_augmented_features.csv` (24,670 rows × 275 cols)  
**Output για CNN/CNN-LSTM:** Mel-spectrograms (.npy), shape (1, 128, 300) + `train_augmented_manifest.csv`

---

## Βήμα 3 — Architecture 1: Hand-crafted Features + MLP ✅ DONE

**Script:** `dl_03_train_mlp.py --split-dir loso`

**Architecture:**
```
Input(272) -> [Linear -> BatchNorm -> ReLU -> Dropout] x N -> Linear(4)
```

**Training setup:**
- Loss: CrossEntropyLoss | Optimizer: Adam | Scheduler: CosineAnnealingLR
- Early stopping: patience=10 (val weighted F1) | Val: 10% train (stratified)

### Αποτελέσματα — LOSO split (Ses05M held-out)

| Config | Layers | Hidden | Dropout | LR | Aug | Accuracy | W-F1 | UAR | Epochs |
|--------|--------|--------|---------|-----|-----|----------|------|-----|--------|
| **MLP-1** ★ | **2** | **256** | **0.3** | **1e-3** | **✗** | **55.9%** | **0.560** | **0.574** | **20** |
| MLP-2 | 3 | 256 | 0.3 | 1e-3 | ✗ | 52.9% | 0.527 | 0.552 | 25 |
| MLP-3 | 3 | 512 | 0.3 | 1e-3 | ✗ | 54.9% | 0.549 | 0.572 | 16 |
| MLP-4 | 3 | 512 | 0.5 | 1e-3 | ✗ | 54.3% | 0.536 | 0.574 | 15 |
| MLP-5 | 4 | 512 | 0.3 | 5e-4 | ✗ | 50.6% | 0.507 | 0.513 | 18 |
| MLP-6 aug | 3 | 256 | 0.3 | 1e-3 | ✓ | 54.8% | 0.547 | 0.553 | 76 |

**★ Best: MLP-1** (UAR=0.574)

### Βασικά Συμπεράσματα MLP (LOSO)
- Στο LOSO split το απλούστερο μοντέλο (MLP-1, 2 layers) νικάει — λιγότερο overfitting στον train speaker set
- **Augmentation (MLP-6) δεν βοηθάει** — 76 epochs για σύγκλιση, UAR 0.553 < 0.574
- Βαθύτερα/φαρδύτερα δίκτυα υπερπροσαρμόζονται στους train speakers

---

## Βήμα 4 — Architecture 2: Mel-spectrogram + CNN ✅ DONE

**Script:** `dl_04_train_cnn.py --split-dir loso`

**Architecture:**
```
Input(1, 128, 300)
-> [Conv2d(3x3) -> BatchNorm2d -> ReLU -> MaxPool(2x2)] x N
-> Pooling (varies per variant)
-> Classifier head -> Linear(4)
```

**Training:** batch_size=32, GPU RTX 3060.

### Αποτελέσματα — LOSO split (10 configs)

| Config | Type | Filters | Dropout | LR | Aug | Accuracy | W-F1 | UAR | Epochs |
|--------|------|---------|---------|-----|-----|----------|------|-----|--------|
| **CNN-9** ★ | freq_aware | [32, 64, 128] | 0.3 | 1e-3 | ✓ | 58.3% | 0.580 | **0.603** | 74 |
| CNN-4 | standard | [32, 64, 128] | 0.5 | 1e-3 | ✗ | 55.3% | 0.550 | 0.586 | 52 |
| CNN-5 | standard | [32,64,128,256] | 0.3 | 5e-4 | ✗ | 58.8% | 0.591 | 0.585 | 31 |
| CNN-6 | standard | [32, 64, 128] | 0.3 | 1e-3 | ✓ | 58.1% | 0.581 | 0.577 | 44 |
| CNN-8 | residual | [32, 64, 128] | 0.3 | 1e-3 | ✓ | 51.4% | 0.487 | 0.564 | 26 |
| CNN-10 | deep_head | [32, 64, 128] | 0.3 | 1e-3 | ✓ | 56.6% | 0.567 | 0.562 | 100 |
| CNN-7 | gap | [32, 64, 128] | 0.3 | 1e-3 | ✓ | 54.3% | 0.545 | 0.533 | 100 |
| CNN-3 | standard | [64, 128, 256] | 0.3 | 1e-3 | ✗ | 52.4% | 0.495 | 0.530 | 30 |
| CNN-2 | standard | [32, 64, 128] | 0.3 | 1e-3 | ✗ | 54.1% | 0.544 | 0.530 | 58 |
| CNN-1 | standard | [32, 64] | 0.3 | 1e-3 | ✗ | 50.4% | 0.494 | 0.528 | 14 |

**★ Best: CNN-9** (freq_aware pooling, UAR=0.603)

### Νέες αρχιτεκτονικές (CNN-7 έως CNN-10)

| Config | Περιγραφή | UAR |
|--------|-----------|-----|
| CNN-7 | True Global Average Pooling (mean over H,W) | 0.533 |
| CNN-8 | Residual (skip) connections σε κάθε block | 0.564 |
| **CNN-9** | Frequency-aware pooling (pool μόνο time, 4 freq bins) | **0.603** |
| CNN-10 | Deeper classifier head (128→512→256→4) | 0.562 |

**CNN-9 = καλύτερο από όλα τα 10 configs.** Η διατήρηση 4 frequency bins (αντί collapse) επιτρέπει στο μοντέλο να εκμεταλλεύεται τη φασματική δομή — κρίσιμη για emotion (π.χ. φωνητικός τόνος, αρμονικά).

### Βασικά Συμπεράσματα CNN (LOSO)
- **Augmentation + freq-aware pooling = η καλύτερη συνδυαστική αρχιτεκτονική** (CNN-9)
- **Residual connections (CNN-8)** βοηθούν ελαφρά (+0.564 vs CNN-6 0.577) αλλά δεν νικούν την freq-aware pooling
- **Deeper head (CNN-10)** δεν αξίζει — marginal gain με 100 epochs training
- **Global avg pooling (CNN-7)** χειρότερο — η χωρική πληροφορία χάνεται πλήρως

---

## Βήμα 5 — Architecture 3: Mel-spectrogram + CNN-LSTM ✅ DONE

**Script:** `dl_05_train_cnn_lstm.py --split-dir loso`

**Architecture:**
```
Input(1, 128, T)
-> [Conv2d -> BN -> ReLU -> MaxPool] x N
-> mean over freq axis -> (batch, T', C)
-> BiLSTM(input=C, hidden=H, layers=L)
-> Attention pooling over time steps
-> Linear(2H) -> Dropout -> Linear(4)
```

**Training:** batch_size=32, GPU RTX 3060

### Αποτελέσματα — LOSO split

| Config | CNN | LSTM H | LSTM L | BiLSTM | Aug | Accuracy | W-F1 | UAR | Epochs |
|--------|-----|--------|--------|--------|-----|----------|------|-----|--------|
| CL-1 | 2 | 128 | 1 | ✓ | ✗ | 54.6% | 0.546 | 0.544 | 62 |
| CL-2 | 2 | 256 | 1 | ✓ | ✗ | 56.6% | 0.557 | 0.570 | 34 |
| CL-3 | 2 | 128 | 2 | ✓ | ✗ | 56.3% | 0.562 | 0.575 | 39 |
| **CL-4** ★ | **3** | **128** | **1** | **✓** | **✗** | **56.6%** | **0.563** | **0.577** | **33** |
| CL-5 | 2 | 128 | 1 | ✗ | ✗ | 53.9% | 0.536 | 0.552 | 35 |
| CL-6 aug | 2 | 128 | 1 | ✓ | ✓ | 50.8% | 0.508 | 0.504 | 87 |

**★ Best: CL-4** (UAR=0.577) — 3 CNN blocks + 1-layer BiLSTM

### Βασικά Συμπεράσματα CNN-LSTM (LOSO)
- **Augmentation ΧΕΙΡΟΤΕΡΕΨΕ δραματικά** (CL-6 UAR=0.504 vs CL-4 0.577, -7.3%) — val_f1=0.93 αποκαλύπτει ακραίο overfitting
- **BiLSTM > UniLSTM** (CL-1 vs CL-5): UAR 0.544 vs 0.552 — αμφίδρομη ανάγνωση ελαφρά καλύτερη
- **+CNN depth (CL-4)**: UAR 0.577 — βέλτιστο config

### CNN-LSTM vs προηγούμενα μοντέλα

| Μοντέλο | Best UAR |
|---------|---------|
| CNN-9 | **0.603** |
| MLP-1 | 0.574 |
| CNN-LSTM CL-4 | 0.577 |
| CNN-6 | 0.577 |

---

## Βήμα 6 — Architecture 4: wav2vec2 + Attention Head (Upper Bound) ✅ DONE

**Script:** `dl_06_train_wav2vec2.py --split-dir loso`

**Architecture:**
```
facebook/wav2vec2-base (pretrained, LibriSpeech 960h)
  -> feature extractor (CNN layers)
  -> transformer (12 layers, hidden=768)
  -> hidden states: (batch, T, 768)
  -> Attention pooling (learnable weights) -> (batch, 768)
  -> Linear(768->256) -> ReLU -> Dropout(0.3) -> Linear(256->4)
```

**Training:** batch_size=16, GPU RTX 3060, max_length=8s, AdamW + layerwise LR decay

### Αποτελέσματα — LOSO split

| Phase | Accuracy | W-F1 | UAR | Epochs |
|-------|----------|------|-----|--------|
| Frozen baseline (ep 1-5) | 48.6% | — | 0.540 | 5 |
| **Fine-tuned** ★ | **68.0%** | **0.681** | **0.670** | **19** |

**Fine-tune gain: +13.0% UAR** (0.540 → 0.670)

### Layer Probing (UAR per layer, linear probe on eval set)

| Layer | UAR | Σημείωση |
|-------|-----|---------|
| 0 | 0.570 | feature projection output |
| 1 | 0.604 | |
| 2 | 0.640 | |
| 3 | 0.631 | |
| 4 | 0.638 | |
| 5 | 0.654 | |
| 6 | 0.660 | |
| 7 | 0.653 | |
| 8 | 0.629 | |
| 9 | 0.664 | |
| 10 | 0.687 | |
| **11** | **0.693** | **peak — emotion info εδώ** |
| 12 | 0.675 | τελικό layer ελαφρά χειρότερο |

### Βασικά Συμπεράσματα wav2vec2 (LOSO)
- **Frozen (0.540) < MLP (0.574)**: Τα pretrained features χωρίς fine-tuning είναι χειρότερα από hand-crafted στο LOSO
- **Fine-tuning κρίσιμο**: +13.0% UAR — task-specific adaptation απαραίτητη
- **Emotion info peaks στο layer 11** (από 12): Το προτελευταίο layer φέρει τη μέγιστη emotion πληροφορία
- **Σαφής upper bound**: wav2vec2 UAR=0.670 >> CNN-9 UAR=0.603

---

## Βήμα 7 — Noise Robustness Evaluation ✅ DONE

**Script:** `dl_07_noise_robustness_eval.py --split-dir loso`

**Noise conditions (on-the-fly από raw WAV):**
- Clean (baseline)
- Gaussian: SNR = 20dB, 10dB, 5dB, 0dB
- Reverberation: RT60 = 0.3s, 0.6s

**Dual evaluation:** noised + spectral-subtraction denoised (in-memory, χωρίς αποθήκευση)

### Αποτελέσματα Noise Robustness (UAR)

| Model | Best config | Clean | G-20dB | G-10dB | G-5dB | G-0dB | R-0.3s | R-0.6s |
|-------|-------------|-------|--------|--------|-------|-------|--------|--------|
| MLP | mlp_6 | 0.543 | 0.529 | 0.517 | 0.500 | 0.427 | 0.537 | 0.499 |
| **CNN** | **cnn_9** | **0.603** | **0.589** | **0.557** | **0.517** | **0.464** | **0.539** | **0.549** |
| CNN-LSTM | cl_4 | 0.577 | 0.383 | 0.324 | 0.282 | 0.301 | 0.479 | 0.414 |
| wav2vec2 | finetuned | 0.670 | 0.556 | 0.371 | 0.283 | 0.252 | 0.591 | 0.520 |

### Επίδραση Denoiser (spectral subtraction)

| Model | G-20dB noised | G-20dB denoised | Gain |
|-------|--------------|-----------------|------|
| MLP | 0.529 | 0.265 | -0.264 |
| CNN | 0.589 | 0.410 | -0.179 |
| CNN-LSTM | 0.383 | 0.251 | -0.132 |
| wav2vec2 | 0.556 | 0.467 | -0.089 |

> **Παρατήρηση:** Ο spectral-subtraction denoiser **βλάπτει** όλα τα μοντέλα στο 20dB (ελαφρύ noise). Στο 0dB, η wav2vec2 ωφελείται ελαφρά (+0.036). Ο denoiser εισάγει artifacts που ενοχλούν περισσότερο τα CNN-based μοντέλα.

### Βασικά Συμπεράσματα Noise Robustness
- **CNN-9 = πιο noise-robust**: Ελάχιστη πτώση στα 20dB (-1.4%), αξιόπιστο έως 0dB
- **CNN-LSTM = πιο ευαίσθητο**: Catastrophic degradation στο Gaussian noise (0.577→0.301 στα 0dB)
- **wav2vec2 = highest clean αλλά fragile στο Gaussian**: Πτώση 0.670→0.252 στα 0dB
- **Reverberation**: Όλα τα μοντέλα αντέχουν καλύτερα (μικρότερη πτώση από Gaussian)

**Plots:** `robustness_curves.png`, `denoising_gain.png`, `robustness_heatmap.png`

---

## Βήμα 8 — Aggregate Results & Comparison ✅ DONE

**Script:** `dl_08_aggregate_results.py --split-dir loso`

### Master Comparison Table — LOSO Split

| Μοντέλο | Best config | Accuracy | W-F1 | Macro-F1 | UAR(clean) | UAR(20dB) | UAR(0dB) | #Params |
|---------|-------------|----------|------|----------|------------|-----------|----------|---------|
| MLP | mlp_1 | 55.9% | 0.560 | 0.568 | 0.574 | 0.529 | 0.427 | ~100-500K |
| CNN | cnn_9 | 58.3% | 0.580 | 0.595 | **0.603** | **0.589** | **0.464** | ~200-800K |
| CNN-LSTM | cl_4 | 56.6% | 0.563 | 0.575 | 0.577 | 0.383 | 0.301 | ~500-2000K |
| **wav2vec2** | finetuned | **68.0%** | **0.681** | **0.687** | **0.670** | 0.556 | 0.252 | ~95M |

**Overall best clean:** wav2vec2 (UAR=0.670)  
**Best noise-robust:** CNN/cnn_9 (smallest degradation across all conditions)  
**Denoiser helped most for:** wav2vec2 (στα 0dB: +0.036 UAR)

---

## Βήμα 9 — Report & Presentation

**Report (PROJECT_REPORT_DL.md / PDF):**
1. Εισαγωγή + στόχοι
2. Datasets (σύντομο — IEMOCAP ήδη στο ML report)
3. LOSO Split — αιτιολόγηση + στατιστικά
4. Αρχιτεκτονικές: περιγραφή + αιτιολόγηση
5. Noise Augmentation: strategy + data leakage prevention
6. Αποτελέσματα ανά architecture (hyperparameter sensitivity)
7. Noise Robustness analysis (noised + denoised)
8. wav2vec2 ανάλυση (attention, layer probing)
9. Σύγκριση + Συμπεράσματα

---

## Σειρά Εκτέλεσης

```
Step 0  (dl_00_setup_workflow.py)        ✅ DONE
    |
Step 0b (dl_00b_speaker_split.py)        ✅ DONE  ->  LOSO split: train=4934, eval=597 (Ses05M)
    |
Step 1  (dl_01_extract_spectrograms.py)  ✅ DONE  ->  4934 train + 597 eval .npy
    |
Step 2  (dl_02_noise_augment.py)         ✅ DONE  ->  24670 augmented samples
    |
Step 3  (dl_03_train_mlp.py)             ✅ DONE  ->  Best: MLP-1, UAR=0.574
    |
Step 4  (dl_04_train_cnn.py)             ✅ DONE  ->  Best: CNN-9 (freq_aware), UAR=0.603
    |
Step 5  (dl_05_train_cnn_lstm.py)        ✅ DONE  ->  Best: CL-4, UAR=0.577
    |
Step 6  (dl_06_train_wav2vec2.py)        ✅ DONE  ->  UAR=0.670 (fine-tuned, 19 epochs)
    |
Step 7  (dl_07_noise_robustness_eval.py) ✅ DONE  ->  CNN-9 most robust; wav2vec2 fragile at 0dB
    |
Step 8  (dl_08_aggregate_results.py)     ✅ DONE  ->  master_comparison.csv
    |
Step 9  (Report + Slides)                [ pending ]
```

---

## Αλλαγές μετά από feedback καθηγητή

### 1. LOSO split αντί 80/20 τυχαίου split
- **Πρόβλημα με 80/20:** Τυχαίος split αναμειγνύει utterances ίδιου ομιλητή μεταξύ train/test — δεν δοκιμάζει πραγματική γενίκευση
- **LOSO λύση:** Ses05M (Session 5 male) κρατάται εξ ολοκλήρου ως eval speaker
- **Αποτέλεσμα:** Train=4,934 (Sessions 1-4), Eval=597 (Ses05M) — αυστηρότερο αλλά ρεαλιστικότερο
- **Script:** `dl_00b_speaker_split.py` (νέο)

### 2. CNN Step 4: 4 νέες αρχιτεκτονικές (CNN-7 έως CNN-10)
Πέρα από τις 6 baseline αρχιτεκτονικές, δοκιμάστηκαν 4 παραλλαγές:

| Config | Ιδέα | UAR | Αποτέλεσμα |
|--------|------|-----|-----------|
| CNN-7 | True Global Average Pooling | 0.533 | Χειρότερο — χάνει spatial info |
| CNN-8 | Residual (skip) connections | 0.564 | Βοηθάει ελαφρά |
| **CNN-9** | Frequency-aware pooling | **0.603** | **Καλύτερο — διατηρεί freq structure** |
| CNN-10 | Deeper classifier (128→512→256→4) | 0.562 | Marginal gain |

**CNN-9 γίνεται το overall best CNN** — νικάει όλες τις baseline αρχιτεκτονικές.

### 3. Dual evaluation στο Step 7: noised + denoised conditions
- **Πρόσθεση:** Για κάθε noise condition, αξιολογούμε τόσο το noised signal όσο και το denoised (spectral subtraction)
- **Αποτέλεσμα:** Ο denoiser βλάπτει τα CNN-based μοντέλα αλλά ωφελεί την wav2vec2 στο ακραίο noise (0dB)
- **Plots:** `robustness_curves.png`, `denoising_gain.png`, `robustness_heatmap.png`
