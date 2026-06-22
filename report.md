# Deep Learning για Αναγνώριση Συναισθήματος σε Ομιλία

## 1. Εισαγωγή

Η εργασία εξετάζει το πρόβλημα Speech Emotion Recognition, δηλαδή την ταξινόμηση του συναισθήματος ενός εκφωνήματος με βάση το ακουστικό σήμα. Το τελικό σύστημα δουλεύει πάνω στο IEMOCAP και περιορίζεται σε τέσσερις κλάσεις, `angry`, `happy`, `neutral` και `sad`, με ενοποίηση των αρχικών ετικετών `happy` και `excited` στη θετική κλάση `happy`. Το ενδιαφέρον της εργασίας δεν είναι μόνο να βρεθεί ένα μοντέλο με υψηλή επίδοση σε καθαρό test set, αλλά να φανεί πώς αλλάζει η συμπεριφορά όταν αλλάζει το πρωτόκολλο αξιολόγησης, η αναπαράσταση εισόδου και η ποιότητα του ήχου.

Η υλοποίηση συγκρίνει τέσσερις οικογένειες deep learning μοντέλων. Το MLP χρησιμοποιεί χειροποίητα ακουστικά χαρακτηριστικά 272 διαστάσεων. Το CNN και το CNN-LSTM χρησιμοποιούν log-mel spectrograms ως εικόνες ενός καναλιού. Το wav2vec2 χρησιμοποιεί raw waveform και αξιοποιεί pretrained self-supervised αναπαραστάσεις. Τα αποτελέσματα χωρίζονται σε δύο βασικά πρωτόκολλα. Το `80/20` είναι ένα stratified random split και λειτουργεί ως πιο αισιόδοξο benchmark. Το `LOSO`, με held-out speaker τον `Ses05M`, είναι αυστηρότερο επειδή ελέγχει speaker-independent γενίκευση.

Τα αριθμητικά αποτελέσματα του report προέρχονται από τα αρχεία `features/*.csv`, `results/**/*.csv`, `results/**/*.json`, `results/aggregate/master_comparison.txt` και από το αρχείο `80_20.html`. Οι εικόνες προέρχονται από όλα τα `.png` που υπάρχουν στο `workflows/iemocap_dl/results`.

## 2. Dataset και splits

Το τελικό IEMOCAP subset περιέχει 5.531 utterances. Η κατανομή είναι σχετικά ισορροπημένη για SER εργασία: 1.708 neutral δείγματα, 1.636 happy δείγματα, 1.103 angry δείγματα και 1.084 sad δείγματα. Η κλάση neutral είναι η συχνότερη με 30,9%, ενώ η sad είναι η μικρότερη με 19,6%. Η διαφορά δεν είναι αμελητέα, αλλά δεν είναι τόσο μεγάλη ώστε να δικαιολογεί βαρύ resampling. Για αυτό η βασική στρατηγική ήταν stratified splitting και αξιολόγηση με UAR, ώστε κάθε κλάση να έχει ίσο βάρος στο τελικό recall.

Στο `80/20` split υπάρχουν 4.424 train utterances και 1.107 test utterances. Η κατανομή παραμένει σχεδόν ίδια με το πλήρες dataset, με 1.366 neutral, 1.309 happy, 882 angry και 867 sad στο train, και 342 neutral, 327 happy, 221 angry και 217 sad στο test. Αυτό το split είναι χρήσιμο για γρήγορη σύγκριση αρχιτεκτονικών, αλλά μπορεί να βάλει utterances του ίδιου ομιλητή και στις δύο πλευρές του split.

Στο `LOSO` split κρατήθηκε εκτός εκπαίδευσης ο speaker `Ses05M`. Το train set έχει 4.934 utterances από τους υπόλοιπους ομιλητές, ενώ το eval set έχει 597 utterances αποκλειστικά από τον `Ses05M`. Η κατανομή του LOSO eval set είναι 206 happy, 192 neutral, 107 sad και 92 angry. Επειδή το μοντέλο αξιολογείται σε φωνή που δεν έχει δει, το LOSO είναι το πιο κοντινό πρωτόκολλο σε πραγματική χρήση.

| Πρωτόκολλο | Train | Test/Eval | Περιγραφή |
|:---|---:|---:|:---|
| `80/20` | 4.424 | 1.107 | Stratified random split |
| `LOSO` | 4.934 | 597 | Held-out speaker `Ses05M` |

## 3. Handcrafted χαρακτηριστικά

Τα handcrafted χαρακτηριστικά χρησιμοποιούνται κυρίως από το MLP και είναι αποθηκευμένα στο `workflows/iemocap_dl/features/iemocap_features.csv`. Το αρχείο έχει 5.531 γραμμές και 275 στήλες. Από αυτές, οι 272 είναι ακουστικές διαστάσεις, ενώ οι υπόλοιπες είναι metadata όπως `label`, `file_path` και `dataset`. Τα feature names δείχνουν ότι η εξαγωγή περιλαμβάνει descriptors όπως zero crossing rate, energy, energy entropy, spectral centroid, spectral spread, spectral entropy, spectral flux, spectral rolloff και MFCCs. Για κάθε frame-level descriptor αποθηκεύονται στατιστικά ανά utterance, όπως mean, min, max και standard deviation, με αποτέλεσμα ένα σταθερό tabular διάνυσμα ανά ηχητικό αρχείο.

Το πλεονέκτημα αυτής της αναπαράστασης είναι ότι συμπυκνώνει την ακουστική πληροφορία σε μικρό αριθμό διαστάσεων. Έτσι το MLP εκπαιδεύεται γρήγορα και χρειάζεται πολύ λιγότερα δεδομένα από ένα μεγάλο μοντέλο raw waveform. Το μειονέκτημα είναι ότι η χρονική δομή του συναισθήματος χάνεται σε μεγάλο βαθμό, επειδή τα στατιστικά συνοψίζουν όλο το utterance. Αυτό φαίνεται και στα αποτελέσματα: το MLP είναι πολύ αξιοπρεπές baseline, αλλά δεν φτάνει την καθαρή επίδοση του fine-tuned wav2vec2.

Ο `StandardScaler` εφαρμόζεται σωστά: γίνεται fit μόνο στο train set και μετά χρησιμοποιείται για transform στο test ή eval set. Αυτό είναι σημαντικό επειδή διαφορετικά η κανονικοποίηση θα μπορούσε να διαρρεύσει πληροφορία από την αξιολόγηση προς την εκπαίδευση. Στο LOSO ειδικά, η μη διαρροή είναι κρίσιμη, γιατί το ζητούμενο είναι να μη δει καθόλου το στατιστικό προφίλ του held-out speaker.

## 4. Spectrograms

Τα CNN-based μοντέλα δεν βλέπουν handcrafted vectors αλλά log-mel spectrograms. Κάθε αρχείο WAV φορτώνεται στα 16 kHz, μετατρέπεται σε mono και περνάει από mel transform με 128 mel bins, hop length 160 samples, window length 400 samples και `n_fft=1024`. Το hop των 160 samples αντιστοιχεί σε περίπου 10 ms, ενώ το παράθυρο των 400 samples σε 25 ms. Το spectrogram μετατρέπεται σε dB κλίμακα και μετά κόβεται ή συμπληρώνεται με padding ώστε να έχει ακριβώς 300 χρονικά frames. Η τελική μορφή είναι `(1, 128, 300)`, δηλαδή μία εικόνα ενός καναλιού με άξονα συχνότητας 128 bins και άξονα χρόνου 300 frames.

Στο workspace υπάρχουν 5.410 αρχεία `.npy` στο `spectrograms/train`, 1.107 στο `spectrograms/test`, 597 στο `spectrograms/eval` και 27.050 στο `spectrograms/train_augmented`. Το `test` αντιστοιχεί στο legacy `80/20` split, ενώ το `eval` στο LOSO split. Τα manifests `train_manifest.csv`, `test_manifest.csv` και `eval_manifest.csv` συνδέουν κάθε δείγμα με το αντίστοιχο `.npy`, την ετικέτα και το αρχικό WAV.

Η επιλογή fixed length τριών δευτερολέπτων κάνει τα batches απλά και σταθερά, αλλά έχει και κόστος. Αν ένα utterance είναι μεγαλύτερο, το tail κόβεται. Αν είναι μικρότερο, γεμίζει με padding. Αυτό είναι αποδεκτό για CNN/CNN-LSTM πειράματα, επειδή τα περισσότερα συναισθηματικά cues βρίσκονται σε τοπικά φασματικά και ρυθμικά patterns, αλλά παραμένει ένας λόγος που το wav2vec2 με raw waveform έως 8 δευτερόλεπτα μπορεί να κρατήσει περισσότερη πληροφορία.

## 5. Noise augmentation και denoising

Το augmentation εφαρμόζεται μόνο στο train set. Για το `80/20` δημιουργούνται 22.120 augmented δείγματα, δηλαδή 4.424 train utterances επί 5 συνθήκες. Για το `LOSO` δημιουργούνται 24.670 augmented δείγματα, δηλαδή 4.934 train utterances επί 5 συνθήκες. Το eval ή test set παραμένει καθαρό κατά την κανονική εκπαίδευση, ώστε η επίδοση να είναι συγκρίσιμη.

Η noise στρατηγική περιλαμβάνει Gaussian θόρυβο με SNR 20 dB, 10 dB και 5 dB, καθώς και reverberation με RT60 0,3s και 0,6s. Στο robustness evaluation προστίθεται και η ακραία συνθήκη Gaussian 0 dB. Όσο μικρότερο είναι το SNR, τόσο δυσκολότερη γίνεται η αναγνώριση, επειδή η ισχύς του θορύβου πλησιάζει ή ισούται με την ισχύ του σήματος. Στο 20 dB ο θόρυβος είναι ήπιος, στο 5 dB είναι έντονος και στο 0 dB είναι εξαιρετικά δύσκολος.

Στο `dl_07_noise_robustness_eval.py` κάθε καλύτερο μοντέλο αξιολογείται πάνω στο καθαρό LOSO eval set και σε έξι noised conditions. Επιπλέον, κάθε noised sample περνάει και από spectral subtraction denoiser, οπότε προκύπτει σύγκριση `clean`, `noised` και `denoised`. Η ιδέα είναι πρακτική: να ελεγχθεί αν ένας κλασικός denoiser βοηθάει τα deep models ή αν εισάγει artifacts που τα μπερδεύουν.

Τα αποτελέσματα δείχνουν ότι ο denoiser συνήθως βλάπτει. Σχεδόν όλες οι denoised UAR τιμές είναι χαμηλότερες από τις noised. Αυτό είναι ιδιαίτερα εμφανές σε MLP, CNN και CNN-LSTM, όπου τα denoising artifacts αλλοιώνουν είτε τα handcrafted descriptors είτε το φάσμα. Το wav2vec2 είναι λίγο πιο ανθεκτικό σε ορισμένες ακραίες Gaussian συνθήκες, όπου ο denoiser μπορεί να δώσει μικρό κέρδος, αλλά συνολικά και εκεί η εικόνα δεν δικαιολογεί blind denoising ως preprocessing.

## 6. Αρχιτεκτονικές

Το MLP δέχεται είσοδο 272 διαστάσεων. Κάθε hidden layer αποτελείται από `Linear`, `BatchNorm1d`, `ReLU` και `Dropout`, ενώ η έξοδος είναι ένα τελικό `Linear` προς τις 4 κλάσεις. Εκπαιδεύτηκαν έξι configs, με 2 έως 4 layers, hidden size 256 ή 512, dropout 0,3 ή 0,5, learning rate 1e-3 ή 5e-4 και μία augmented παραλλαγή. Η εκπαίδευση χρησιμοποιεί Adam, CrossEntropyLoss, CosineAnnealingLR, validation fraction 10% και early stopping με patience 10.

Το CNN παίρνει spectrograms `(1,128,300)`. Η βασική δομή είναι επαναλαμβανόμενα blocks `Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d`, ακολουθούμενα από pooling, flattening και MLP head. Τα αρχικά `cnn_1` έως `cnn_6` καλύπτουν βάθος, πλάτος, dropout, learning rate και augmentation. Στη LOSO εκδοχή προστέθηκαν επιπλέον αρχιτεκτονικές: `cnn_7` με true global average pooling, `cnn_8` με residual blocks, `cnn_9` με frequency-aware pooling και `cnn_10` με βαθύτερο classifier head. Το `cnn_9` είναι ιδιαίτερα σημαντικό, επειδή δεν συμπιέζει πλήρως τον άξονα συχνότητας. Κρατάει 4 frequency bins μετά το pooling και έτσι δίνει στο classifier περισσότερη φασματική δομή.

Το CNN-LSTM ξεκινά επίσης από spectrogram. Το CNN κομμάτι εξάγει local φασματικά patterns και μετά γίνεται μέσος όρος στον άξονα συχνότητας ώστε να μείνει χρονική ακολουθία χαρακτηριστικών `(batch, T', C)`. Αυτή η ακολουθία περνάει από LSTM, συνήθως bidirectional, και μετά από attention pooling. Το classifier παίρνει το attention context και παράγει logits για τις τέσσερις κλάσεις. Η αρχιτεκτονική είναι θεωρητικά πιο ταιριαστή σε ομιλία επειδή κρατάει χρονική εξέλιξη, αλλά στην πράξη το dataset είναι αρκετά μικρό και το LSTM τείνει να υπερπροσαρμόζεται.

Το wav2vec2 χρησιμοποιεί `facebook/wav2vec2-base`, pretrained σε LibriSpeech 960h. Η είσοδος είναι raw waveform στα 16 kHz, padded ή truncated σε 8 δευτερόλεπτα. Το backbone παράγει hidden states διάστασης 768 ανά frame. Πάνω σε αυτά μπαίνει learnable attention pooling, μετά `Linear(768 -> 256)`, `ReLU`, `Dropout` και `Linear(256 -> 4)`. Η εκπαίδευση έχει δύο φάσεις. Στα πρώτα 5 epochs το wav2vec2 είναι frozen και εκπαιδεύεται μόνο το classification head. Μετά ξεπαγώνουν τα τελευταία 4 transformer layers και το feature projection, με AdamW, weight decay 1e-4 και layerwise learning-rate decay. Αυτή η στρατηγική είναι πιο ασφαλής από πλήρες fine-tuning από την αρχή, επειδή το IEMOCAP είναι μικρό για transformer.

## 7. Σειρά εκτέλεσης των workflows

Η πλήρης deep learning ροή ξεκινά από προετοιμασία φακέλων και splits, συνεχίζει με εξαγωγή spectrograms και augmentation, μετά εκπαιδεύει τις τέσσερις οικογένειες μοντέλων και τελειώνει με robustness evaluation και aggregation. Η σειρά εκτέλεσης είναι η εξής:

```text
python scripts/dl_00_setup_workflow.py
python scripts/dl_00b_speaker_split.py
python scripts/dl_01_extract_spectrograms.py --split-dir loso
python scripts/dl_01_extract_spectrograms.py --split-dir 80_20
python scripts/dl_02_noise_augment.py --split-dir loso
python scripts/dl_02_noise_augment.py --split-dir 80_20
python scripts/dl_03_train_mlp.py
python scripts/dl_04_train_cnn.py
python scripts/dl_05_train_cnn_lstm.py
python scripts/dl_06_train_wav2vec2.py
python scripts/dl_07_noise_robustness_eval.py
```

Τα `dl_03`, `dl_04`, `dl_05` και `dl_06` μπορούν να θεωρηθούν ανεξάρτητα training branches αφού έχουν δημιουργηθεί τα features, spectrograms και augmented manifests. Το `dl_07` πρέπει να τρέξει μετά τα training scripts, επειδή φορτώνει τα καλύτερα checkpoints από κάθε οικογένεια. Το `dl_08` πρέπει να τρέξει τελευταίο, επειδή διαβάζει τα summary CSVs και το `robustness_summary.csv` για να δημιουργήσει το `master_comparison.csv` και το `master_comparison.txt`.

## 8. Αποτελέσματα 80/20

Το `80_20.html` κρατάει τα αποτελέσματα του αρχικού random split. Σε αυτό το πρωτόκολλο το fine-tuned wav2vec2 είναι καθαρά πρώτο με UAR 0,727 και accuracy 72,0%. Το MLP είναι πολύ δυνατό σε σχέση με το μέγεθός του, με καλύτερο config το MLP-5 και UAR 0,633. Το καλύτερο CNN είναι το CNN-6 με UAR 0,617, όπου το augmentation δίνει σημαντική βελτίωση σε σχέση με το αντίστοιχο non-augmented CNN-2. Το CNN-LSTM δεν ξεπερνά το απλό CNN και κορυφώνεται στο CL-1 με UAR 0,579.

| Μοντέλο | Καλύτερο config 80/20 | Accuracy | Weighted F1 | UAR |
|:---|:---|---:|---:|---:|
| MLP | MLP-5 | 61,9% | 0,616 | 0,633 |
| CNN | CNN-6 | 60,5% | 0,602 | 0,617 |
| CNN-LSTM | CL-1 | 55,9% | 0,546 | 0,579 |
| wav2vec2 | Fine-tuned | 72,0% | 0,720 | 0,727 |

| Config | Layers | Hidden | Dropout | LR | Accuracy | Weighted F1 | UAR | Epochs |
|:---|---:|---:|---:|:---|---:|---:|---:|---:|
| MLP-1 | 2 | 256 | 0,3 | 1e-3 | 62,8% | 0,627 | 0,626 | 18 |
| MLP-2 | 3 | 256 | 0,3 | 1e-3 | 61,2% | 0,609 | 0,630 | 19 |
| MLP-3 | 3 | 512 | 0,3 | 1e-3 | 59,9% | 0,599 | 0,602 | 34 |
| MLP-4 | 3 | 512 | 0,5 | 1e-3 | 62,5% | 0,623 | 0,629 | 20 |
| MLP-5 | 4 | 512 | 0,3 | 5e-4 | 61,9% | 0,616 | 0,633 | 16 |
| MLP-6 aug | 3 | 256 | 0,3 | 1e-3 | 61,2% | 0,611 | 0,616 | 93 |

| Config | Blocks | Filters | Aug | Accuracy | Weighted F1 | UAR | Epochs |
|:---|---:|:---|:---:|---:|---:|---:|---:|
| CNN-1 | 2 | [32,64] | no | 49,7% | 0,481 | 0,523 | 21 |
| CNN-2 | 3 | [32,64,128] | no | 55,6% | 0,554 | 0,558 | 31 |
| CNN-3 | 3 | [64,128,256] | no | 55,2% | 0,548 | 0,532 | 39 |
| CNN-4 | 3 | [32,64,128] | no | 54,7% | 0,535 | 0,560 | 25 |
| CNN-5 | 4 | [32,64,128,256] | no | 57,6% | 0,572 | 0,567 | 26 |
| CNN-6 | 3 | [32,64,128] | yes | 60,5% | 0,602 | 0,617 | 80 |

| Config | CNN blocks | LSTM hidden | LSTM layers | BiLSTM | Aug | Accuracy | Weighted F1 | UAR | Epochs |
|:---|---:|---:|---:|:---:|:---:|---:|---:|---:|---:|
| CL-1 | 2 | 128 | 1 | yes | no | 55,9% | 0,546 | 0,579 | 41 |
| CL-2 | 2 | 256 | 1 | yes | no | 56,9% | 0,569 | 0,570 | 35 |
| CL-3 | 2 | 128 | 2 | yes | no | 53,8% | 0,539 | 0,534 | 18 |
| CL-4 | 3 | 128 | 1 | yes | no | 56,3% | 0,563 | 0,571 | 26 |
| CL-5 | 2 | 128 | 1 | no | no | 55,3% | 0,553 | 0,563 | 28 |
| CL-6 aug | 2 | 128 | 1 | yes | yes | 53,8% | 0,536 | 0,544 | 100 |

Στο 80/20 φαίνεται καθαρά ότι το fine-tuning του wav2vec2 αλλάζει επίπεδο επίδοσης. Το frozen baseline έχει UAR 0,570, ενώ το fine-tuned μοντέλο φτάνει 0,727. Η διαφορά 0,157 UAR δείχνει ότι το pretrained μοντέλο έχει ήδη χρήσιμες αναπαραστάσεις, αλλά η προσαρμογή στα emotion labels του IEMOCAP είναι απαραίτητη για το τελικό άλμα.

## 9. Αποτελέσματα LOSO

Στο LOSO οι αριθμοί πέφτουν, όπως αναμένεται, επειδή ο `Ses05M` είναι άγνωστος ομιλητής. Το fine-tuned wav2vec2 παραμένει πρώτο σε καθαρές συνθήκες με accuracy 0,680, weighted F1 0,681, macro F1 0,687 και UAR 0,670. Το καλύτερο CNN είναι το `cnn_9`, όχι κάποιο από τα αρχικά standard CNNs. Το `cnn_9` φτάνει UAR 0,603, κάτι που επιβεβαιώνει ότι η frequency-aware pooling επιλογή είναι χρήσιμη σε speaker-independent αξιολόγηση. Το καλύτερο CNN-LSTM είναι το `cl_4` με UAR 0,577 και το καλύτερο MLP με βάση το summary είναι το `mlp_1` με UAR 0,574.

| Οικογένεια | Best config | Accuracy | Weighted F1 | Macro F1 | UAR |
|:---|:---|---:|---:|---:|---:|
| MLP | mlp_1 | 0,559 | 0,560 | 0,568 | 0,574 |
| CNN | cnn_9 | 0,583 | 0,580 | 0,595 | 0,603 |
| CNN-LSTM | cl_4 | 0,566 | 0,563 | 0,575 | 0,577 |
| wav2vec2 | finetuned | 0,680 | 0,681 | 0,687 | 0,670 |

| MLP config | Accuracy | Weighted F1 | Macro F1 | UAR | Best val F1 | Epochs | Augmented |
|:---|---:|---:|---:|---:|---:|---:|:---:|
| mlp_1 | 0,559 | 0,560 | 0,568 | 0,574 | 0,615 | 20 | no |
| mlp_2 | 0,529 | 0,527 | 0,539 | 0,552 | 0,601 | 25 | no |
| mlp_3 | 0,549 | 0,549 | 0,560 | 0,572 | 0,621 | 16 | no |
| mlp_4 | 0,543 | 0,536 | 0,555 | 0,574 | 0,611 | 15 | no |
| mlp_5 | 0,506 | 0,507 | 0,515 | 0,513 | 0,611 | 18 | no |
| mlp_6 | 0,548 | 0,547 | 0,551 | 0,553 | 0,818 | 76 | yes |

| CNN config | Type | Accuracy | Weighted F1 | Macro F1 | UAR | Best val F1 | Epochs | Augmented |
|:---|:---|---:|---:|---:|---:|---:|---:|:---:|
| cnn_1 | standard | 0,504 | 0,494 | 0,506 | 0,528 | 0,469 | 14 | no |
| cnn_2 | standard | 0,541 | 0,544 | 0,553 | 0,530 | 0,597 | 58 | no |
| cnn_3 | standard | 0,524 | 0,495 | 0,513 | 0,530 | 0,536 | 30 | no |
| cnn_4 | standard | 0,553 | 0,550 | 0,560 | 0,586 | 0,586 | 52 | no |
| cnn_5 | standard | 0,588 | 0,591 | 0,602 | 0,585 | 0,585 | 31 | no |
| cnn_6 | standard | 0,581 | 0,581 | 0,585 | 0,577 | 0,621 | 44 | yes |
| cnn_7 | gap | 0,543 | 0,545 | 0,552 | 0,533 | 0,750 | 100 | yes |
| cnn_8 | residual | 0,514 | 0,487 | 0,519 | 0,564 | 0,509 | 26 | yes |
| cnn_9 | freq_aware | 0,583 | 0,580 | 0,595 | 0,603 | 0,794 | 74 | yes |
| cnn_10 | deep_head | 0,566 | 0,567 | 0,574 | 0,562 | 0,758 | 100 | yes |

| CNN-LSTM config | CNN blocks | Hidden | Layers | BiLSTM | Accuracy | Weighted F1 | Macro F1 | UAR | Best val F1 | Epochs | Augmented |
|:---|---:|---:|---:|:---:|---:|---:|---:|---:|---:|---:|:---:|
| cl_1 | 2 | 128 | 1 | yes | 0,546 | 0,546 | 0,555 | 0,544 | 0,589 | 62 | no |
| cl_2 | 2 | 256 | 1 | yes | 0,566 | 0,557 | 0,569 | 0,570 | 0,567 | 34 | no |
| cl_3 | 2 | 128 | 2 | yes | 0,563 | 0,562 | 0,575 | 0,575 | 0,555 | 39 | no |
| cl_4 | 3 | 128 | 1 | yes | 0,566 | 0,563 | 0,575 | 0,577 | 0,588 | 33 | no |
| cl_5 | 2 | 128 | 1 | no | 0,539 | 0,536 | 0,551 | 0,552 | 0,575 | 35 | no |
| cl_6 | 2 | 128 | 1 | yes | 0,508 | 0,508 | 0,512 | 0,504 | 0,930 | 87 | yes |

Στο CNN-LSTM, το `cl_6` είναι χαρακτηριστική περίπτωση overfitting. Το best validation F1 φτάνει 0,930, αλλά το eval UAR είναι μόλις 0,504. Αυτό δείχνει ότι το validation split μέσα στο train set δεν αρκεί για να προβλέψει speaker-independent συμπεριφορά όταν το augmented training κάνει το μοντέλο να μάθει patterns που δεν μεταφέρονται καλά στον held-out speaker.

## 10. wav2vec2 interpretability

Το `wav2vec2_metrics.json` δείχνει ότι το frozen μοντέλο στο LOSO έχει UAR 0,540 και accuracy 0,486, ενώ το fine-tuned μοντέλο έχει UAR 0,670 και accuracy 0,680 μετά από 19 epochs. Η διαφορά είναι μεγάλη και επιβεβαιώνει ότι η emotion πληροφορία δεν ανακτάται πλήρως από ένα frozen generic speech encoder. Χρειάζεται fine-tuning τουλάχιστον στα ανώτερα transformer layers.

Το `layer_probing.json` δίνει μια πιο λεπτή εικόνα. Ένα logistic regression probe εκπαιδεύεται πάνω στα mean-pooled hidden states κάθε layer. Το layer 0 ξεκινά με UAR 0,570, τα πρώτα transformer layers ανεβάζουν την επίδοση πάνω από 0,60, και η καλύτερη τιμή εμφανίζεται στο layer 11 με UAR 0,693. Το τελικό layer 12 πέφτει ελαφρά σε 0,674. Αυτό δείχνει ότι η πληροφορία συναισθήματος κορυφώνεται κοντά στο τέλος αλλά όχι απαραίτητα στο τελευταίο layer.

| Layer | UAR |
|---:|---:|
| 0 | 0,570 |
| 1 | 0,604 |
| 2 | 0,640 |
| 3 | 0,631 |
| 4 | 0,638 |
| 5 | 0,654 |
| 6 | 0,660 |
| 7 | 0,653 |
| 8 | 0,628 |
| 9 | 0,664 |
| 10 | 0,687 |
| 11 | 0,693 |
| 12 | 0,674 |

![wav2vec2 training history](workflows/iemocap_dl/results/wav2vec2/training_history.png)

![wav2vec2 confusion matrix](workflows/iemocap_dl/results/wav2vec2/confusion_finetuned.png)

![wav2vec2 layer probing](workflows/iemocap_dl/results/wav2vec2/layer_probing.png)

![wav2vec2 attention visualization](workflows/iemocap_dl/results/wav2vec2/attention_visualization.png)

Η attention visualization δείχνει ότι το classification head δεν χρησιμοποιεί ομοιόμορφα όλο το utterance. Αντίθετα, δίνει βάρος σε συγκεκριμένα χρονικά σημεία, κάτι που είναι λογικό για emotion recognition, επειδή η ένταση, η προσωδία και τα φωνητικά cues συχνά εμφανίζονται σε σύντομες εκφραστικές περιοχές.

## 11. Noise robustness

Στο robustness evaluation η καθαρή επίδοση δεν αρκεί για να χαρακτηρίσει ένα μοντέλο. Το wav2vec2 είναι πρώτο στο clean LOSO με UAR 0,670, αλλά πέφτει πολύ έντονα σε Gaussian noise. Στα 10 dB πέφτει σε 0,371, στα 5 dB σε 0,283 και στα 0 dB σε 0,252. Το CNN-9, αντίθετα, ξεκινά χαμηλότερα στο clean με UAR 0,603, αλλά είναι πιο σταθερό σε Gaussian noise, κρατώντας 0,589 στα 20 dB, 0,557 στα 10 dB, 0,517 στα 5 dB και 0,464 στα 0 dB.

| Model | Best config στο robustness | Clean | G-20dB | G-10dB | G-5dB | G-0dB | R-0.3s | R-0.6s |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|
| MLP | mlp_6 | 0,543 | 0,529 | 0,517 | 0,500 | 0,427 | 0,537 | 0,499 |
| CNN | cnn_9 | 0,603 | 0,589 | 0,557 | 0,517 | 0,464 | 0,539 | 0,549 |
| CNN-LSTM | cl_4 | 0,577 | 0,383 | 0,324 | 0,282 | 0,301 | 0,479 | 0,414 |
| wav2vec2 | wav2vec2_finetuned_best.pt | 0,670 | 0,556 | 0,371 | 0,283 | 0,252 | 0,591 | 0,520 |

| Model | G-20dB denoised | G-10dB denoised | G-5dB denoised | G-0dB denoised | R-0.3s denoised | R-0.6s denoised |
|:---|---:|---:|---:|---:|---:|---:|
| MLP | 0,265 | 0,262 | 0,252 | 0,254 | 0,273 | 0,302 |
| CNN | 0,410 | 0,293 | 0,277 | 0,270 | 0,391 | 0,403 |
| CNN-LSTM | 0,251 | 0,250 | 0,250 | 0,250 | 0,252 | 0,268 |
| wav2vec2 | 0,467 | 0,396 | 0,335 | 0,288 | 0,397 | 0,387 |

![Noise robustness curves](workflows/iemocap_dl/results/noise_robustness/robustness_curves.png)

![Noise robustness heatmap](workflows/iemocap_dl/results/noise_robustness/robustness_heatmap.png)

![Denoising gain](workflows/iemocap_dl/results/noise_robustness/denoising_gain.png)

Το βασικό συμπέρασμα είναι ότι το καλύτερο μοντέλο σε καθαρή ομιλία δεν είναι απαραίτητα το καλύτερο μοντέλο σε θόρυβο. Για εφαρμογές όπου ο ήχος είναι ελεγχόμενος, το wav2vec2 είναι η καλύτερη επιλογή. Για εφαρμογές με έντονο Gaussian θόρυβο, το CNN-9 είναι πιο προβλέψιμο. Για reverberation, το wav2vec2 διατηρεί σχετικά καλή συμπεριφορά σε 0,3s και 0,6s, πιθανότατα επειδή το pretrained backbone έχει δει μεγάλη ποικιλία ακουστικών συνθηκών.

## 12. Όλα τα γραφήματα ανά μοντέλο

Στις επόμενες υποενότητες ενσωματώνονται όλα τα `.png` αρχεία που έχουν παραχθεί από τα πειράματα στο `workflows/iemocap_dl/results`. Για τα MLP, CNN και CNN-LSTM κάθε config έχει δύο εικόνες: training history και confusion matrix. Για το wav2vec2 υπάρχουν training history, confusion matrix, layer probing και attention visualization. Για το robustness υπάρχουν οι τρεις συγκεντρωτικές εικόνες που παρουσιάστηκαν ήδη και επαναλαμβάνονται εδώ ως πλήρης κατάλογος εξόδων.

### 12.1 MLP γραφήματα

| Config | History | Confusion matrix |
|:---|:---:|:---:|
| mlp_1 | ![mlp_1 history](workflows/iemocap_dl/results/mlp/mlp_1_history.png) | ![mlp_1 confusion](workflows/iemocap_dl/results/mlp/mlp_1_confusion.png) |
| mlp_2 | ![mlp_2 history](workflows/iemocap_dl/results/mlp/mlp_2_history.png) | ![mlp_2 confusion](workflows/iemocap_dl/results/mlp/mlp_2_confusion.png) |
| mlp_3 | ![mlp_3 history](workflows/iemocap_dl/results/mlp/mlp_3_history.png) | ![mlp_3 confusion](workflows/iemocap_dl/results/mlp/mlp_3_confusion.png) |
| mlp_4 | ![mlp_4 history](workflows/iemocap_dl/results/mlp/mlp_4_history.png) | ![mlp_4 confusion](workflows/iemocap_dl/results/mlp/mlp_4_confusion.png) |
| mlp_5 | ![mlp_5 history](workflows/iemocap_dl/results/mlp/mlp_5_history.png) | ![mlp_5 confusion](workflows/iemocap_dl/results/mlp/mlp_5_confusion.png) |
| mlp_6 | ![mlp_6 history](workflows/iemocap_dl/results/mlp/mlp_6_history.png) | ![mlp_6 confusion](workflows/iemocap_dl/results/mlp/mlp_6_confusion.png) |

### 12.2 CNN γραφήματα

| Config | History | Confusion matrix |
|:---|:---:|:---:|
| cnn_1 | ![cnn_1 history](workflows/iemocap_dl/results/cnn/cnn_1_history.png) | ![cnn_1 confusion](workflows/iemocap_dl/results/cnn/cnn_1_confusion.png) |
| cnn_2 | ![cnn_2 history](workflows/iemocap_dl/results/cnn/cnn_2_history.png) | ![cnn_2 confusion](workflows/iemocap_dl/results/cnn/cnn_2_confusion.png) |
| cnn_3 | ![cnn_3 history](workflows/iemocap_dl/results/cnn/cnn_3_history.png) | ![cnn_3 confusion](workflows/iemocap_dl/results/cnn/cnn_3_confusion.png) |
| cnn_4 | ![cnn_4 history](workflows/iemocap_dl/results/cnn/cnn_4_history.png) | ![cnn_4 confusion](workflows/iemocap_dl/results/cnn/cnn_4_confusion.png) |
| cnn_5 | ![cnn_5 history](workflows/iemocap_dl/results/cnn/cnn_5_history.png) | ![cnn_5 confusion](workflows/iemocap_dl/results/cnn/cnn_5_confusion.png) |
| cnn_6 | ![cnn_6 history](workflows/iemocap_dl/results/cnn/cnn_6_history.png) | ![cnn_6 confusion](workflows/iemocap_dl/results/cnn/cnn_6_confusion.png) |
| cnn_7 | ![cnn_7 history](workflows/iemocap_dl/results/cnn/cnn_7_history.png) | ![cnn_7 confusion](workflows/iemocap_dl/results/cnn/cnn_7_confusion.png) |
| cnn_8 | ![cnn_8 history](workflows/iemocap_dl/results/cnn/cnn_8_history.png) | ![cnn_8 confusion](workflows/iemocap_dl/results/cnn/cnn_8_confusion.png) |
| cnn_9 | ![cnn_9 history](workflows/iemocap_dl/results/cnn/cnn_9_history.png) | ![cnn_9 confusion](workflows/iemocap_dl/results/cnn/cnn_9_confusion.png) |
| cnn_10 | ![cnn_10 history](workflows/iemocap_dl/results/cnn/cnn_10_history.png) | ![cnn_10 confusion](workflows/iemocap_dl/results/cnn/cnn_10_confusion.png) |

### 12.3 CNN-LSTM γραφήματα

| Config | History | Confusion matrix |
|:---|:---:|:---:|
| cl_1 | ![cl_1 history](workflows/iemocap_dl/results/cnn_lstm/cl_1_history.png) | ![cl_1 confusion](workflows/iemocap_dl/results/cnn_lstm/cl_1_confusion.png) |
| cl_2 | ![cl_2 history](workflows/iemocap_dl/results/cnn_lstm/cl_2_history.png) | ![cl_2 confusion](workflows/iemocap_dl/results/cnn_lstm/cl_2_confusion.png) |
| cl_3 | ![cl_3 history](workflows/iemocap_dl/results/cnn_lstm/cl_3_history.png) | ![cl_3 confusion](workflows/iemocap_dl/results/cnn_lstm/cl_3_confusion.png) |
| cl_4 | ![cl_4 history](workflows/iemocap_dl/results/cnn_lstm/cl_4_history.png) | ![cl_4 confusion](workflows/iemocap_dl/results/cnn_lstm/cl_4_confusion.png) |
| cl_5 | ![cl_5 history](workflows/iemocap_dl/results/cnn_lstm/cl_5_history.png) | ![cl_5 confusion](workflows/iemocap_dl/results/cnn_lstm/cl_5_confusion.png) |
| cl_6 | ![cl_6 history](workflows/iemocap_dl/results/cnn_lstm/cl_6_history.png) | ![cl_6 confusion](workflows/iemocap_dl/results/cnn_lstm/cl_6_confusion.png) |

### 12.4 wav2vec2 γραφήματα

| Ομάδα | Εικόνα |
|:---|:---:|
| wav2vec2 training | ![wav2vec2 training history](workflows/iemocap_dl/results/wav2vec2/training_history.png) |
| wav2vec2 confusion | ![wav2vec2 confusion](workflows/iemocap_dl/results/wav2vec2/confusion_finetuned.png) |
| wav2vec2 layer probing | ![wav2vec2 layer probing](workflows/iemocap_dl/results/wav2vec2/layer_probing.png) |
| wav2vec2 attention | ![wav2vec2 attention](workflows/iemocap_dl/results/wav2vec2/attention_visualization.png) |


## 13. Συνολική σύγκριση

Η σύγκριση `80/20` και `LOSO` δείχνει πόσο σημαντικός είναι ο τρόπος με τον οποίο χωρίζονται τα δεδομένα. Στο `80/20` γίνεται τυχαίος stratified διαχωρισμός σε train και test, οπότε η αξιολόγηση είναι πιο αισιόδοξη επειδή μπορούν να εμφανιστούν παρόμοια speaker characteristics και στις δύο πλευρές. Στο `LOSO` κρατάμε ολόκληρο τον speaker `Ses05M` εκτός εκπαίδευσης, άρα μετράμε πιο αυστηρά αν το μοντέλο γενικεύει σε άγνωστη φωνή. Η πτώση του wav2vec2 από UAR 0,727 σε 0,670 είναι αναμενόμενη. Η μικρότερη πτώση του CNN από 0,617 σε 0,603 δείχνει ότι το spectrogram CNN, ειδικά με frequency-aware pooling, είναι αρκετά σταθερό όταν περνάμε από random split σε speaker-independent αξιολόγηση.

| Οικογένεια | Best 80/20 | UAR 80/20 | Best LOSO | UAR LOSO | Μεταβολή |
|:---|:---|---:|:---|---:|---:|
| MLP | MLP-5 | 0,633 | mlp_1 | 0,574 | -0,059 |
| CNN | CNN-6 | 0,617 | cnn_9 | 0,603 | -0,014 |
| CNN-LSTM | CL-1 | 0,579 | cl_4 | 0,577 | -0,002 |
| wav2vec2 | fine-tuned | 0,727 | fine-tuned | 0,670 | -0,057 |

Το `master_comparison.csv` συνοψίζει την τελική LOSO εικόνα. Σε καθαρό ήχο, το wav2vec2 είναι η καλύτερη επιλογή. Σε θορυβώδη Gaussian συνθήκη, το CNN-9 είναι πιο αξιόπιστο. Το MLP είναι απλό και δυνατό baseline, ειδικά αν ο στόχος είναι μικρό computational κόστος. Το CNN-LSTM δεν δικαιολογεί την πρόσθετη πολυπλοκότητά του σε αυτό το μέγεθος δεδομένων.

| Οικογένεια | Best config | Params | Accuracy | W-F1 | Macro F1 | Clean UAR | Best noised UAR | Worst noised UAR | Best denoised UAR |
|:---|:---|:---|---:|---:|---:|---:|---:|---:|---:|
| MLP | mlp_1 | ~100-500K | 0,5595 | 0,5602 | 0,5685 | 0,5742 | 0,5372 | 0,4272 | 0,3016 |
| CNN | cnn_9 | ~200-800K | 0,5829 | 0,5805 | 0,5946 | 0,6033 | 0,5895 | 0,4638 | 0,4099 |
| CNN-LSTM | cl_4 | ~500-2000K | 0,5662 | 0,5629 | 0,5755 | 0,5768 | 0,4794 | 0,2816 | 0,2675 |
| wav2vec2 | wav2vec2 | ~95M | 0,6801 | 0,6811 | 0,6873 | 0,6703 | 0,5911 | 0,2522 | 0,4674 |

## 14. Συμπεράσματα

Το ισχυρότερο καθαρό μοντέλο είναι το fine-tuned wav2vec2. Η χρήση pretrained self-supervised speech representations δίνει πολύ υψηλότερη επίδοση από τα μικρότερα μοντέλα, ειδικά στο 80/20. Στο LOSO εξακολουθεί να είναι πρώτο, αλλά η διαφορά μειώνεται και γίνεται φανερό ότι η γενίκευση σε νέο speaker είναι πιο δύσκολη από την απλή ταξινόμηση σε random split.

Το πιο πρακτικά ανθεκτικό μοντέλο είναι το CNN-9. Δεν έχει την καλύτερη clean επίδοση, αλλά έχει την καλύτερη συμπεριφορά σε Gaussian noise. Η frequency-aware pooling επιλογή είναι ουσιαστική, επειδή επιτρέπει στο CNN να διατηρεί πληροφορία για το πού βρίσκεται η ενέργεια στον άξονα συχνότητας αντί να συμπιέζει όλο το φάσμα σε ένα μόνο vector.

Το MLP πάνω σε handcrafted χαρακτηριστικά παραμένει ισχυρό baseline. Είναι ελαφρύ, γρήγορο και ερμηνεύσιμο σε επίπεδο features. Όμως η απώλεια χρονικής δομής το περιορίζει, ειδικά όταν ζητάμε υψηλή clean επίδοση ή λεπτή διάκριση μεταξύ κοντινών συναισθημάτων.

Το CNN-LSTM δεν κερδίζει από την επιπλέον χρονική μοντελοποίηση στο συγκεκριμένο setup. Η ιδέα είναι σωστή, αλλά τα αποτελέσματα δείχνουν ότι το μικρό μέγεθος του IEMOCAP και η speaker-independent αξιολόγηση κάνουν το recurrent κομμάτι ευάλωτο σε overfitting.

Το denoising δεν πρέπει να θεωρείται δεδομένα ωφέλιμο preprocessing. Ο spectral subtraction denoiser συχνά ρίχνει την UAR περισσότερο από τον ίδιο τον θόρυβο, πιθανότατα επειδή εισάγει artifacts που αλλάζουν τα φασματικά patterns. Σε μελλοντική εργασία θα είχε νόημα να δοκιμαστεί learned speech enhancement ή training με noised και denoised examples μαζί, αντί για ανεξάρτητο denoising πριν την πρόβλεψη.

Η τελική εικόνα είναι ότι ο διαχωρισμός των δεδομένων, η αναπαράσταση εισόδου και η ακουστική συνθήκη αλλάζουν ουσιαστικά την κατάταξη των μοντέλων. Στο `80/20` βλέπουμε την επίδοση σε ένα πιο εύκολο random split, ενώ στο `LOSO` βλέπουμε την επίδοση σε άγνωστο ομιλητή. Για καθαρή ομιλία, το wav2vec2 είναι η προφανής επιλογή. Για θορυβώδη περιβάλλοντα, το CNN-9 είναι πιο σταθερό. Για χαμηλό κόστος και απλή υλοποίηση, το MLP είναι το καλύτερο baseline. Αυτή η διάκριση είναι ίσως το σημαντικότερο αποτέλεσμα της εργασίας, γιατί δείχνει ότι το “καλύτερο” μοντέλο εξαρτάται από το σενάριο χρήσης και όχι μόνο από ένα clean benchmark.
