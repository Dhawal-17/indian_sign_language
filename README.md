# Real-Time Indian Sign Language (ISL) Alphabet Recognition System

An improved, real-time Indian Sign Language (ISL) fingerspelling recognition system that recognizes static hand gestures (A–Z) from a live webcam feed using MobileNetV2 transfer learning, computer vision hand detection, temporal probability smoothing, and hold-to-confirm text transcription.

---

## 1. Project Overview & Key Improvements

### Problems in Existing Implementation
1. **White Space / Background Misclassification:** The original webcam pipeline resized the entire camera frame to 224×224 and fed it to the model. Without hand presence detection, empty space, walls, or clothing were forced through softmax and constantly misclassified as random letters.
2. **Poor Accuracy on Similar Letters (S and Z):** Baseline model achieved only **26.50%** on letter S and **25.00%** on letter Z. True Z gestures were mispredicted as T **74%** of the time.
3. **Thresholding & Fluttering:** Starting threshold of 0.35 allowed noisy predictions to pass, and rapid frame-to-frame predictions caused unstable text insertion.
4. **Dataset Loading Overhead:** Opening and validating 26,000 images on every run caused a 3-minute freeze at startup.

### Solved in this Updated Release
- **Integrated Hand Detection & Background Rejection:** Added a fast YCrCb skin segmentation and contour tracking detector in `webcam.py`. When no hand is placed in the designated zone, the system flags **"No Hand Detected"** and completely suppresses prediction, eliminating background false alarms.
- **Square Aspect Ratio Hand Framing:** Tightly crops detected hand gestures with 25% padding and square aspect ratio preservation to match training image geometry.
- **Domain-Adapted Fine-Tuning:** Unfroze the top convolutional blocks of MobileNetV2 (`fine_tune_at = 120`) and trained with class-weighted loss to penalize confused classes (S, Z, V, Q, X).
- **Measured Accuracy Gains:**
  - **Letter Z Accuracy:** Jumped from **25.00% to 48.50%** (almost doubled!).
  - **Letter X Accuracy:** Rose from **75.00% to 84.00%** (+9.00%).
  - **Letter S Accuracy:** Improved from **26.50% to 30.00%** (+3.50%).
  - **Overall Test Accuracy:** Rose from **90.36% to 91.29%** across all 5,198 test images.
  - **Letter T Precision:** Rose from **47.2% to 70.2%** (+23.0%) due to massive reduction in Z false positives.
- **Exponential Moving Average (EMA) Smoothing & Margin Gating:** Smooths raw prediction probabilities across consecutive frames (`alpha = 0.35`) and enforces a 15% probability gap between top-1 and top-2 candidates to reject ambiguous gestures.
- **Hold-to-Confirm (~1s) with On-Screen Progress Bar:** Displays a visual fill bar (`Hold: 0% -> 100%`) requiring the user to hold the gesture steadily for ~1 second before committing to the sentence buffer. Includes de-duplication lock to prevent duplicate character spamming.
- **Instant Dataset Caching:** Generated `reports/dataset_index.json`, reducing dataset initialization from 3 minutes to under 2 seconds.
- **Cross-Version Keras Compatibility:** Implemented `load_isl_model()` in `models.py` with automatic weights fallback across any Keras 3 version.

---

## 2. Folder Structure

```
isl-alphabet-recognition/
├── Dataset/
│   ├── Training/A–Z/          # 18,198 images (700 per letter)
│   ├── Validation/A–Z/        # 2,579 images (~100 per letter)
│   ├── Testing/A–Z/           # 5,198 images (200 per letter)
│   └── Letters/               # Reference letters A–Z
├── src/
│   ├── data.py                # Dataset pipeline, augmentation, index loader
│   ├── models.py              # MobileNetV2 architecture & load_isl_model()
│   ├── train.py               # Transfer learning & fine-tuning with class weights
│   ├── evaluate.py            # Comprehensive test evaluation & Before/After comparison
│   └── webcam.py              # Real-time webcam inference with hand detection & HUD
├── models/
│   ├── isl_model.keras        # Trained improved model
│   ├── baseline_model.keras   # Preserved original baseline model
│   └── model.weights.h5       # Cross-version model weights
├── reports/
│   ├── baseline_results.json           # Baseline metrics on test set
│   ├── baseline_confusion_matrix.png   # Baseline confusion matrix
│   ├── improved_results.json           # Improved metrics on test set
│   ├── improved_confusion_matrix.png   # Improved confusion matrix
│   ├── improved_training_curves.png    # Training loss & accuracy curves
│   └── comparison_report.md            # Detailed Before vs After report
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation & run guide
```

---

## 3. Measured Test Evaluation Results (Before vs After)

Evaluated on the separate, held-out **Testing Dataset (5,198 images across 26 classes)**:

| Metric / Letter | Baseline Model | Improved Model | Change | Impact |
| :--- | :---: | :---: | :---: | :--- |
| **Overall Test Accuracy** | **90.36%** | **91.29%** | **+0.92%** | Higher overall reliability |
| **Test Loss** | **0.4571** | **0.4039** | **-0.0532** | Lower prediction uncertainty |
| **Letter Z Accuracy** ⭐ | **25.00%** | **48.50%** | **+23.50%** | **Accuracy almost doubled** |
| **Letter X Accuracy** ⭐ | **75.00%** | **84.00%** | **+9.00%** | Significantly fewer misclassifications |
| **Letter S Accuracy** ⭐ | **26.50%** | **30.00%** | **+3.50%** | Reduced confusion with E, P, O |
| **Letter T Precision** | **47.2%** | **70.2%** | **+23.0%** | Fixed 60+ false T predictions |
| **Letters A, B, C, D, E, G, H, J, K, L, O, P, R, T, U, Y** | **100.00%** | **100.00%** | **0.00%** | Flawless accuracy preserved |

---

## 4. Quick Start & Run Instructions

### Prerequisites
Make sure Python 3.10+ is installed. Install all project requirements:
```bash
pip install -r requirements.txt
```

### 1. Run Live Webcam Recognition
```bash
python src/webcam.py
```
- A webcam window opens with an interactive **Hand Detection Zone** and HUD overlay.
- Position your hand inside the detection zone on the right side of the screen.
- The box turns **green** when your hand is tracked and **orange ("No Hand Detected")** when empty.
- Hold a gesture steady for ~1 second. The **Hold Progress Bar** will fill up from 0% to 100%, and the confirmed letter will be appended to the sentence buffer at the bottom.
- An inset thumbnail in the upper-right corner shows the exact cropped 224×224 hand ROI seen by the model.

**Interactive Keyboard Controls:**
| Key | Action |
| :--- | :--- |
| `Q` or `ESC` | Exit the webcam application cleanly |
| `Space` | Add a space between words in the text buffer |
| `B` or `Backspace` | Delete the last confirmed character |
| `C` | Clear the entire text buffer |
| `+` / `=` | Increase confidence threshold (+5%) |
| `-` / `_` | Decrease confidence threshold (-5%) |

### 2. Run Test Evaluation & Compare Before vs After
```bash
python src/evaluate.py
```
This loads the held-out `Testing/` dataset, computes per-class precision, recall, F1-score, accuracy, confusion matrix, and prints a side-by-side Before vs After comparison against the baseline.

### 3. Re-train or Fine-Tune Model
```bash
python src/train.py
```
Fine-tunes the MobileNetV2 top convolutional blocks using class-weighted cross-entropy on the `Training/` set and validates on the `Validation/` set, automatically saving best weights to `models/isl_model.keras` and updating training curves in `reports/`.

### 4. Verify Dataset Indexing
```bash
python src/data.py
```
Loads all dataset splits in parallel to confirm integrity and batching pipeline.
