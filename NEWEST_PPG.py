# ppg_hypertension_pipeline.py  (or run cell‑by‑cell in Jupyter)

import os, glob, re, json, warnings
import numpy as np
import pandas as pd
from scipy.signal import cheby2, filtfilt, find_peaks
from scipy.stats import skew
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.layers import (Conv1D, MaxPooling1D, GlobalAveragePooling1D,
                                     Dense, Dropout, BatchNormalization)
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ------------------------------------------------------------------
# 1. LOAD LABELS  (adjust path if needed)
# ------------------------------------------------------------------
xl_path = "PPG-BP dataset.xlsx"
df = pd.read_excel(xl_path, header=1)      # second row is the real header
df = df.rename(columns=str.strip)          # trim whitespace
LABEL_COL = "Hypertension"                 # exact header from screenshot
ID_COL    = "subject_ID"

label_map = {c: i for i, c in enumerate(
    ["Normal", "Prehypertension", "Stage 1 hypertension", "Stage 2 hypertension"])}
df["label"] = df[LABEL_COL].map(label_map)
print("Label distribution:", df["label"].value_counts().to_dict())

# ------------------------------------------------------------------
# 2. LOAD WAVEFORMS
# ------------------------------------------------------------------
DATA_DIR = "Data File/0_subject"           # adjust if different
SEGMENTS = ["1", "2", "3"]
FS       = 1000.0                          # Hz
LEN      = 2100                            # samples

def load_txt(path):
    """Load a <ID>_<seg>.txt into 1‑D numpy array of length 2100."""
    raw = np.loadtxt(path, dtype=float, delimiter=None)  # autodetect
    # The provided sample shows each true sample duplicated.
    if len(raw) == 2*LEN:
        raw = raw[::2]
    if len(raw) != LEN:
        raise ValueError(f"{path}: expected {LEN}, got {len(raw)}")
    return raw

signals, labels = [], []
bad_segments   = 0
for sid, row in df.iterrows():
    subj_id = int(row[ID_COL])
    for seg in SEGMENTS:
        fpath = os.path.join(DATA_DIR, f"{subj_id}_{seg}.txt")
        if not os.path.exists(fpath):
            continue
        sig = load_txt(fpath)

        # 3a. Chebyshev II filter (0.5–10 Hz)
        nyq   = 0.5*FS
        b, a  = cheby2(4, 20, [0.5/nyq, 10.0/nyq], btype='bandpass')
        sig_f = filtfilt(b, a, sig)

        # 3b. SQI via skewness
        sk = skew(sig_f)
        if abs(sk) < 0.5:
            bad_segments += 1
            continue

        signals.append(sig_f.astype(np.float32))
        labels.append(row["label"])

print(f"Loaded {len(signals)} good segments  |  skipped {bad_segments} low‑SQI segments")

X = np.stack(signals)[..., None]   # shape: (N,2100,1)
y = np.array(labels)

# ------------------------------------------------------------------
# 4. OPTIONAL FEATURE EXTRACTION (for plots / interpretability)
# ------------------------------------------------------------------
def auc(signal):       # trapezoidal integral
    return np.trapz(signal, dx=1.0/FS)

def notch_features(signal):
    peaks, _ = find_peaks(signal, distance=300)
    if len(peaks) < 1:
        return np.nan, np.nan
    p = peaks[0]
    # search 50‑300 ms after peak for local minimum (notch)
    search = signal[p+50:p+300] if p+300 < len(signal) else signal[p+50:]
    if len(search) == 0:
        return np.nan, np.nan
    notch_rel = np.argmin(search)
    notch_idx = p + 50 + notch_rel
    t_delay   = (notch_idx - p)/FS
    amp_ratio = signal[notch_idx] / signal[p]
    return t_delay, amp_ratio

feat_rows = []
for s in signals:
    td, ar = notch_features(s)
    feat_rows.append({"AUC": auc(s), "NotchDelay": td, "NotchRatio": ar})
feat_df = pd.DataFrame(feat_rows)

# ------------------------------------------------------------------
# 5. TRAIN / TEST SPLIT
# ------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

cw = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
class_weights = {i: w for i, w in enumerate(cw)}

# ------------------------------------------------------------------
# 6. BUILD 1‑D CNN
# ------------------------------------------------------------------
model = Sequential([
    Conv1D(16, 5, activation='relu', input_shape=(LEN,1)),
    MaxPooling1D(2),
    BatchNormalization(),
    Conv1D(32, 5, activation='relu'),
    MaxPooling1D(2),
    Conv1D(64, 5, activation='relu'),
    GlobalAveragePooling1D(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(len(label_map), activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

es = EarlyStopping(monitor='val_accuracy', patience=7, restore_best_weights=True)
history = model.fit(X_train, y_train, epochs=60, batch_size=64,
                    validation_split=0.15, class_weight=class_weights,
                    callbacks=[es], verbose=2)

# ------------------------------------------------------------------
# 7. EVALUATION
# ------------------------------------------------------------------
y_pred = np.argmax(model.predict(X_test), axis=1)
print(classification_report(y_test, y_pred,
      target_names=list(label_map.keys())))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_map.keys(),
            yticklabels=label_map.keys())
plt.title("Confusion Matrix"); plt.ylabel("True"); plt.xlabel("Predicted")
plt.show()

# ------------------------------------------------------------------
# 8. BAR & PIE CHARTS  (mirroring your reference graphs)
# ------------------------------------------------------------------
age_bins = pd.cut(df["Age(year)"], bins=[20,30,40,50,60,70,80,90], right=False)
age_stage = pd.crosstab(age_bins, df[LABEL_COL])
age_stage.plot(kind='bar', stacked=True, figsize=(8,4))
plt.title("Hypertension stage distribution by age group")
plt.ylabel("No. of subjects"); plt.show()

df[LABEL_COL].value_counts().plot(kind='pie', autopct='%1.1f%%',
                                  colors=["#4CAF50","#64B5F6","#FFD54F","#E57373"])
plt.title("Overall hypertension distribution"); plt.ylabel(""); plt.show()

# ------------------------------------------------------------------
# 9. SIMPLE TRAINER CLASS FOR BIGGER DATASETS
# ------------------------------------------------------------------
class Trainer:
    def __init__(self, data_dir, label_file):
        self.data_dir = data_dir
        self.labels   = pd.read_excel(label_file, header=1).rename(columns=str.strip)

    def load_all(self):
        # same logic as above – omitted for brevity
        ...

    def train(self):
        self.load_all()
        # build model, fit, save weights
        ...

# Example future use:
# big = Trainer("/mnt/40k_ppg", "big_labels.xlsx")
# big.train()
