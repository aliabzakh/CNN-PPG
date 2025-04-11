import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import cheby2, filtfilt, find_peaks
from scipy.stats import skew

# -------------------------------
# Constants and Parameters
# -------------------------------
FS = 1000.0              # Sampling frequency in Hz
SEGMENT_LENGTH = 2100    # Expected number of samples per file

# -------------------------------
# Utility Functions for Signal Processing
# -------------------------------
def load_txt_file(file):
    """
    Load a .txt file containing a 1D PPG signal.
    If the file has double the expected length (duplicated samples),
    downsample by taking every second sample.
    Returns None if the signal length is not as expected.
    """
    try:
        data = np.loadtxt(file)
        if len(data) == 2 * SEGMENT_LENGTH:
            data = data[::2]
        if len(data) != SEGMENT_LENGTH:
            st.error(f"File length is {len(data)} (expected {SEGMENT_LENGTH}). Skipping file.")
            return None
        return data.astype(np.float32)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None

def bandpass_filter(signal):
    """
    Apply a fourth-order Chebyshev Type II bandpass filter (0.5–10 Hz)
    to the input signal using zero-phase filtering.
    """
    nyq = 0.5 * FS
    # Design filter: 4th order Chebyshev II with 20 dB attenuation outside passband
    b, a = cheby2(N=4, rs=20, Wn=[0.5/nyq, 10/nyq], btype='bandpass')
    filtered = filtfilt(b, a, signal)
    return filtered

def compute_sqi(signal):
    """
    Compute the Signal Quality Index (SQI) based on the skewness.
    A higher absolute skewness indicates a well-defined (asymmetric) pulse waveform.
    """
    return skew(signal)

def compute_auc(signal):
    """
    Compute the Area Under the Curve (AUC) of the signal,
    using the trapezoidal rule.
    """
    return np.trapz(signal, dx=1.0/FS)

# -------------------------------
# Dummy Prediction Function
# -------------------------------
def dummy_predict(input_signal):
    """
    Simulate predictions for three outputs:
      - Atrial fibrillation (2 classes: Normal Sinus Rhythm, Atrial Fibrillation)
      - Hypertension stage (4 classes: Normal, Prehypertension, Stage 1, Stage 2)
      - Anxiety (2 classes: No Anxiety, Anxiety)
    
    Returns three probability distributions (each sums to 1).
    """
    # Random probability for AF classification (2 classes)
    af = np.random.rand(2)
    af = af / np.sum(af)
    
    # Random probability for Hypertension classification (4 classes)
    ht = np.random.rand(4)
    ht = ht / np.sum(ht)
    
    # Random probability for Anxiety classification (2 classes)
    anx = np.random.rand(2)
    anx = anx / np.sum(anx)
    
    return af, ht, anx

# -------------------------------
# Streamlit Web App Interface
# -------------------------------
st.title("PPG Multi-Condition Classification (Dummy Model)")
st.write("""
Upload your PPG waveform text files below. Each file should contain a 1D array of numeric values representing
the PPG signal (expected length: 2100 samples, or double that if each sample is duplicated).
The application applies a Chebyshev II bandpass filter (0.5–10 Hz), computes the Signal Quality Index (SQI) via skewness,
and extracts the Area Under the Curve (AUC). Then, using a dummy predictor that simulates a multi-output model,
it outputs predictions for:
- **Atrial Fibrillation** (Normal vs Atrial Fibrillation)
- **Hypertension Stage** (Normal, Prehypertension, Stage 1, Stage 2)
- **Anxiety** (No Anxiety vs Anxiety)

For each file, the raw and filtered signals are plotted, and the SQI and AUC are displayed alongside the simulated model predictions
and a written explanation.
""")

uploaded_files = st.file_uploader("Upload your PPG .txt file(s)", type=["txt"], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        st.write(f"---\n### File: {uploaded_file.name}")
        
        # Load the PPG signal from the text file
        raw_signal = load_txt_file(uploaded_file)
        if raw_signal is None:
            continue  # skip this file if error
        
        # Preprocessing: filtering, SQI and AUC computation
        filtered_signal = bandpass_filter(raw_signal)
        sqi_value = compute_sqi(filtered_signal)
        auc_value = compute_auc(filtered_signal)
        
        # Plot the raw and filtered signals
        fig, ax = plt.subplots(2, 1, figsize=(10, 6))
        ax[0].plot(raw_signal, color='blue')
        ax[0].set_title("Raw PPG Signal")
        ax[0].set_xlabel("Sample")
        ax[0].set_ylabel("Amplitude")
        
        ax[1].plot(filtered_signal, color='orange')
        ax[1].set_title("Filtered PPG Signal")
        ax[1].set_xlabel("Sample")
        ax[1].set_ylabel("Amplitude")
        st.pyplot(fig)
        
        # Display computed features
        st.write(f"**Signal Quality Index (SQI):** {sqi_value:.2f}")
        st.write(f"**Area Under Curve (AUC):** {auc_value:.2f}")
        if abs(sqi_value) < 0.5:
            st.warning("Low signal quality detected (absolute skewness < 0.5). Classification results may be less reliable.")
        
        # Prepare the signal input for dummy prediction (dummy_predict expects an array shape similar to real model input)
        input_signal = filtered_signal.reshape(1, SEGMENT_LENGTH, 1)
        
        # Get simulated model predictions
        af_pred, ht_pred, anx_pred = dummy_predict(input_signal)
        
        # Convert probabilities to percentages
        af_percent = af_pred * 100
        ht_percent = ht_pred * 100
        anx_percent = anx_pred * 100
        
        # Define label names
        af_labels = ["Normal Sinus Rhythm", "Atrial Fibrillation"]
        ht_labels = ["Normal", "Prehypertension", "Stage 1 HT", "Stage 2 HT"]
        anx_labels = ["No Anxiety", "Anxiety"]
        
        # Determine predicted class for each condition (class with highest probability)
        af_idx = np.argmax(af_pred)
        ht_idx = np.argmax(ht_pred)
        anx_idx = np.argmax(anx_pred)
        
        af_pred_label = af_labels[af_idx]
        ht_pred_label = ht_labels[ht_idx]
        anx_pred_label = anx_labels[anx_idx]
        
        st.write("#### Dummy Model Predictions:")
        st.write(f"**Atrial Fibrillation:** {af_pred_label} (Probability: {af_pred[af_idx]:.2f})")
        st.write(f"**Hypertension Stage:** {ht_pred_label} (Probability: {ht_pred[ht_idx]:.2f})")
        st.write(f"**Anxiety:** {anx_pred_label} (Probability: {anx_pred[anx_idx]:.2f})")
        
        # Generate a written justification for the (dummy) predictions
        justification_text = "### Explanation:\n"
        justification_text += f"- The **SQI** is {sqi_value:.2f}. "
        if abs(sqi_value) >= 0.5:
            justification_text += "This indicates well-defined pulses, lending reliability to the extracted features.\n"
        else:
            justification_text += "This low SQI suggests the signal may be noisy, so the prediction should be interpreted with caution.\n"
        justification_text += f"- The **AUC** is {auc_value:.2f}, which provides insight into the overall energy of the pulse waveform.\n"
        
        # Atrial Fibrillation justification
        if af_pred[af_idx] > 0.90:
            justification_text += f"- The AF prediction has high confidence ({af_pred[af_idx]:.2f}); "
            if af_pred_label == "Normal Sinus Rhythm":
                justification_text += "this is consistent with regular pulse intervals.\n"
            else:
                justification_text += "irregularity in the pulse morphology suggests atrial fibrillation.\n"
        elif af_pred[af_idx] > 0.60:
            justification_text += f"- The AF prediction confidence is moderate ({af_pred[af_idx]:.2f}); some irregular features are present.\n"
        else:
            justification_text += f"- The AF prediction shows low confidence ({af_pred[af_idx]:.2f}), indicating uncertainty.\n"
            
        # Hypertension justification
        if ht_pred[ht_idx] > 0.90:
            justification_text += f"- The hypertension prediction is highly confident ({ht_pred[ht_idx]:.2f}), suggesting waveform features are clearly indicative of **{ht_pred_label}**.\n"
        elif ht_pred[ht_idx] > 0.60:
            justification_text += f"- The hypertension prediction is moderately confident ({ht_pred[ht_idx]:.2f}); certain features may point toward **{ht_pred_label}**.\n"
        else:
            justification_text += f"- The model is uncertain about the blood pressure stage (confidence: {ht_pred[ht_idx]:.2f}); the features may be ambiguous.\n"
            
        # Anxiety justification
        if anx_pred[anx_idx] > 0.90:
            justification_text += f"- The anxiety prediction is highly confident ({anx_pred[anx_idx]:.2f}), indicating clear stress markers in the signal.\n"
        elif anx_pred[anx_idx] > 0.60:
            justification_text += f"- The anxiety prediction is moderately confident ({anx_pred[anx_idx]:.2f}); there are some indications of stress-related changes.\n"
        else:
            justification_text += f"- The model shows low confidence in anxiety detection (confidence: {anx_pred[anx_idx]:.2f}), suggesting ambiguous features.\n"
        
        st.markdown(justification_text)
        st.markdown("---")
