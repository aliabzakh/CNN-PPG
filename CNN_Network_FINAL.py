import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from scipy.signal import cheby2, filtfilt
from scipy.stats import skew
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, BatchNormalization, Dropout, Flatten, Dense

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
            st.error(f"File length is {len(data)} (expected {SEGMENT_LENGTH}). Skipping file: {file.name}")
            return None
        return data.astype(np.float32)
    except Exception as e:
        st.error(f"Error reading file {file.name}: {e}")
        return None

def bandpass_filter(signal):
    """
    Apply a fourth-order Chebyshev Type II bandpass filter (0.5–10 Hz)
    to the input signal using zero-phase filtering.
    """
    nyq = 0.5 * FS
    b, a = cheby2(N=4, rs=20, Wn=[0.5/nyq, 10/nyq], btype='bandpass')
    filtered = filtfilt(b, a, signal)
    return filtered

def compute_sqi(signal):
    """
    Compute the Signal Quality Index (SQI) based on the skewness.
    A higher absolute skewness indicates a well‐defined (asymmetric) pulse waveform.
    """
    return skew(signal)

def compute_auc(signal):
    """
    Compute the Area Under the Curve (AUC) of the signal, using the trapezoidal rule.
    """
    return np.trapz(signal, dx=1.0/FS)

# -------------------------------
# Build the Multi-Output 1D CNN (Original)
# -------------------------------
def build_model():
    """
    Build a 1D CNN that accepts an input of shape (SEGMENT_LENGTH, 1)
    and outputs three predictions:
      - Atrial fibrillation (2 classes: Normal Sinus Rhythm, Atrial Fibrillation)
      - Hypertension stage (4 classes: Normal, Prehypertension, Stage 1 HT, Stage 2 HT)
      - Anxiety (2 classes: No Anxiety, Anxiety)
    """
    inputs = Input(shape=(SEGMENT_LENGTH, 1))
    
    # First convolution block
    x = Conv1D(filters=32, kernel_size=5, activation='relu')(inputs)
    x = MaxPooling1D(pool_size=2)(x)
    x = BatchNormalization()(x)
    
    # Second convolution block
    x = Conv1D(filters=64, kernel_size=5, activation='relu')(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = BatchNormalization()(x)
    
    # Third convolution block
    x = Conv1D(filters=128, kernel_size=5, activation='relu')(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = BatchNormalization()(x)
    
    # Flatten and fully-connected layers
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.25)(x)
    x = Dense(64, activation='relu')(x)
    
    # Three output heads:
    af_out  = Dense(2, activation='softmax', name='af')(x)            # Atrial fibrillation: Normal vs AF
    ht_out  = Dense(4, activation='softmax', name='hypertension')(x)    # Hypertension: 4 classes
    anx_out = Dense(2, activation='softmax', name='anxiety')(x)         # Anxiety: No Anxiety vs Anxiety
    
    model = Model(inputs=inputs, outputs=[af_out, ht_out, anx_out])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Instantiate the model. (Replace or load pretrained weights here in production.)
model = build_model()

# -------------------------------
# Streamlit Web Interface
# -------------------------------
st.title("Aggregated PPG Multi-Condition Classification")
st.write("""
Upload up to **200 PPG waveform text files**. Each file should contain a 1D array of numeric values (expected length: 2100 samples, or double that if samples are duplicated).  
After uploading, click the **"Run the Network"** button to process all files.  
Aggregated outputs will display:
- A set of pie charts summarizing the predicted conditions for Atrial Fibrillation, Hypertension stage, and Anxiety.
- (Optional: Aggregated graphs or statistics can be added as needed.)
""")

uploaded_files = st.file_uploader("Upload your PPG .txt file(s)", type=["txt"], accept_multiple_files=True)

if uploaded_files:
    st.write(f"Uploaded {len(uploaded_files)} file(s).")
    
    # Add a "Run the Network" button to trigger processing.
    if st.button("Run the Network"):
        # Limit processing to at most 200 files.
        files_to_process = uploaded_files[:200]
        
        # Initialize lists to store predictions
        af_results = []
        ht_results = []
        anx_results = []
        sqi_list = []  # (for possible quality stats)
        auc_list = []  # (for possible energy stats)
        
        # Define label names
        af_labels = ["Normal Sinus Rhythm", "Atrial Fibrillation"]
        ht_labels = ["Normal", "Prehypertension", "Stage 1 HT", "Stage 2 HT"]
        anx_labels = ["No Anxiety", "Anxiety"]
        
        # Process each file:
        for file in files_to_process:
            # Load signal
            raw_signal = load_txt_file(file)
            if raw_signal is None:
                continue  # skip if file format issues
            
            # Filter the signal
            filtered_signal = bandpass_filter(raw_signal)
            # Compute features
            sqi = compute_sqi(filtered_signal)
            auc_val = compute_auc(filtered_signal)
            sqi_list.append(sqi)
            auc_list.append(auc_val)
            
            # Prepare the input for CNN
            input_signal = filtered_signal.reshape(1, SEGMENT_LENGTH, 1)
            
            # Get predictions (using the original model)
            preds = model.predict(input_signal, verbose=0)
            af_pred = preds[0]
            ht_pred = preds[1]
            anx_pred = preds[2]
            
            # Determine predicted class indices (highest probability)
            af_idx = np.argmax(af_pred)
            ht_idx = np.argmax(ht_pred)
            anx_idx = np.argmax(anx_pred)
            
            # Save results
            af_results.append(af_labels[af_idx])
            ht_results.append(ht_labels[ht_idx])
            anx_results.append(anx_labels[anx_idx])
        
        # If no files were processed successfully:
        if not af_results:
            st.error("No files were processed successfully. Please check your file formats.")
        else:
            st.success(f"Processed {len(af_results)} file(s).")
            
            # Aggregate results using Counter from collections
            af_counts = Counter(af_results)
            ht_counts = Counter(ht_results)
            anx_counts = Counter(anx_results)
            
            # Display aggregated counts (optional text output)
            st.write("### Aggregated Classification Results")
            st.write("**Atrial Fibrillation:**", dict(af_counts))
            st.write("**Hypertension Stage:**", dict(ht_counts))
            st.write("**Anxiety:**", dict(anx_counts))
            
            # Create aggregated pie charts for each condition in one figure.
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            # Pie chart for Atrial Fibrillation results
            axes[0].pie(list(af_counts.values()), labels=list(af_counts.keys()), autopct='%1.1f%%', startangle=90)
            axes[0].set_title("Atrial Fibrillation Distribution")
            
            # Pie chart for Hypertension results
            axes[1].pie(list(ht_counts.values()), labels=list(ht_counts.keys()), autopct='%1.1f%%', startangle=90)
            axes[1].set_title("Hypertension Stage Distribution")
            
            # Pie chart for Anxiety results
            axes[2].pie(list(anx_counts.values()), labels=list(anx_counts.keys()), autopct='%1.1f%%', startangle=90)
            axes[2].set_title("Anxiety Distribution")
            
            st.pyplot(fig)
            
            # (Optional) Display aggregated box plots or histograms for SQI and AUC if desired:
            fig2, ax2 = plt.subplots(1, 2, figsize=(12, 5))
            ax2[0].hist(sqi_list, bins=20, color='skyblue', edgecolor='black')
            ax2[0].set_title("Distribution of SQI")
            ax2[0].set_xlabel("SQI (skewness)")
            ax2[0].set_ylabel("Count")
            
            ax2[1].hist(auc_list, bins=20, color='salmon', edgecolor='black')
            ax2[1].set_title("Distribution of AUC")
            ax2[1].set_xlabel("AUC")
            ax2[1].set_ylabel("Count")
            
            st.pyplot(fig2)
