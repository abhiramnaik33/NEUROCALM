import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
from joblib import load
import torch
import warnings
import logging

# Setup logging
logging.basicConfig(filename='app.log', level=logging.DEBUG, 
                    format='%(asctime)s %(levelname)s: %(message)s')

# ----------------------------
# EEGCNN PyTorch Model
# ----------------------------
class EEGCNN(torch.nn.Module):
    def __init__(self, input_channels, output_dim, seq_length):
        super(EEGCNN, self).__init__()
        self.conv1 = torch.nn.Conv1d(input_channels, 32, kernel_size=1)
        self.relu1 = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(32, 64, kernel_size=1)
        self.relu2 = torch.nn.ReLU()
        self.fc1 = torch.nn.Linear(64 * seq_length, 128)
        self.relu3 = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(128, output_dim)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x

# Load EEGCNN model
@st.cache_resource
def load_eegcnn_model():
    try:
        model = EEGCNN(input_channels=64, output_dim=64, seq_length=1)
        model.load_state_dict(torch.load("eeg_cnn_model.pth", map_location=torch.device("cpu")))
        model.eval()
        logging.info("EEGCNN model loaded successfully.")
        return model
    except FileNotFoundError:
        st.error("Error: 'eeg_cnn_model.pth' file not found in the working directory.")
        logging.error("EEGCNN model file not found.")
        return None
    except Exception as e:
        st.error(f"Error loading EEGCNN model: {e}")
        logging.error(f"EEGCNN model loading failed: {e}")
        return None

# Load ML model for stress detection
@st.cache_resource
def load_stress_model():
    try:
        model = load("NeuroLens.joblib")
        logging.info("Stress model loaded successfully.")
        return model
    except FileNotFoundError:
        st.error("Error: 'NeuroLens.joblib' file not found in the working directory.")
        logging.error("Stress model file not found.")
        return None
    except Exception as e:
        st.error(f"Error loading stress model: {e}")
        logging.error(f"Stress model loading failed: {e}")
        return None

# Inject Custom CSS for styling
def local_css():
    st.markdown("""
        <style>
        html, body, [class*="css"] {
            background-color: #0f0c29;
            background-image: radial-gradient(circle at top left, #0f0c29, #302b63 40%, #24243e 90%);
            color: #f0f0f0;
            font-family: 'Segoe UI', sans-serif;
        }
        h1, h2, h3, h4 {
            background: linear-gradient(90deg, #f81ce5, #7928ca, #2afadf);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 800;
        }
        .stAlert > div {
            font-size: 18px;
            font-weight: 600;
            background: linear-gradient(to right, #7928ca, #ff0080);
            color: white;
            border-radius: 8px;
            padding: 10px;
        }
        </style>
    """, unsafe_allow_html=True)

# Load EEG CSV
def load_data(file):
    try:
        df = pd.read_csv(file)
        if df.shape[1] < 66:  # 64 channels + timestamp + label
            st.error("Invalid CSV format: Must contain 64 EEG channels, timestamp, and label columns.")
            logging.error("Invalid CSV format: Insufficient columns.")
            return None, None, None
        X = df.iloc[:, :-2]  # All columns except last two
        timestamps = df.iloc[:, -2]  # Second-to-last column
        logging.info(f"CSV loaded: {len(df)} rows, {X.shape[1]} features.")
        return df, X, timestamps
    except Exception as e:
        st.error(f"Error loading CSV file: {e}")
        logging.error(f"CSV loading failed: {e}")
        return None, None, None

# Predict stress
def predict_stress(model, row):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return model.predict(row)[0]
    except Exception as e:
        st.error(f"Error predicting stress: {e}")
        logging.error(f"Stress prediction failed: {e}")
        return 0

# Generate calming signal
def generate_calming_signal(eegcnn_model, current_X):
    try:
        if current_X.shape[0] != 64:
            st.error("Input EEG frame must have 64 channels.")
            logging.error("Invalid EEG frame: Incorrect number of channels.")
            return None
        current_X = current_X.reshape(1, 64, 1)
        input_tensor = torch.tensor(current_X, dtype=torch.float32)
        with torch.no_grad():
            predicted = eegcnn_model(input_tensor).numpy()[0]
        logging.info("Calming signal generated successfully.")
        return predicted
    except Exception as e:
        st.error(f"Error generating calming signal: {e}")
        logging.error(f"Calming signal generation failed: {e}")
        return None

# Real-time EEG Plotting
def create_dynamic_plot(X, timestamps):
    model = load_stress_model()
    eegcnn_model = load_eegcnn_model()
    if model is None or eegcnn_model is None:
        st.error("Cannot proceed: Models failed to load.")
        logging.error("Models not loaded, aborting dynamic plot.")
        return

    # Dynamic channel selection with 5 default channels
    selected_channels = st.multiselect("Select EEG Channels to Plot", X.columns, default=list(X.columns[:5]))
    if not selected_channels:
        st.warning("Please select at least one channel to plot.")
        logging.warning("No channels selected for plotting.")
        return

    window_size = 300

    # Create three columns for EEG, Calming Signal, and X-Z plots
    with st.container():
        col1, col2, col3 = st.columns([3, 1, 1])

    # EEG plot setup
    fig_eeg, ax_eeg = plt.subplots(figsize=(8, 6))
    eeg_placeholder = col1.empty()

    # Calming signal plot setup
    fig_calming, ax_calming = plt.subplots(figsize=(4, 6))
    ax_calming.set_title("Calming Signal (Z)")
    ax_calming.set_xlabel("Channels")
    ax_calming.set_ylabel("Amplitude")
    ax_calming.grid(True)
    calming_placeholder = col2.empty()

    # X-Z difference plot setup
    fig_diff, ax_diff = plt.subplots(figsize=(4, 6))
    ax_diff.set_title("X - Z Waveform")
    ax_diff.set_xlabel("Channels")
    ax_diff.set_ylabel("Amplitude")
    ax_diff.grid(True)
    diff_placeholder = col3.empty()

    prediction_placeholder = col1.empty()
    message_placeholder = col1.empty()
    progress = st.progress(0)

    latest_calming_signal = None
    latest_eeg_frame = None
    calming_signal_active = False  # Track if calming signal was recently generated

    total_steps = (len(X) - window_size) // 10 + 1
    for i, end in enumerate(range(window_size, len(X), 10)):
        try:
            start = end - window_size
            ax_eeg.clear()

            # Plot EEG channels
            for ch in selected_channels:
                ax_eeg.plot(X[ch].iloc[start:end].values, label=ch, linewidth=1)

            row_data = X.iloc[end - 1].values.reshape(1, -1)
            timestamp = timestamps.iloc[end - 1]
            prediction = predict_stress(model, row_data)

            # Update EEG plot
            ax_eeg.set_xlim(0, window_size)
            ax_eeg.set_ylim(X.iloc[start:end].values.min(), X.iloc[start:end].values.max())
            ax_eeg.set_xlabel("Time (samples)")
            ax_eeg.set_ylabel("Amplitude")
            ax_eeg.set_title("EEG Waveform")
            ax_eeg.legend(loc='upper right')
            ax_eeg.grid(True)

            # Handle stress detection
            if prediction == 1:
                prediction_placeholder.markdown(
                    f"<div style='color:#ff4d4d; font-size:20px;'>⏱ {timestamp} | ⚠️ <strong>Stress Detected!</strong></div>",
                    unsafe_allow_html=True
                )
                message_placeholder.info("💆 Releasing Distress Signal...")
                if row_data.shape[1] != 64:
                    st.error("Input EEG frame must have 64 channels.")
                    logging.error("Invalid EEG frame for calming signal.")
                    continue
                latest_eeg_frame = row_data.reshape(64)
                latest_calming_signal = generate_calming_signal(eegcnn_model, latest_eeg_frame)
                calming_signal_active = True  # Mark calming signal as active
                
                if latest_calming_signal is not None:
                    # Update calming signal plot
                    ax_calming.clear()
                    ax_calming.plot(latest_calming_signal, label="Z", color="lime")
                    ax_calming.set_title("Calming Signal (Z)")
                    ax_calming.set_xlabel("Channels")
                    ax_calming.set_ylabel("Amplitude")
                    ax_calming.legend(loc='upper right')
                    ax_calming.grid(True)
                    calming_placeholder.pyplot(fig_calming)

                    # Update X-Z difference plot
                    ax_diff.clear()
                    x_minus_z = latest_eeg_frame - latest_calming_signal
                    ax_diff.plot(x_minus_z, label="X - Z", color="orange")
                    ax_diff.set_title("X - Z Waveform")
                    ax_diff.set_xlabel("Channels")
                    ax_diff.set_ylabel("Amplitude")
                    ax_diff.legend(loc='upper right')
                    ax_diff.grid(True)
                    diff_placeholder.pyplot(fig_diff)
            else:
                if calming_signal_active:
                    # Stress is no longer detected after calming signal
                    prediction_placeholder.markdown(
                        f"<div style='color:#00ff99; font-size:20px;'>⏱ {timestamp} | ✅ <strong>Stress Reduced!</strong></div>",
                        unsafe_allow_html=True
                    )
                    calming_signal_active = False  # Reset calming signal flag
                else:
                    prediction_placeholder.markdown(
                        f"<div style='color:#00ff99; font-size:20px;'>⏱ {timestamp} | ✅ No Stress</div>",
                        unsafe_allow_html=True
                    )
                message_placeholder.empty()  # Clear distress signal message

            # Always update EEG plot
            eeg_placeholder.pyplot(fig_eeg)

            # Keep showing latest calming and diff plots if they exist
            if latest_calming_signal is not None:
                ax_calming.clear()
                ax_calming.plot(latest_calming_signal, label="Z", color="lime")
                ax_calming.set_title("Calming Signal (Z)")
                ax_calming.set_xlabel("Channels")
                ax_calming.set_ylabel("Amplitude")
                ax_calming.legend(loc='upper right')
                ax_calming.grid(True)
                calming_placeholder.pyplot(fig_calming)

                ax_diff.clear()
                x_minus_z = latest_eeg_frame - latest_calming_signal
                ax_diff.plot(x_minus_z, label="X - Z", color="orange")
                ax_diff.set_title("X - Z Waveform")
                ax_diff.set_xlabel("Channels")
                ax_diff.set_ylabel("Amplitude")
                ax_diff.legend(loc='upper right')
                ax_diff.grid(True)
                diff_placeholder.pyplot(fig_diff)

            progress.progress(min(i / total_steps, 1.0))
            time.sleep(0.2)
        except Exception as e:
            st.error(f"Error in dynamic plot loop: {e}")
            logging.error(f"Dynamic plot loop failed: {e}")
            break

    st.success("🎉 Real-time EEG animation completed.")
    progress.empty()
    message_placeholder.empty()
    logging.info("Dynamic plot completed.")

# ----------------------------
# Streamlit App Entry
# ----------------------------
def main():
    st.set_page_config(page_title="NeuroCalm - EEG Stress Detection", layout="wide", page_icon="🧠")
    local_css()

    st.title("NeuroCalm - EEG Stress Detection Dashboard")
    st.markdown("Upload your EEG data to detect **stress levels** and counteract them with our AI calming system.")

    # Check if models are loaded
    if load_stress_model() is None or load_eegcnn_model() is None:
        st.error("Cannot start: Required models are missing or failed to load.")
        logging.error("Models missing, app startup aborted.")
        return

    uploaded_file = st.file_uploader("📂 Upload EEG CSV File", type="csv")

    if uploaded_file is not None:
        df, X, timestamps = load_data(uploaded_file)

        if df is not None:
            st.success("✅ Data loaded successfully.")
            st.write(f"📊 Total Records: {len(df)} | Features: {X.shape[1]}")

            if st.button("▶ Start EEG Prediction"):
                with st.spinner("Processing EEG signals and predicting stress..."):
                    try:
                        create_dynamic_plot(X, timestamps)
                    except Exception as e:
                        st.error(f"Error during EEG prediction: {e}")
                        logging.error(f"EEG prediction failed: {e}")

if __name__ == "__main__":
    main()