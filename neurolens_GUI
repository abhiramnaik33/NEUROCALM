import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
from joblib import load
import warnings

# ----------------------------
# Inject Custom CSS
# ----------------------------
def local_css():
    st.markdown("""
        <style>
        /* Background Gradient Overlay */
        html, body, [class*="css"] {
            background-color: #0f0c29;
            background-image: radial-gradient(circle at top left, #0f0c29, #302b63 40%, #24243e 90%);
            color: #f0f0f0;
            font-family: 'Segoe UI', sans-serif;
        }

        /* Title and Headings */
        h1, h2, h3, h4 {
            background: linear-gradient(90deg, #f81ce5, #7928ca, #2afadf);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 800;
        }

        /* Buttons */
        .stButton > button {
            background: linear-gradient(to right, #7928ca, #ff0080);
            border: none;
            color: white;
            font-weight: 600;
            padding: 0.75em 1.5em;
            border-radius: 12px;
            box-shadow: 0 0 12px rgba(248, 28, 229, 0.6);
            transition: all 0.3s ease-in-out;
        }

        .stButton > button:hover {
            background: linear-gradient(to right, #ff4d4d, #f81ce5);
            transform: scale(1.05);
            box-shadow: 0 0 20px rgba(255, 77, 255, 0.8);
        }

        /* File Uploader */
        .stFileUploader {
            background-color: rgba(25, 25, 50, 0.6);
            border: 2px dashed #7928ca;
            border-radius: 10px;
            padding: 1rem;
            backdrop-filter: blur(10px);
        }

        /* Predictions */
        .stMarkdown {
            color: #dcdcdc;
        }

        /* Center content */
        .main > div {
            padding-top: 2rem;
        }
        </style>
    """, unsafe_allow_html=True)

# ----------------------------
# Load EEG Data
# ----------------------------
def load_data(file):
    try:
        df = pd.read_csv(file)
    except Exception as e:
        st.error(f"Error loading CSV file: {e}")
        return None, None, None

    X = df.iloc[:, :-2]
    timestamps = df.iloc[:, -2]
    return df, X, timestamps

# ----------------------------
# Load Trained Model
# ----------------------------
def load_model():
    return load("NeuroLens.joblib")

# ----------------------------
# Predict Stress
# ----------------------------
def predict_data(model, row_data):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return model.predict(row_data)[0]

# ----------------------------
# Plot EEG + Prediction
# ----------------------------
def create_dynamic_plot(X, timestamps):
    model = load_model()
    selected_channels = X.columns[:1].tolist()
    window_size = 300
    fig, ax = plt.subplots(figsize=(18, 9))
    plot_placeholder = st.empty()
    prediction_placeholder = st.empty()

    for end in range(window_size, len(X), 10):
        start = end - window_size
        ax.clear()

        # Plot EEG lines
        for ch in selected_channels:
            ax.plot(X[ch].iloc[start:end].values, label=ch, linewidth=1)

        row_data = X.iloc[end - 1].values.reshape(1, -1)
        timestamp = timestamps.iloc[end - 1]
        prediction = predict_data(model, row_data)

        # Prediction display
        if prediction == 1:
            label = "⚠️ <strong>Stress Detected!</strong>"
            prediction_placeholder.markdown(
                f"<div style='color:#ff4d4d; font-size:20px;'>⏱ {timestamp} &nbsp;&nbsp;|&nbsp;&nbsp; {label}</div>",
                unsafe_allow_html=True)
        else:
            label = "✅ No Stress"
            prediction_placeholder.markdown(
                f"<div style='color:#00ff99; font-size:20px;'>⏱ {timestamp} &nbsp;&nbsp;|&nbsp;&nbsp; {label}</div>",
                unsafe_allow_html=True)

        ax.set_xlim(0, window_size)
        ax.set_ylim(X.values.min(), X.values.max())
        ax.set_xlabel("Time (samples)")
        ax.set_ylabel("Amplitude")
        ax.set_title("EEG Waveform - Real-Time Prediction")
        ax.legend(loc='upper right')
        ax.grid(True)

        plot_placeholder.pyplot(fig)
        time.sleep(0.2)

    st.success("🎉 Real-time EEG animation completed.")

# ----------------------------
# Streamlit App Entry
# ----------------------------
def main():
    st.set_page_config(page_title="NeuroCalm - EEG Stress Detection", layout="wide", page_icon="🧠")
    local_css()

    st.title("NeuroCalm - EEG Stress Detection Dashboard")
    st.markdown("Upload your EEG data to detect **stress levels** in real-time using our machine learning model.")

    uploaded_file = st.file_uploader("📂 Upload EEG CSV File", type="csv")

    if uploaded_file is not None:
        df, X, timestamps = load_data(uploaded_file)

        if df is not None:
            st.success("✅ Data loaded successfully.")
            st.write(f"📊 Total Records: {len(df)} &nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp; Features: {X.shape[1]}")

            if st.button("▶ Start EEG Prediction"):
                with st.spinner("Processing EEG signals and predicting stress..."):
                    create_dynamic_plot(X, timestamps)

# ----------------------------
if __name__ == "__main__":
    main()
