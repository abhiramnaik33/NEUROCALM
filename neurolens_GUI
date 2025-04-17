import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
from joblib import load
import warnings

# ----------------------------
# Load EEG CSV data
# ----------------------------
def load_data(file):
    """
    Load the EEG dataset from a CSV file.
    :param file: Uploaded file object or file path.
    :return: DataFrame, feature data (X), and timestamps.
    """
    try:
        df = pd.read_csv(file)
    except Exception as e:
        st.error(f"Error loading CSV file: {e}")
        return None, None, None

    X = df.iloc[:, :-2]  # All feature columns (exclude last 2)
    timestamps = df.iloc[:, -2]  # Timestamp column
    return df, X, timestamps

# ----------------------------
# Load trained model
# ----------------------------
def load_model():
    return load("NeuroLens.joblib")

# ----------------------------
# Predict stress
# ----------------------------
def predict_data(model, row_data):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prediction = model.predict(row_data)[0]
    return prediction

# ----------------------------
# Plot EEG animation + prediction
# ----------------------------
def create_dynamic_plot(X, timestamps):
    model = load_model()
    num_samples = len(X)
    selected_channels = X.columns[:5].tolist()  # First 10 EEG channels
    window_size = 300  # Number of time samples in moving window

    fig, ax = plt.subplots(figsize=(18, 9))
    plot_placeholder = st.empty()
    prediction_placeholder = st.empty()

    for end in range(window_size, num_samples, 10):
        start = end - window_size
        ax.clear()

        # Plot EEG lines
        for ch in selected_channels:
            ax.plot(X[ch].iloc[start:end].values, label=ch)

        # Get prediction for current timestamp
        row_data = X.iloc[end - 1].values.reshape(1, -1)
        timestamp = timestamps.iloc[end - 1]
        prediction = predict_data(model, row_data)

        # Format prediction
        if prediction == 1:
            label = "⚠️ **Stress Detected!**"
            prediction_placeholder.markdown(f"<div style='color:red; font-size:20px;'>⏱ Timestamp: {timestamp} &nbsp;&nbsp;|&nbsp;&nbsp; {label}</div>", unsafe_allow_html=True)
        else:
            label = "✅ No Stress"
            prediction_placeholder.markdown(f"<div style='color:green; font-size:20px;'>⏱ Timestamp: {timestamp} &nbsp;&nbsp;|&nbsp;&nbsp; {label}</div>", unsafe_allow_html=True)

        # Set plot parameters
        ax.set_xlim(0, window_size)
        ax.set_ylim(X.values.min(), X.values.max())
        ax.set_xlabel("Time (samples)")
        ax.set_ylabel("Amplitude")
        ax.set_title("EEG Waveform - Live Prediction")
        ax.legend(loc='upper right')
        ax.grid(True)

        # Display plot
        plot_placeholder.pyplot(fig)
        time.sleep(0.2)

    st.success("EEG animation with prediction completed.")

# ----------------------------
# Streamlit app
# ----------------------------
def main():
    st.title("🧠 Real-Time EEG Stress Detection")
    st.markdown("Upload an EEG `.csv` file to visualize and detect **stress levels** dynamically.")
    
    uploaded_file = st.file_uploader("📂 Upload EEG CSV", type="csv")

    if uploaded_file is not None:
        df, X, timestamps = load_data(uploaded_file)
        
        if df is not None:
            st.success("✅ Data loaded successfully.")
            st.write(f"📊 Rows: {len(df)} &nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp; Features: {X.shape[1]}")
            
            if st.button("▶ Start EEG Animation with Stress Prediction"):
                with st.spinner("Running real-time prediction..."):
                    create_dynamic_plot(X, timestamps)

# ----------------------------
if __name__ == "__main__":
    main()
