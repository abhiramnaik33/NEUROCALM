import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time

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
    
    # Get feature columns (exclude last 2 columns)
    X = df.iloc[:, :-2]
    # Get timestamp column (second last column)
    timestamps = df.iloc[:, -2]
    
    return df, X, timestamps
def create_dynamic_plot(X, timestamps):
    """
    Create a dynamic EEG waveform plot for multiple channels over time.
    X: EEG data (rows = time samples, columns = channels)
    timestamps: Time values (optional)
    """
    num_samples = len(X)
    selected_channels = X.columns[:5].tolist()
 # Use all channels or pick specific ones
   
    window_size = 300  # Number of samples to display at a time

    fig, ax = plt.subplots(figsize=(20, 10))

    # Create placeholder
    plot_placeholder = st.empty()

    for end in range(window_size, num_samples, 10):  # Slide window
        start = end - window_size
        ax.clear()

        for ch in selected_channels:
            ax.plot(X[ch].iloc[start:end].values, label=ch)

        ax.set_xlim(0, window_size)
        ax.set_ylim(X.values.min(), X.values.max())
        ax.set_xlabel("Time (samples)")
        ax.set_ylabel("Amplitude")
        ax.set_title(f"EEG Waveform for Selected Channels (Samples {start}–{end})")
        ax.legend(loc='upper right')
        ax.grid(True)

        plot_placeholder.pyplot(fig)
        time.sleep(0.2)  # Adjust speed

    st.success("EEG waveform animation completed.")

    
    # Create a placeholder for the plot
    plot_placeholder = st.empty()
    
    # Animation loop
    for i in range(len(X)):
        # Get data for the current row
        row_data = X.iloc[i].values
        timestamp = timestamps.iloc[i]
        
        # Update line data
        line.set_ydata(row_data)
        
        # Update title with timestamp
        ax.set_title(f'EEG Data: Timestamp = {timestamp}')
        
        # Display the plot
        plot_placeholder.pyplot(fig)
        
        # Add a delay to simulate animation
        time.sleep(0.5)  # 500ms delay
        
        # Clear the plot for the next frame
        plt.cla()
        line, = ax.plot(x_axis, np.zeros(num_features), 'b-', label='EEG Features')
        ax.set_xlim(0, num_features - 1)
        ax.set_ylim(X.min().min(), X.max().max())
        ax.set_xlabel('Feature Index')
        ax.set_ylabel('Feature Value')
        ax.set_title(f'EEG Data: Timestamp = {timestamp}')
        ax.grid(True)
        ax.legend()
    
    # Clear the placeholder after animation
    plot_placeholder.empty()
    st.write("Animation completed.")

def main():
    """
    Main Streamlit app function.
    """
    st.title("EEG Data Dynamic Visualization")
    st.write("Upload a CSV file containing EEG data to visualize feature values over time.")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        # Load data
        df, X, timestamps = load_data(uploaded_file)
        
        if df is not None:
            st.write("Data loaded successfully!")
            st.write(f"Number of rows: {len(df)}")
            st.write(f"Number of features: {X.shape[1]}")
            
            # Button to start animation
            if st.button("Start Animation"):
                with st.spinner("Generating animation..."):
                    create_dynamic_plot(X, timestamps)

if __name__ == "__main__":
    main()