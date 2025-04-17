import torch
import numpy as np

# Define the EEGCNN model class (must match the one used during training)
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

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize model
input_channels = 64  # Number of EEG channels
output_dim = 64      # Output dimension (length of Z)
seq_length = 1       # Sequence length used during training
model = EEGCNN(input_channels=input_channels, output_dim=output_dim, seq_length=seq_length).to(device)

# Load the saved model state dictionary
model.load_state_dict(torch.load('/media/sudarshan/Windows-SSD/Users/sudar/OneDrive/Documents/Neuro Calm Updated/Model 2/eeg_cnn_model.pth', map_location=device))
model.eval()  # Set model to evaluation mode
print("Model loaded from 'eeg_cnn_model.pth'")

# Prepare input data (current_stress_features from original code)
current_stress_features = np.array([[
    -8.9e-05, -8.7e-05, -8.2e-05, -7.1e-05, -8e-05, -8.2e-05, -6e-05, -4.4e-05,
    -6.4e-05, -6.9e-05, -6.4e-05, -7.9e-05, -7.8e-05, -6.2e-05, -6.8e-05, -7.8e-05,
    -7.7e-05, -7.2e-05, -7.6e-05, -6.9e-05, -6.3e-05, -0.000168, -0.000157, -0.000165,
    -0.000177, -0.000169, -0.000119, -0.000115, -0.000128, -0.000125, -0.00013, -0.000117,
    -0.000101, -0.000108, -0.0001, -9.7e-05, -0.0001, -9.2e-05, -7.3e-05, -8.1e-05,
    -4e-05, -8.5e-05, -3.1e-05, -6.3e-05, -4.2e-05, -4.5e-05, -2.6e-05, -6.6e-05,
    -0.000103, -8.6e-05, -9.7e-05, -0.000102, -9.7e-05, -9.1e-05, -8.9e-05, -5e-05,
    -7.3e-05, -9.3e-05, -0.000108, -0.0001, -7.1e-05, -8.7e-05, -5.9e-05, 5e-06
]])
current_stress_features = current_stress_features.reshape(1, input_channels, seq_length)
current_stress_tensor = torch.tensor(current_stress_features, dtype=torch.float32).to(device)

# Make prediction
with torch.no_grad():
    predicted_distress = model(current_stress_tensor).cpu().numpy()[0]
print("Predicted calming signal features:", predicted_distress)


