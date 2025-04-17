from joblib import load
import pandas as pd
import warnings

def load_model():
    """
    Load the pre-trained model from the joblib file.
    """
    load_NeuroLence = load('NeuroLens.joblib')
    return load_NeuroLence

def predict_data(model, data):
    """
    Predict the output using the pre-trained model.
    :param model: The pre-trained model.
    :param data: The input data for prediction.
    :return: The predicted output.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prediction = l_model.predict(row_data)[0]
    return prediction

# Replace with actual names used during training
df=pd.read_csv("new_eeg_labeled_dataset.csv")
l_model=load_model()
# Get only feature columns (exclude last 2 columns)
X = df.iloc[:, :-2]

# Get timestamp column (second last column)
timestamps = df.iloc[:, -2]


for i in range(len(X)):
    row_data = X.iloc[i].values.reshape(1, -1)
    timestamp = timestamps.iloc[i]
    prediction = predict_data(l_model,row_data)
    print(f"data :{row_data},Timestamp: {timestamp}, Prediction: {prediction}")
        

# p


