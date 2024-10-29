import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Flatten, Dropout
from sklearn.preprocessing import StandardScaler

# Load and preprocess the data
def load_data(file_path):
    df = pd.read_csv(file_path, header=None)
    # Drop empty rows
    df = df.dropna()
    
    # Convert DataFrame to numpy array
    data = df.values
    return data

def create_chunks(data, chunk_size=16):
    chunks = []
    current_chunk = []
    
    for row in data:
        if len(current_chunk) < chunk_size:
            current_chunk.append(row)
        if len(current_chunk) == chunk_size:
            chunks.append(np.array(current_chunk))
            current_chunk = []
    
    return np.array(chunks)

def prepare_data(chunks):
    X = []
    y = []
    
    for chunk in chunks:
        heart_rate = chunk[:, -1]  # Assuming heart rate is the last element
        X.append(chunk[:, :-1])  # All except the last element for features
        y.append(heart_rate)
    
    return np.array(X), np.array(y)

# Define the model
def create_model(input_shape):
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(1))  # Output layer for predicting heart rate
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Main execution
input_file_path = r'R:\OHG\Cardiac_Respiratory_Phantom\Blood Pressure Processing\Training\Human Training\Data Management\split_data.csv'
output_model_path = r'R:\OHG\Cardiac_Respiratory_Phantom\Blood Pressure Processing\Training\Human Training\Human Models\cnn_lstm_heart_rate_model.h5'

data = load_data(input_file_path)
chunks = create_chunks(data)
X, y = prepare_data(chunks)

# Normalize the data
scaler = StandardScaler()
X = scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)

# Reshape for CNN input (samples, time steps, features)
X = X.reshape(X.shape[0], X.shape[1], X.shape[2])

# Create and train the model
model = create_model(X.shape[1:])
model.fit(X, y, epochs=50, batch_size=32)

# Save the model (architecture + weights)
model.save(output_model_path)

# Function for live prediction
def predict_live(model, new_data):
    # Assume new_data is a 2D array with the same structure as the training data
    new_data = scaler.transform(new_data)  # Normalize the new data
    new_data = new_data.reshape(1, new_data.shape[0], new_data.shape[1])  # Reshape for model
    prediction = model.predict(new_data)
    return prediction

# Example usage for live data
# new_data should be a 2D array with the last column as heart rate
# new_data = np.array([[...], [...], ..., [...]])
# predicted_heart_rate = predict_live(model, new_data)
