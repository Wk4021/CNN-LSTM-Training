import pandas as pd
import numpy as np
import shutil
import tempfile
import os
import ast
from tensorflow.keras.models import load_model
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# Hide the root window
Tk().withdraw()

# Function to select computation mode
def select_computation_mode():
    while True:
        mode = input("Select computation mode (cpu/gpu): ").strip().lower()
        if mode in ['cpu', 'gpu']:
            return mode
        else:
            print("Invalid input. Please enter 'cpu' or 'gpu'.")

# Function to load the model and data
def load_model_and_data():
    model_file = askopenfilename(title="Select the .h5 model file", filetypes=[("H5 Files", "*.h5")])
    if not model_file:
        print("No model file selected.")
        return None, None

    data_file = askopenfilename(title="Select the data.csv file", filetypes=[("CSV Files", "*.csv")])
    if not data_file:
        print("No data file selected.")
        return None, None

    model = load_model(model_file)
    
    return model, data_file

# Select computation mode
computation_mode = select_computation_mode()

# Configure TensorFlow based on the selected mode
if computation_mode == 'cpu':
    tf.config.set_visible_devices([], 'GPU')
    print("Running on CPU.")
else:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print("Running on GPU.")
    else:
        print("No GPU found. Defaulting to CPU.")

# Load model and data
model, data_file = load_model_and_data()
if model is None or data_file is None:
    exit()

# Create a temporary cache directory
with tempfile.TemporaryDirectory() as tmpdirname:
    cached_data_file = os.path.join(tmpdirname, 'cached_data.csv')
    shutil.copy(data_file, cached_data_file)
    
    data = pd.read_csv(cached_data_file)

# Process the first row of data
try:
    radar_data = data.iloc[0, 0]

    if not isinstance(radar_data, str) or not all(c in '0123456789abcdefABCDEF' for c in radar_data):
        raise ValueError("Radar data is not valid hexadecimal.")

    byte_data = bytes.fromhex(radar_data)

    if len(byte_data) % 2 != 0:
        raise ValueError("Buffer size must be a multiple of element size (2 bytes for float16).")

    radar_values = np.frombuffer(byte_data, dtype=np.float16)

    label_str = data.iloc[0, 1]
    labels = ast.literal_eval(label_str)

    y = np.array(labels).reshape(1, -1)

except Exception as e:
    print(f"Error processing the data: {e}")
    exit()

# Preprocess X if necessary
num_features = 1030
sequence_length = 50

if radar_values.size < num_features * sequence_length:
    print("Warning: Input radar data length is insufficient. Padding with zeros.")
    radar_values = np.pad(radar_values, (0, num_features * sequence_length - radar_values.size), mode='constant')

X = radar_values.reshape(1, sequence_length, num_features)

# Check the shape of X
print(f"Input shape for prediction: {X.shape}")

# Generate predictions
predictions = model.predict(X)

# Convert predictions to class labels if applicable
predicted_classes = np.round(predictions)  # Adjust this if needed for your task

# Print predictions
print(f"Predictions: {predicted_classes}")

# Evaluate model performance
if y.shape[1] == 1:  # Binary classification
    y_true = y.flatten()
    y_pred = predicted_classes.flatten()
else:  # Multi-class classification
    y_true = np.argmax(y, axis=1)
    y_pred = np.argmax(predicted_classes, axis=1)

# Calculate metrics
accuracy = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred, average='weighted')  # Use 'macro' for unweighted average
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')

# Print metrics
print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
