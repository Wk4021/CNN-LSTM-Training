import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Load the data
data = pd.read_csv('Data.csv')

# Split the raw radar data and the target values
radar_data = data['raw_data'].values
labels = data['labels'].values

# Convert labels to numeric format
targets = []
for label in labels:
    try:
        targets.append(eval(label))  # Convert string to list
    except Exception as e:
        print("Error parsing label:", label, "Error:", e)
        targets.append([0.0, 0.0, 0.0])  # Default value if parsing fails

targets = np.array(targets)

# Prepare radar points
def extract_radar_points(radar_data):
    radar_points = []
    for entry in radar_data:
        points = []
        for i in range(10):  # Extract first 10 segments
            segment = entry[i * 206:i * 206 + 206]
            try:
                float_value = int(segment, 16)  # Convert hex to int
                normalized_value = float_value / (2**64)  # Normalize based on expected range
                points.append(normalized_value)
            except ValueError as e:
                print(f"Error converting segment '{segment}' to float:", e)
                points.append(0.0)  # Default value if conversion fails
        radar_points.append(points)
    return np.array(radar_points)

# Extract radar points
X = extract_radar_points(radar_data)

# Convert X to float type explicitly
X = X.astype(float)

# Check for NaN or infinite values in the dataset
print("X NaN values:", np.isnan(X).any())
print("X infinite values:", np.isinf(X).any())
print("Statistics of X:", np.min(X), np.max(X), np.mean(X))

# Normalize target values
targets = (targets - np.mean(targets, axis=0)) / np.std(targets, axis=0)

# Check for NaN or infinite values in targets
print("Targets NaN values:", np.isnan(targets).any())
print("Targets infinite values:", np.isinf(targets).any())
print("Statistics of targets:", np.min(targets, axis=0), np.max(targets, axis=0), np.mean(targets, axis=0))

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, targets, test_size=0.2, random_state=42)

# Normalize the input data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create a simpler neural network
model = Sequential()
model.add(InputLayer(input_shape=(10,)))  # Input shape corresponds to the 10 radar points
model.add(Dense(8, activation='relu'))  # Fewer units to reduce complexity
model.add(Dense(3))  # Output layer for BPM, RR, PSI

# Compile the model with a smaller learning rate and gradient clipping
model.compile(optimizer=Adam(learning_rate=0.0001, clipnorm=1.0), loss='mean_squared_error', metrics=['mae'])

# Add early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=32, callbacks=[early_stopping])

# Save the model
model.save('bpm_rr_psi_model.h5')

# Evaluate the model on the test set
loss, mae = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}, Test MAE: {mae}")

# Example of making predictions with new radar data
def predict_bpm_rr_psi(new_radar_points):
    # new_radar_points should be preprocessed similarly to the training data
    prediction = model.predict(new_radar_points)
    return prediction

# Use this function with real-time radar data after preprocessing
