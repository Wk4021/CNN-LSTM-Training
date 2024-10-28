import csv
import numpy as np
from tqdm import tqdm  # Import tqdm for progress bar
import random
import ast  # To safely evaluate the BPM, RR, PSI from string

# File paths
input_file = 'preprocessed_data.csv'
train_file = 'train_data.csv'
val_file = 'val_data.csv'
test_file = 'test_data.csv'

# Load preprocessed data
data = []
labels = []

# Read the preprocessed CSV file
with open(input_file, 'r') as csvfile:
    reader = csv.reader(csvfile)
    for row in tqdm(reader, desc="Loading data", unit="row"):
        # Each row contains the preprocessed hex data followed by labels
        data.append(row[:-1])  # All except last are data
        labels.append(row[-1])  # Last element is the label (BPM, RR, PSI)

# Convert to numpy arrays for easier manipulation
data = np.array(data, dtype=np.float32)  # Use float32 for numerical operations
labels = np.array(labels)

# Prepare to extract BPM and RR from labels
bpm_rr = []
for label in labels:
    try:
        bpm_rr_value = ast.literal_eval(label)  # Convert string to list [BPM, RR, PSI]
        bpm_rr.append(bpm_rr_value)  # Add the BPM and RR list to our list
    except (ValueError, SyntaxError):
        print(f"Invalid label format: {label}")
        bpm_rr.append([None, None, None])  # Append None for invalid entries

# Convert bpm_rr to a numpy array for easier indexing
bpm_rr = np.array(bpm_rr)

# Filter data based on the specified ranges for BPM and RR
mask = (bpm_rr[:, 0] >= 80) & (bpm_rr[:, 0] <= 100) & \
       (bpm_rr[:, 1] >= 20) & (bpm_rr[:, 1] <= 30)

filtered_data = data[mask]
filtered_labels = labels[mask]

# Debugging: Print the number of filtered data points
print(f"Number of filtered data points: {len(filtered_data)}")

# Group by unique combinations of BPM and RR
combinations = {}
for d, label in zip(filtered_data, filtered_labels):
    bpm = ast.literal_eval(label)[0]  # Get BPM from the label
    rr = ast.literal_eval(label)[1]   # Get RR from the label
    key = (bpm, rr)  # Create a tuple as the key
    if key not in combinations:
        combinations[key] = []
    combinations[key].append((d, label))

# Debugging: Print combinations
print(f"Number of unique combinations: {len(combinations)}")

# Prepare training data with a maximum of 500 points per combination
train_data = []
train_labels = []

for key, points in combinations.items():
    selected_points = random.sample(points, min(len(points), 500))  # Randomly select up to 500 points
    for point in selected_points:
        train_data.append(point[0])
        train_labels.append(point[1])

# Convert to numpy arrays
train_data = np.array(train_data)
train_labels = np.array(train_labels)

# Debugging: Print the number of training data points
print(f"Number of training data points: {len(train_data)}")

# Remaining data for validation and testing
remaining_data = []
remaining_labels = []

for key, points in combinations.items():
    if len(points) > 500:
        remaining_points = points[500:]  # Get the points beyond the first 500
        for point in remaining_points:
            remaining_data.append(point[0])
            remaining_labels.append(point[1])

# Shuffle remaining data for validation and testing
remaining_indices = np.arange(len(remaining_data))
np.random.shuffle(remaining_indices)
remaining_data = np.array(remaining_data)[remaining_indices]
remaining_labels = np.array(remaining_labels)[remaining_indices]

# Split remaining data into validation and test sets (50/50)
val_size = len(remaining_data) // 2  # Half for validation
val_data = remaining_data[:val_size]
val_labels = remaining_labels[:val_size]
test_data = remaining_data[val_size:]
test_labels = remaining_labels[val_size:]

# Debugging: Print the number of validation and test data points
print(f"Number of validation data points: {len(val_data)}")
print(f"Number of test data points: {len(test_data)}")

# Save the split datasets into separate CSV files
def save_to_csv(file_path, data, labels):
    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for d, l in tqdm(zip(data, labels), total=len(data), desc=f"Saving {file_path}", unit="row"):
            writer.writerow(list(d) + [l])  # Combine data and labels

# Saving the split data to CSV files
save_to_csv(train_file, train_data, train_labels)
save_to_csv(val_file, val_data, val_labels)
save_to_csv(test_file, test_data, test_labels)

print("Data split into training, validation, and test sets successfully!")
