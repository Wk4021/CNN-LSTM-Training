import csv
import numpy as np
from tqdm import tqdm  # Import tqdm for progress bar

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
        labels.append(row[-1])  # Last element is the label

# Convert to numpy arrays for easier manipulation
data = np.array(data, dtype=np.uint8)  # Using uint8 since we're dealing with byte-level data
labels = np.array(labels)

# Set random seed for reproducibility
np.random.seed(42)

# Shuffle the data
indices = np.arange(data.shape[0])
np.random.shuffle(indices)

# Shuffle data and labels based on the same indices
data = data[indices]
labels = labels[indices]

# Calculate split indices
train_size = int(0.8 * len(data))
val_size = int(0.1 * len(data))

# Split the data
train_data = data[:train_size]
val_data = data[train_size:train_size + val_size]
test_data = data[train_size + val_size:]

# Split the labels
train_labels = labels[:train_size]
val_labels = labels[train_size:train_size + val_size]
test_labels = labels[train_size + val_size:]

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
