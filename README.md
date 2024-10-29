# CSV Data Processing Scripts

This repository contains Python scripts for processing CSV files, specifically designed to split data into chunks, remove empty rows, and verify chunk integrity. Additionally, it includes a CNN + LSTM model for training on heart rate data.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
  - [Split and Insert Empty Rows](#1-split-and-insert-empty-rows)
  - [Verify Chunks](#2-verify-chunks)
  - [Remove Empty Rows](#3-remove-empty-rows)
  - [Train CNN + LSTM Model](#4-train-cnn--lstm-model)
- [Scripts](#scripts)
  - [Split and Insert Empty Rows](#split-and-insert-empty-rows)
  - [Verify Chunks](#verify-chunks)
  - [Remove Empty Rows](#remove-empty-rows)
  - [Train CNN + LSTM Model](#train-cnn--lstm-model)
- [Contributing](#contributing)
- [License](#license)

## Installation

To use these scripts, you'll need Python installed on your machine along with the following libraries:

```bash
pip install numpy pandas tensorflow tqdm scikit-learn
```

## Usage

### 1. Split and Insert Empty Rows

This script splits a given CSV file into chunks of a specified size (default: 16 rows) and inserts an empty row between each chunk.

**Example Usage:**
```python
input_file = "path/to/your/input.csv"
split_and_insert_empty_rows(input_file, chunk_size=16)
```

### 2. Verify Chunks

This script checks the output CSV file to ensure that all chunks are of the specified length (default: 16 rows). It prints any chunks that do not meet this requirement and counts the valid chunks.

**Example Usage:**
```python
output_file = "path/to/your/split_data.csv"
verify_chunks(output_file, chunk_size=16)
```

### 3. Remove Empty Rows

This script removes all empty rows from a specified CSV file and creates a new cleaned version of the file.

**Example Usage:**
```python
input_file = "path/to/your/preprocessed_data_HR.csv"
remove_empty_rows(input_file)
```

### 4. Train CNN + LSTM Model

This script trains a CNN + LSTM model using the processed CSV data. It automatically splits the data into training, validation, and test sets, then trains the model to predict heart rates.

**Example Usage:**
```python
file_path = r'R:\OHG\Cardiac_Respiratory_Phantom\Blood Pressure Processing\Training\Human Training\Data Management\split_data.csv'
X_train, y_train, X_val, y_val, X_test, y_test = load_data(file_path)

model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, batch_size=32)
model.save(r'R:\OHG\Cardiac_Respiratory_Phantom\Blood Pressure Processing\Training\Human Training\Human Models\cnn_lstm_heart_rate_model.h5')
```

## Scripts

### Split and Insert Empty Rows

```python
import os
import csv
from tqdm import tqdm

def split_and_insert_empty_rows(input_file, chunk_size=16):
    ...
```

### Verify Chunks

```python
import csv

def verify_chunks(input_file, chunk_size=16):
    ...
```

### Remove Empty Rows

```python
import os
import csv
from tqdm import tqdm

def remove_empty_rows(input_file):
    ...
```

### Train CNN + LSTM Model

```python
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(file_path, test_size=0.2, val_size=0.2):
    ...
```

## Contributing

Contributions are welcome! If you have suggestions for improvements or new features, please create an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
```

### Key Additions:
- A new section for "Train CNN + LSTM Model" that describes the usage of the new training script.
- A mention of necessary libraries for the training script in the Installation section.
- Example usage of the training script, including how to load data and train the model.
