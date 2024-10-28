# CSV Data Processing Scripts

This repository contains Python scripts for processing CSV files, specifically designed to split data into chunks, remove empty rows, and verify chunk integrity. The scripts are built to streamline data management tasks for CSV files in data analysis workflows.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Scripts](#scripts)
  - [Split and Insert Empty Rows](#split-and-insert-empty-rows)
  - [Verify Chunks](#verify-chunks)
  - [Remove Empty Rows](#remove-empty-rows)
- [Contributing](#contributing)
- [License](#license)

## Installation

To use these scripts, you'll need Python installed on your machine along with the `tqdm` library for progress bars. You can install `tqdm` using pip:

```bash
pip install tqdm
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

## Contributing

Contributions are welcome! If you have suggestions for improvements or new features, please create an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
