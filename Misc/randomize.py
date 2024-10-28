import csv
import random
from tqdm import tqdm

def shuffle_csv(input_file, output_file):
    # Initialize an empty list to store the data
    data = []

    # Read the CSV file into a list with a progress bar
    print("Reading the CSV file...")
    with open(input_file, 'r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        # Count the total number of lines for the progress bar
        total_lines = sum(1 for _ in open(input_file, 'r', encoding='utf-8'))
        f.seek(0)
        for row in tqdm(reader, total=total_lines, desc="Reading rows"):
            data.append(row)

    # Shuffle the data
    print("Shuffling the data...")
    random.shuffle(data)

    # Write the shuffled data to a new CSV file with a progress bar
    print("Writing the shuffled data to the output file...")
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for row in tqdm(data, total=len(data), desc="Writing rows"):
            writer.writerow(row)

    print("Data shuffling complete.")

if __name__ == "__main__":
    input_file = 'val_data.csv'  # Replace with your input file name
    output_file = 'val_rand.csv'  # Replace with your desired output file name

    shuffle_csv(input_file, output_file)
