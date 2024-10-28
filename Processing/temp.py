import csv

def count_values_in_csv(file_path):
    try:
        with open(file_path, mode='r', newline='', encoding='utf-8') as csvfile:
            # Use the csv reader to read lines
            reader = csv.reader(csvfile)
            for line_number, row in enumerate(reader, start=1):
                value_count = len(row)  # Count the number of values in the row
                print(f'Line {line_number}: {value_count} values')
    except FileNotFoundError:
        print(f'Error: The file {file_path} was not found.')
    except Exception as e:
        print(f'An error occurred: {e}')

if __name__ == '__main__':
    count_values_in_csv('preprocessed_data.csv')
