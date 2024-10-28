import pandas as pd
import matplotlib.pyplot as plt
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import matplotlib.ticker as ticker

# Hide the main tkinter window
Tk().withdraw()

# Open a file dialog to select the CSV file
file_path = askopenfilename(title='Select a CSV file', filetypes=[('CSV Files', '*.csv')])

# Check if a file was selected
if not file_path:
    print("No file selected.")
else:
    # Load your data from the selected CSV file
    df = pd.read_csv(file_path)

    # Assuming the first column contains the radar data
    radar_data = df.iloc[:, 0]

    # Count the number of characters in each radar data entry
    char_counts = radar_data.str.len()

    # Count the frequency of each character count
    frequency = char_counts.value_counts().sort_index()

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.bar(frequency.index, frequency.values, color='skyblue')
    plt.title('Frequency of Character Counts in Radar Data')
    plt.xlabel('Number of Characters')
    plt.ylabel('Frequency')
    
    # Set X-axis ticks to be integer values
    plt.gca().xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:0.0f}'))

    plt.xticks(rotation=45)
    plt.grid(axis='y')
    plt.tight_layout()

    # Show the plot
    plt.show()
