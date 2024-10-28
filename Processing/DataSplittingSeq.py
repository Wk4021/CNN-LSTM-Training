import pandas as pd
from sklearn.model_selection import train_test_split

# Specify the path to your preprocessed data file
preprocessed_file = 'preprocessed_data.csv'

# Load the preprocessed data
data = pd.read_csv(preprocessed_file, header=None)

# Check the shape of the data
print(f"Data shape: {data.shape}")

# Ensure that the data has at least 2 columns
if data.shape[1] < 2:
    raise ValueError("Insufficient columns in the data. Ensure it has both features and labels.")

# Separate the features and labels
features = data.iloc[:, 0]  # First column (radar data)
labels = data.iloc[:, 1].apply(lambda x: eval(x))  # Convert string representation of lists to actual lists

# Split the labels into separate columns for BPM, RR, PSI
labels_df = pd.DataFrame(labels.tolist(), columns=['BPM', 'RR', 'PSI'])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels_df, test_size=0.2, random_state=42)

# Print the shapes of the resulting datasets
print(f"Training features shape: {X_train.shape}")
print(f"Testing features shape: {X_test.shape}")
print(f"Training labels shape: {y_train.shape}")
print(f"Testing labels shape: {y_test.shape}")

# Save the split data if needed (optional)
# X_train.to_csv('X_train.csv', index=False)
# y_train.to_csv('y_train.csv', index=False)
# X_test.to_csv('X_test.csv', index=False)
# y_test.to_csv('y_test.csv', index=False)
