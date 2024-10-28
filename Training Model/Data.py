import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os

# Create figures for plotting
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

# Function to update the training/validation metrics plot
def update_metrics(frame):
    if os.path.exists('Models/training_metrics.csv'):
        metrics_df = pd.read_csv('Models/training_metrics.csv')

        ax1.cla()  # Clear the current figure

        # Plot Training Loss and Validation Loss
        ax1.plot(metrics_df['epoch'], metrics_df['loss'], label='Training Loss', color='blue')
        ax1.plot(metrics_df['epoch'], metrics_df['val_loss'], label='Validation Loss', color='orange')

        # Plot Training MAE and Validation MAE
        ax1.plot(metrics_df['epoch'], metrics_df['mae'], label='Training MAE', color='green', linestyle='--')
        ax1.plot(metrics_df['epoch'], metrics_df['val_mae'], label='Validation MAE', color='red', linestyle='--')

        # Set plot title and labels
        ax1.set_title('Training and Validation Metrics')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Value')
        ax1.legend()
        ax1.grid()

# Function to update the prediction error plot
def update_predictions(frame):
    if os.path.exists('Models/prediction.csv'):
        prediction_df = pd.read_csv('Models/prediction.csv')

        ax2.cla()  # Clear the current figure

        # Plot the percentage errors for BPM, RR, and PSI
        ax2.scatter(prediction_df['Epoch'], prediction_df['%Error_BPM'], label='% Error BPM', color='blue')
        ax2.scatter(prediction_df['Epoch'], prediction_df['%Error_RR'], label='% Error RR', color='green')
        #ax2.plot(prediction_df['Epoch'], prediction_df['%Error_PSI'], label='% Error PSI', color='red')

        # Set plot title and labels
        ax2.set_title('Prediction Errors')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('% Error')
        ax2.legend()
        ax2.grid()

# Set up the animation to update every 30 seconds
ani_metrics = FuncAnimation(fig, update_metrics, interval=30000)
ani_predictions = FuncAnimation(fig, update_predictions, interval=30000)

# Show the plot
plt.tight_layout()
plt.show()
