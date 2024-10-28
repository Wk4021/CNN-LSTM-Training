import numpy as np
import pandas as pd
import ast
import os
import threading
import time
import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import load_model

# Set NumPy print options to avoid scientific notation
np.set_printoptions(suppress=True)

# Function to safely parse labels
def safe_parse_label(label):
    try:
        return ast.literal_eval(label)  # Try to parse the label
    except (ValueError, SyntaxError):  # Handle parsing errors
        return [0.0, 0.0, 0.0]  # Or some default value

# Data Augmentation Function
def augment_data(X, y):
    augmented_X = []
    augmented_y = []

    for i in range(len(X)):
        # Original data
        augmented_X.append(X[i])
        augmented_y.append(y[i])

        # Adding noise
        noise = np.random.normal(0, 0.01, X[i].shape)
        augmented_X.append(X[i] + noise)
        augmented_y.append(y[i])

        # Time shift
        shift = np.random.randint(-5, 6)  # Shift between -5 to +5 time steps
        if shift > 0:
            augmented_X.append(np.pad(X[i], ((shift, 0)), mode='constant')[:-shift])
        elif shift < 0:
            augmented_X.append(np.pad(X[i], ((0, -shift)), mode='constant')[-shift:])
        else:
            augmented_X.append(X[i])

        augmented_y.append(y[i])  # Keep labels the same for augmented samples

    return np.array(augmented_X), np.array(augmented_y)

# GUI Application
class TrainingApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Training Application")

        # Variables
        self.num_epochs_var = tk.IntVar(value=100)
        self.batch_size_var = tk.IntVar(value=512)
        self.error_threshold_var = tk.DoubleVar(value=10.0)
        self.learning_rate_var = tk.DoubleVar(value=0.0001)
        self.save_interval_var = tk.IntVar(value=10)
        self.lr_schedule_var = tk.StringVar(value="")
        self.model_type_var = tk.IntVar(value=1)  # Default to CNN
        self.current_epoch = 0
        self.total_epochs = 0
        self.training = False
        self.paused = False
        self.stop_training_flag = False
        self.lr_schedule = {}
        self.training_thread = None

        # Initialize metric lists
        self.loss_values = []
        self.mae_values = []
        self.val_loss_values = []
        self.val_mae_values = []

        # **New Variables for Label Selection**
        self.train_bpm_var = tk.BooleanVar(value=True)
        self.train_rr_var = tk.BooleanVar(value=True)
        self.train_psi_var = tk.BooleanVar(value=True)

        # Load data
        self.load_data()

        # Build GUI
        self.build_gui()

        # Initialize model
        self.model = None

        # Create Models directory if it doesn't exist
        if not os.path.exists('Models'):
            os.makedirs('Models')

    def load_data(self):
        # Load your preprocessed data
        train_data = pd.read_csv('train_data.csv', header=None)
        val_data = pd.read_csv('val_rand.csv', header=None)
        test_data = pd.read_csv('test_rand.csv', header=None)

        # Convert features and labels to numpy arrays
        X_train = train_data.iloc[:, :-1].values  # All columns except the last (features)
        y_train = train_data.iloc[:, -1].apply(safe_parse_label).tolist()  # Parse the last column (labels)
        y_train = np.array(y_train)  # Convert to numpy array

        X_val = val_data.iloc[:, :-1].values
        y_val = val_data.iloc[:, -1].apply(safe_parse_label).tolist()
        y_val = np.array(y_val)  # Convert to numpy array

        X_test = test_data.iloc[:, :-1].values
        y_test = test_data.iloc[:, -1].apply(safe_parse_label).tolist()
        y_test = np.array(y_test)  # Convert to numpy array

        # **Select only the labels chosen by the user**
        selected_indices = self.get_selected_label_indices()

        # Convert features and labels to float32
        X_train = X_train.astype(np.float32)
        y_train = y_train[:, selected_indices].astype(np.float32)
        X_val = X_val.astype(np.float32)
        y_val = y_val[:, selected_indices].astype(np.float32)
        X_test = X_test.astype(np.float32)
        y_test = y_test[:, selected_indices].astype(np.float32)

        # Augment training data
        self.X_train, self.y_train = augment_data(X_train, y_train)
        self.X_val, self.y_val = X_val, y_val
        self.X_test, self.y_test = X_test, y_test

        # Check dimensions after stacking
        print(f"X_train shape: {self.X_train.shape}, y_train shape: {self.y_train.shape}")
        print(f"X_val shape: {self.X_val.shape}, y_val shape: {self.y_val.shape}")
        print(f"X_test shape: {self.X_test.shape}, y_test shape: {self.y_test.shape}")

    # **Function to get selected label indices**
    def get_selected_label_indices(self):
        indices = []
        if self.train_bpm_var.get():
            indices.append(0)
        if self.train_rr_var.get():
            indices.append(1)
        if self.train_psi_var.get():
            indices.append(2)
        return indices

    def build_gui(self):
        # Configure grid layout to make the GUI responsive
        self.master.columnconfigure(0, weight=1)
        self.master.columnconfigure(1, weight=1)
        self.master.columnconfigure(2, weight=1)
        self.master.columnconfigure(3, weight=1)
        self.master.columnconfigure(4, weight=1)
        self.master.columnconfigure(5, weight=1)

        self.master.rowconfigure(9, weight=1)  # For the console output
        self.master.rowconfigure(10, weight=1)  # For the graph

        # Epochs input
        tk.Label(self.master, text="Number of Epochs:").grid(row=0, column=0, sticky='e')
        tk.Entry(self.master, textvariable=self.num_epochs_var).grid(row=0, column=1, sticky='we')

        # Batch size input
        tk.Label(self.master, text="Batch Size:").grid(row=1, column=0, sticky='e')
        tk.Entry(self.master, textvariable=self.batch_size_var).grid(row=1, column=1, sticky='we')

        # Error threshold input
        tk.Label(self.master, text="Error Threshold (%):").grid(row=2, column=0, sticky='e')
        tk.Entry(self.master, textvariable=self.error_threshold_var).grid(row=2, column=1, sticky='we')

        # Learning rate display
        tk.Label(self.master, text="Learning Rate:").grid(row=3, column=0, sticky='e')
        self.learning_rate_label = tk.Label(self.master, textvariable=self.learning_rate_var)
        self.learning_rate_label.grid(row=3, column=1, sticky='w')

        # Save model every N epochs
        tk.Label(self.master, text="Save Model Every N Epochs:").grid(row=4, column=0, sticky='e')
        tk.Entry(self.master, textvariable=self.save_interval_var).grid(row=4, column=1, sticky='we')

        # Learning rate schedule input
        tk.Label(self.master, text="Learning Rate Schedule:").grid(row=5, column=0, sticky='e')
        tk.Entry(self.master, textvariable=self.lr_schedule_var).grid(row=5, column=1, columnspan=3, sticky='we')
        tk.Label(self.master, text="Format: epoch:lr,epoch:lr").grid(row=5, column=4, sticky='w')

        # Model type selection
        tk.Label(self.master, text="Model Type:").grid(row=6, column=0, sticky='e')
        model_options = [("CNN", 1), ("LSTM", 2), ("Linear", 3)]
        for i, (text, value) in enumerate(model_options):
            tk.Radiobutton(self.master, text=text, variable=self.model_type_var, value=value).grid(row=6, column=1 + i, sticky='w')

        # **Add checkboxes for label selection**
        tk.Label(self.master, text="Train For:").grid(row=7, column=0, sticky='e')
        tk.Checkbutton(self.master, text="BPM", variable=self.train_bpm_var).grid(row=7, column=1, sticky='w')
        tk.Checkbutton(self.master, text="RR", variable=self.train_rr_var).grid(row=7, column=2, sticky='w')
        tk.Checkbutton(self.master, text="PSI", variable=self.train_psi_var).grid(row=7, column=3, sticky='w')

        # Time per epoch and total time labels
        self.time_per_epoch_var = tk.StringVar(value="00:00:00")
        self.total_time_var = tk.StringVar(value="00:00:00")
        tk.Label(self.master, text="Time per Epoch:").grid(row=0, column=2, sticky='e')
        tk.Label(self.master, textvariable=self.time_per_epoch_var).grid(row=0, column=3, sticky='w')
        tk.Label(self.master, text="Total Time Remaining:").grid(row=1, column=2, sticky='e')
        tk.Label(self.master, textvariable=self.total_time_var).grid(row=1, column=3, sticky='w')

        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(self.master, variable=self.progress_var, maximum=100)
        self.progress_bar.grid(row=2, column=2, columnspan=2, sticky='we')

        # Buttons
        tk.Button(self.master, text="Start Training", command=self.start_training).grid(row=3, column=2, sticky='we')
        tk.Button(self.master, text="Pause Training", command=self.pause_training).grid(row=3, column=3, sticky='we')
        tk.Button(self.master, text="Stop Training", command=self.stop_training).grid(row=3, column=4, sticky='we')
        tk.Button(self.master, text="Restart", command=self.restart_application).grid(row=3, column=5, sticky='we')

        # Console output
        tk.Label(self.master, text="Console Output:").grid(row=8, column=0, sticky='nw')
        self.console = tk.Text(self.master, height=10)
        self.console.grid(row=9, column=0, columnspan=6, sticky='nsew')

        # Graph
        self.figure = Figure(figsize=(6, 4), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_title("Training Metrics")
        self.ax.set_xlabel("Epoch")
        self.ax.set_ylabel("Value")
        self.line_loss, = self.ax.plot([], [], label='Loss')
        self.line_mae, = self.ax.plot([], [], label='MAE')
        self.line_val_loss, = self.ax.plot([], [], label='Val Loss')
        self.line_val_mae, = self.ax.plot([], [], label='Val MAE')
        self.ax.legend()

        self.canvas = FigureCanvasTkAgg(self.figure, master=self.master)
        self.canvas.get_tk_widget().grid(row=10, column=0, columnspan=6, sticky='nsew')

        # Configure row and column weights for resizing
        for i in range(11):
            self.master.rowconfigure(i, weight=1)
        for i in range(6):
            self.master.columnconfigure(i, weight=1)

    def start_training(self):
        if not self.training:
            self.training = True
            self.stop_training_flag = False
            self.paused = False
            self.total_epochs = self.num_epochs_var.get()

            # Parse learning rate schedule
            self.lr_schedule = {}
            lr_schedule_str = self.lr_schedule_var.get()
            if lr_schedule_str:
                try:
                    for item in lr_schedule_str.split(','):
                        epoch_str, lr_str = item.strip().split(':')
                        epoch = int(epoch_str)
                        lr = float(lr_str)
                        self.lr_schedule[epoch] = lr
                except ValueError:
                    self.print_to_console("Invalid learning rate schedule format.")
                    return

            # **Reload data to apply label selection**
            self.load_data()

            self.training_thread = threading.Thread(target=self.train_model)
            self.training_thread.start()
        elif self.paused:
            self.paused = False
            self.stop_training_flag = False
            self.training_thread = threading.Thread(target=self.train_model)
            self.training_thread.start()

    def pause_training(self):
        if self.training:
            self.paused = True
            self.stop_training_flag = True
            self.training = False

    def stop_training(self):
        if self.training:
            self.stop_training_flag = True
            self.training = False
            self.print_to_console("Stopping training...")
            if self.training_thread is not None:
                self.training_thread.join()
            self.print_to_console("Training stopped.")

    def restart_application(self):
        self.stop_training()
        self.reset_application()

    def reset_application(self):
        # Reset variables
        self.num_epochs_var.set(100)
        self.batch_size_var.set(512)
        self.error_threshold_var.set(10.0)
        self.learning_rate_var.set(0.0001)
        self.save_interval_var.set(10)
        self.lr_schedule_var.set("")
        self.model_type_var.set(1)
        self.current_epoch = 0
        self.total_epochs = 0
        self.training = False
        self.paused = False
        self.stop_training_flag = False
        self.lr_schedule = {}
        self.training_thread = None

        # Reset label selection
        self.train_bpm_var.set(True)
        self.train_rr_var.set(True)
        self.train_psi_var.set(True)

        # Clear metrics
        self.loss_values = []
        self.mae_values = []
        self.val_loss_values = []
        self.val_mae_values = []

        # Reset model
        self.model = None

        # Clear console
        self.console.delete('1.0', tk.END)

        # Reset progress bar
        self.progress_var.set(0)

        # Clear graph
        self.ax.clear()
        self.ax.set_title("Training Metrics")
        self.ax.set_xlabel("Epoch")
        self.ax.set_ylabel("Value")
        self.line_loss, = self.ax.plot([], [], label='Loss')
        self.line_mae, = self.ax.plot([], [], label='MAE')
        self.line_val_loss, = self.ax.plot([], [], label='Val Loss')
        self.line_val_mae, = self.ax.plot([], [], label='Val MAE')
        self.ax.legend()
        self.canvas.draw()

        # Reset time estimates
        self.time_per_epoch_var.set("00:00:00")
        self.total_time_var.set("00:00:00")

        # Update GUI elements
        self.update_learning_rate_display()

        self.print_to_console("Application reset. Ready to start a new training session.")

    def load_model(self):
        model_path = filedialog.askopenfilename(title="Select Model", filetypes=(("H5 files", "*.h5"),))
        if model_path:
            self.model = load_model(model_path)
            self.print_to_console(f"Loaded model from {model_path}")

    def train_model(self):
        self.print_to_console("Training started...")
        batch_size = self.batch_size_var.get()
        epochs = self.num_epochs_var.get()
        error_threshold = self.error_threshold_var.get()

        # Build model if not already loaded
        if self.model is None:
            model_type = self.model_type_var.get()

            # Adjust input shape based on model type
            if model_type == 1 or model_type == 2:
                # CNN or LSTM expects 3D input
                X_train = self.X_train.reshape(self.X_train.shape[0], self.X_train.shape[1], 1)
                X_val = self.X_val.reshape(self.X_val.shape[0], self.X_val.shape[1], 1)
                X_test = self.X_test.reshape(self.X_test.shape[0], self.X_test.shape[1], 1)
            elif model_type == 3:
                # Linear model expects 2D input
                X_train = self.X_train
                X_val = self.X_val
                X_test = self.X_test

            selected_labels = self.get_selected_label_indices()
            output_size = len(selected_labels)

            if model_type == 1:
                # Build CNN model
                self.model = tf.keras.Sequential([
                    layers.Conv1D(64, kernel_size=3, activation='relu',
                                  kernel_regularizer=tf.keras.regularizers.l2(0.001),
                                  input_shape=(X_train.shape[1], 1)),
                    layers.MaxPooling1D(pool_size=2),
                    layers.Conv1D(128, kernel_size=3, activation='relu',
                                  kernel_regularizer=tf.keras.regularizers.l2(0.001)),
                    layers.MaxPooling1D(pool_size=2),
                    layers.Conv1D(256, kernel_size=3, activation='relu',
                                  kernel_regularizer=tf.keras.regularizers.l2(0.001)),
                    layers.MaxPooling1D(pool_size=2),
                    layers.Flatten(),
                    layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
                    layers.Dropout(0.5),
                    layers.Dense(output_size)
                ])
            elif model_type == 2:
                # Build LSTM model
                self.model = tf.keras.Sequential([
                    layers.Bidirectional(layers.LSTM(128, return_sequences=True, input_shape=(X_train.shape[1], 1))),
                    layers.Dropout(0.5),  # Dropout to reduce overfitting
                    layers.Bidirectional(layers.LSTM(64)),  # Bidirectional for capturing context in both directions
                    layers.Dropout(0.5),
                    layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),  # L2 regularization
                    layers.Dense(output_size)  # **Removed 'activation' parameter**
                ])
            elif model_type == 3:
                # Build Linear model
                self.model = tf.keras.Sequential([
                    layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
                    layers.Dense(64, activation='relu'),
                    layers.Dense(output_size)
                ])
        else:
            # Adjust input shape based on model type
            model_type = self.model_type_var.get()
            if model_type == 1 or model_type == 2:
                # CNN or LSTM expects 3D input
                X_train = self.X_train.reshape(self.X_train.shape[0], self.X_train.shape[1], 1)
                X_val = self.X_val.reshape(self.X_val.shape[0], self.X_val.shape[1], 1)
                X_test = self.X_test.reshape(self.X_test.shape[0], self.X_test.shape[1], 1)
            elif model_type == 3:
                # Linear model expects 2D input
                X_train = self.X_train
                X_val = self.X_val
                X_test = self.X_test

        # Compile model
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate_var.get())
        self.model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

        # Custom Callback for GUI updates
        class GUIUpdateCallback(tf.keras.callbacks.Callback):
            def __init__(self, app, lr_schedule):
                super().__init__()
                self.app = app
                self.start_time = time.time()
                self.lr_schedule = lr_schedule
                self.epoch_start_time = None
                self.batch_start_time = None
                self.batch_times = []

            def on_epoch_begin(self, epoch, logs=None):
                if self.app.stop_training_flag:
                    self.model.stop_training = True
                    return

                self.epoch_start_time = time.time()
                self.app.current_epoch = epoch + 1
                self.app.update_progress(0)
                self.batch_times = []
                self.batch_start_time = time.time()

                # Adjust learning rate if in schedule
                if (epoch + 1) in self.lr_schedule:
                    new_lr = self.lr_schedule[epoch + 1]
                    tf.keras.backend.set_value(self.model.optimizer.lr, new_lr)
                    self.app.learning_rate_var.set(new_lr)
                    self.app.update_learning_rate_display()
                    self.app.print_to_console(f"Learning rate adjusted to {new_lr} at epoch {epoch + 1}")

            def on_epoch_end(self, epoch, logs=None):
                if self.app.stop_training_flag:
                    self.model.stop_training = True
                    return

                # Append metrics to the lists
                self.app.loss_values.append(logs.get('loss'))
                self.app.mae_values.append(logs.get('mae'))
                self.app.val_loss_values.append(logs.get('val_loss'))
                self.app.val_mae_values.append(logs.get('val_mae'))

                # Update metrics graph
                epochs = range(1, epoch + 2)
                self.app.update_graph(epochs)

                # Determine which parameters are being trained for
                trained_params = []
                if self.app.train_bpm_var.get():
                    trained_params.append("BPM")
                if self.app.train_rr_var.get():
                    trained_params.append("RR")
                if self.app.train_psi_var.get():
                    trained_params.append("PSI")
                
                # Join them into a string
                trained_params_str = "-".join(trained_params) if trained_params else "None"

                # Save model every N epochs
                if (epoch + 1) % self.app.save_interval_var.get() == 0:
                    # Save model with epoch and trained parameters in the name
                    model_name = f"Models/Model-{epoch + 1}-{trained_params_str}.h5"
                    self.model.save(model_name)
                    self.app.print_to_console(f"Model saved as {model_name}")

                # Print predictions
                self.app.evaluate_and_print_predictions(self.app.X_test)

            def on_train_batch_end(self, batch, logs=None):
                if self.app.stop_training_flag:
                    self.model.stop_training = True
                    return

                # Update progress bar
                progress = (batch + 1) / self.params['steps']
                self.app.update_progress(progress * 100)

                # Record batch time
                batch_time = time.time() - self.batch_start_time
                self.batch_times.append(batch_time)
                self.batch_start_time = time.time()

                # Estimate time per epoch
                avg_batch_time = np.mean(self.batch_times)
                batches_remaining = self.params['steps'] - (batch + 1)
                time_remaining_epoch = avg_batch_time * batches_remaining

                # Update time per epoch
                estimated_epoch_time = (avg_batch_time * self.params['steps'])
                self.app.update_time_per_epoch(estimated_epoch_time)

                # Estimate total time remaining
                epochs_remaining = self.app.total_epochs - self.app.current_epoch
                total_time_remaining = time_remaining_epoch + (epochs_remaining * estimated_epoch_time)
                self.app.update_total_time(total_time_remaining)

        gui_callback = GUIUpdateCallback(self, self.lr_schedule)

        # Train the model
        self.model.fit(X_train, self.y_train,
                       validation_data=(X_val, self.y_val),
                       epochs=epochs,
                       batch_size=batch_size,
                       callbacks=[gui_callback],
                       verbose=0)

        if not self.stop_training_flag:
            # Save final model in Models directory
            final_model_path = "Models/FinalModel.h5"
            self.model.save(final_model_path)
            self.print_to_console(f"Final model saved as {final_model_path}")
            self.print_to_console("Training completed.")
        else:
            self.print_to_console("Training was stopped before completion.")

        self.training = False

    def update_progress(self, value):
        self.progress_var.set(value)
        self.master.update_idletasks()

    def update_graph(self, epochs):
        self.ax.clear()
        self.ax.set_title("Training Metrics")
        self.ax.set_xlabel("Epoch")
        self.ax.set_ylabel("Value")
        self.ax.plot(epochs, self.loss_values, label='Loss')
        self.ax.plot(epochs, self.mae_values, label='MAE')
        self.ax.plot(epochs, self.val_loss_values, label='Val Loss')
        self.ax.plot(epochs, self.val_mae_values, label='Val MAE')
        self.ax.legend()
        self.canvas.draw()

    def update_time_per_epoch(self, epoch_time):
        time_str = time.strftime('%H:%M:%S', time.gmtime(epoch_time))
        self.time_per_epoch_var.set(time_str)
        self.master.update_idletasks()

    def update_total_time(self, total_time):
        time_str = time.strftime('%H:%M:%S', time.gmtime(total_time))
        self.total_time_var.set(time_str)
        self.master.update_idletasks()

    def update_learning_rate_display(self):
        # Force update of the learning rate label
        self.learning_rate_label.config(text=f"{self.learning_rate_var.get():.6f}")
        self.master.update_idletasks()

    def print_to_console(self, message):
        self.console.insert(tk.END, f"{message}\n")
        self.console.see(tk.END)
        self.master.update_idletasks()

    def evaluate_and_print_predictions(self, X_test):
        # Select 5 random test samples
        random_indices = np.random.choice(len(X_test), size=10, replace=False)
        random_test_data = X_test[random_indices]
        random_actual_labels = self.y_test[random_indices]

        # Reshape the test data to be (batch_size, sequence_length, 1) if needed
        if len(random_test_data.shape) == 2:  # (batch_size, sequence_length)
            random_test_data = random_test_data.reshape(random_test_data.shape[0], random_test_data.shape[1], 1)

        # Make predictions
        predictions = self.model.predict(random_test_data)

        # Determine how many target variables we have (BPM, RR, PSI)
        num_targets = predictions.shape[1]  # This tells us whether it's 1, 2, or 3 outputs

        # Calculate percentage errors
        percent_errors_bpm = np.abs((predictions[:, 0] - random_actual_labels[:, 0]) / random_actual_labels[:, 0]) * 100
        percent_errors_rr = percent_errors_psi = None  # Initialize for RR and PSI

        if num_targets >= 2:  # If RR is included
            percent_errors_rr = np.abs((predictions[:, 1] - random_actual_labels[:, 1]) / random_actual_labels[:, 1]) * 100

        if num_targets == 3:  # If PSI is included
            percent_errors_psi = np.abs((predictions[:, 2] - random_actual_labels[:, 2]) / random_actual_labels[:, 2]) * 100

        # Print a header for the current epoch
        self.print_to_console(f"\n--- Predictions at Epoch {self.current_epoch} ---")

        # Print predictions and actual values to the console
        for i in range(10):
            # Prepare the output message
            output_message = (
                f"Epoch {self.current_epoch}: "
                f"Predicted: [BPM: {predictions[i, 0]:.2f}"
            )
            actual_message = f"Actual: [BPM: {random_actual_labels[i, 0]:.2f}"

            # Add RR if available
            if num_targets >= 2:
                output_message += f", RR: {predictions[i, 1]:.2f}"
                actual_message += f", RR: {random_actual_labels[i, 1]:.2f}"

            # Add PSI if available
            if num_targets == 3:
                output_message += f", PSI: {predictions[i, 2]:.2f}"
                actual_message += f", PSI: {random_actual_labels[i, 2]:.2f}"

            # Finalize the message strings
            output_message += "]"
            actual_message += "]"

            # Print the full message with errors, formatted on one line
            error_message = f"%Error: [BPM: {percent_errors_bpm[i]:.2f}%"
            if percent_errors_rr is not None:
                error_message += f", RR: {percent_errors_rr[i]:.2f}%"
            if percent_errors_psi is not None:
                error_message += f", PSI: {percent_errors_psi[i]:.2f}%"
            error_message += "]"  # Close the bracket for errors

            # Print all parts together
            self.print_to_console(f"{output_message}, {actual_message}, {error_message}")



    def calculate_percent_errors(self, X_val):
        # Make predictions on validation data
        predictions = self.model.predict(X_val)
        actual_labels = self.y_val

        # Calculate percentage errors
        percent_errors = np.mean(np.abs((predictions - actual_labels) / actual_labels) * 100, axis=0)

        # Return the average error of the first label (e.g., BPM)
        return percent_errors[0]

if __name__ == "__main__":
    root = tk.Tk()
    app = TrainingApp(root)
    root.mainloop()
