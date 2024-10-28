import socket
import threading
import time
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
import sys
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from tensorflow.keras import layers, models
import tkinter as tk
from tkinter import messagebox
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import pandas as pd
import json

# Server connection details
HOST = '127.0.0.1'  # Replace with your server's IP address
PORT = 65435        # Replace with the port used by your server script for data clients

# Data file
DATA_FILE = 'data.csv'

# Cache variables
raw_data_cache = []
labels_cache = []
cache_threshold = 100  # Cache size before saving to CSV

# Global variables for synchronization
data_lock = threading.Lock()
control_lock = threading.Lock()
training_started = False
training_paused = False
training_stopped = False
model_saved = False

# Global placeholders for training and data
tf_model = None
loss_values = []
current_epoch = 0
data_receiving_active = False
data_receiving_thread = None

# GUI for training and monitoring
class TrainingGUI:
    def __init__(self, master):
        self.master = master
        master.title("Training GUI")

        # Variables to display
        self.data_count = tk.IntVar()
        self.training_status = tk.StringVar()
        self.training_status.set("Idle")
        self.current_epoch = tk.IntVar()
        self.total_data_points = tk.IntVar()
        self.loss_values = []

        # Number of epochs variable
        self.num_epochs_var = tk.StringVar(value='10')  # Default number of epochs
        self.batch_size_var = tk.StringVar(value='32')  # Default batch size

        # Iterations per second
        self.iterations_per_second = tk.StringVar()
        self.iterations_per_second.set("0")

        # Use GPU variable
        self.use_gpu_var = tk.BooleanVar()
        self.use_gpu_var.set(True)  # Default to using GPU

        # Receive Live Data variable
        self.receive_live_data_var = tk.BooleanVar()
        self.receive_live_data_var.set(False)  # Default to not receiving live data

        # Labels
        self.label_data_count = tk.Label(master, text="Data Points Collected:")
        self.label_data_count_value = tk.Label(master, textvariable=self.data_count)
        self.label_training_status = tk.Label(master, text="Training Status:")
        self.label_training_status_value = tk.Label(master, textvariable=self.training_status)
        self.label_current_epoch = tk.Label(master, text="Current Epoch:")
        self.label_current_epoch_value = tk.Label(master, textvariable=self.current_epoch)
        self.label_total_data_points = tk.Label(master, text="Total Data Points:")
        self.label_total_data_points_value = tk.Label(master, textvariable=self.total_data_points)
        self.label_num_epochs = tk.Label(master, text="Number of Epochs:")
        self.entry_num_epochs = tk.Entry(master, textvariable=self.num_epochs_var)
        self.label_batch_size = tk.Label(master, text="Batch Size:")
        self.entry_batch_size = tk.Entry(master, textvariable=self.batch_size_var)
        self.label_it_per_sec = tk.Label(master, text="Iterations per Second:")
        self.label_it_per_sec_value = tk.Label(master, textvariable=self.iterations_per_second)

        # Buttons
        self.button_start = tk.Button(master, text="Start Training", command=self.start_training)
        self.button_pause = tk.Button(master, text="Pause Training", command=self.pause_training)
        self.button_stop_reset = tk.Button(master, text="Stop/Reset Training", command=self.stop_reset_training)
        self.button_save = tk.Button(master, text="Save Model", command=self.save_model)
        self.button_force_save_csv = tk.Button(master, text="Force Save CSV", command=self.force_save_csv)
        self.button_load_csv = tk.Button(master, text="Load CSV", command=self.load_csv)
        self.button_load_model = tk.Button(master, text="Load Model", command=self.load_model)
        self.button_exit = tk.Button(master, text="Exit", command=self.exit_program)
        self.check_use_gpu = tk.Checkbutton(master, text="Use GPU", variable=self.use_gpu_var)

        # Receive Live Data checkbox
        self.check_receive_live_data = tk.Checkbutton(
            master,
            text="Receive Live Data",
            variable=self.receive_live_data_var,
            command=self.toggle_data_receiving
        )

        # Text widget to display training log
        self.text_log = tk.Text(master, height=10, width=80)
        self.scrollbar = tk.Scrollbar(master, command=self.text_log.yview)
        self.text_log.configure(yscrollcommand=self.scrollbar.set)

        # Layout
        self.label_data_count.grid(row=0, column=0, sticky='e')
        self.label_data_count_value.grid(row=0, column=1, sticky='w')
        self.label_training_status.grid(row=1, column=0, sticky='e')
        self.label_training_status_value.grid(row=1, column=1, sticky='w')
        self.label_current_epoch.grid(row=0, column=2, sticky='e')
        self.label_current_epoch_value.grid(row=0, column=3, sticky='w')
        self.label_total_data_points.grid(row=2, column=0, sticky='e')
        self.label_total_data_points_value.grid(row=2, column=1, sticky='w')
        self.label_num_epochs.grid(row=0, column=4, sticky='e')
        self.entry_num_epochs.grid(row=0, column=5, sticky='w')
        self.label_batch_size.grid(row=1, column=4, sticky='e')
        self.entry_batch_size.grid(row=1, column=5, sticky='w')
        self.label_it_per_sec.grid(row=1, column=2, sticky='e')
        self.label_it_per_sec_value.grid(row=1, column=3, sticky='w')

        # Place buttons and checkbox
        self.button_start.grid(row=2, column=2)
        self.button_pause.grid(row=2, column=3)
        self.button_stop_reset.grid(row=2, column=4)
        self.button_save.grid(row=2, column=5)
        self.button_force_save_csv.grid(row=2, column=6)
        self.check_use_gpu.grid(row=3, column=2)
        self.check_receive_live_data.grid(row=3, column=4)
        self.button_load_csv.grid(row=3, column=0)
        self.button_load_model.grid(row=3, column=1)
        self.button_exit.grid(row=3, column=3)

        self.text_log.grid(row=4, column=0, columnspan=7)
        self.scrollbar.grid(row=4, column=7, sticky='ns')

        # Matplotlib Figure
        self.figure = Figure(figsize=(5, 4), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_title("Training Loss")
        self.ax.set_xlabel("Epoch")
        self.ax.set_ylabel("Loss")
        self.line, = self.ax.plot([], [], 'r-')  # Initialize an empty line

        self.canvas = FigureCanvasTkAgg(self.figure, master)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=5, column=0, columnspan=7)

        # Start loss plot updater
        self.update_plot()

    def toggle_data_receiving(self):
        if self.receive_live_data_var.get():
            # Start data receiving
            self.log_message("Starting live data reception.")
            start_data_receiving()
        else:
            # Stop data receiving
            self.log_message("Stopping live data reception.")
            stop_data_receiving()

    def start_training(self):
        global training_started, training_paused, training_stopped
        with control_lock:
            if not training_started:
                training_started = True
                training_paused = False
                training_stopped = False
                self.training_status.set("Training")
                self.log_message("Training started.")
            elif training_paused:
                training_paused = False
                self.training_status.set("Training")
                self.log_message("Training resumed.")
            else:
                self.log_message("Training is already running.")

    def pause_training(self):
        global training_started, training_paused
        with control_lock:
            if training_started and not training_paused:
                training_paused = True
                self.training_status.set("Paused")
                self.log_message("Training paused.")
            else:
                self.log_message("Training is not running.")

    def stop_reset_training(self):
        global training_started, training_paused, training_stopped
        with control_lock:
            if training_started or training_paused:
                training_stopped = True
                training_started = False
                training_paused = False
                self.training_status.set("Stopped")
                self.log_message("Training will be stopped and reset after current epoch.")
            else:
                self.log_message("Training is not running.")

    def save_model(self):
        global training_started, training_paused, model_saved
        with control_lock:
            if (training_started or training_paused) and tf_model is not None:
                model_saved = True
                self.log_message("Model will be saved.")
            else:
                self.log_message("Cannot save model. Training is not active.")

    def force_save_csv(self):
        with data_lock:
            save_data_to_csv()
        self.log_message("Data saved to CSV.")

    def load_csv(self):
        global raw_data_list, labels_list
        from tkinter import filedialog
        csv_path = filedialog.askopenfilename(title="Select CSV File", filetypes=(("CSV files", "*.csv"), ("All files", "*.*")))
        if csv_path:
            try:
                data_df = pd.read_csv(csv_path)
                raw_data_list = [np.frombuffer(bytes.fromhex(s), dtype=np.uint8) for s in data_df['raw_data']]
                labels_list = [eval(s) for s in data_df['labels']]  # Convert string representation back to list
                self.update_data_count(len(raw_data_list))
                self.update_total_data_points(len(raw_data_list))
                self.log_message(f"Loaded {len(raw_data_list)} data samples from {csv_path}.")
            except Exception as e:
                self.log_message(f"Error loading CSV file: {e}")

    def load_model(self):
        global tf_model, current_epoch
        # Open file dialog to select model file
        from tkinter import filedialog
        model_path = filedialog.askopenfilename(title="Select Model File", filetypes=(("H5 files", "*.h5"), ("All files", "*.*")))
        if model_path:
            try:
                # Load the model
                tf_model = tf.keras.models.load_model(model_path)
                self.log_message(f"Loaded model from {model_path}")
                # Try to load the epoch info
                import os
                info_path = os.path.splitext(model_path)[0] + '_info.json'
                if os.path.exists(info_path):
                    with open(info_path, 'r') as f:
                        info = json.load(f)
                        current_epoch = info.get('epoch', 0)
                        self.current_epoch.set(current_epoch)
                        self.log_message(f"Loaded training info: Epoch {current_epoch}")
                else:
                    current_epoch = 0
                    self.current_epoch.set(current_epoch)
                    self.log_message("No training info found. Starting from epoch 0.")
            except Exception as e:
                self.log_message(f"Error loading model: {e}")

    def exit_program(self):
        if messagebox.askokcancel("Exit", "Do you really want to exit?"):
            self.master.quit()
            sys.exit(0)

    def log_message(self, message):
        self.text_log.insert(tk.END, message + '\n')
        self.text_log.see(tk.END)

    def update_data_count(self, count):
        self.data_count.set(count)

    def update_current_epoch(self, epoch):
        self.current_epoch.set(epoch)

    def update_total_data_points(self, total):
        self.total_data_points.set(total)

    def update_loss_values(self, loss_values):
        self.loss_values = loss_values

    def update_plot(self):
        # Update the loss plot
        if self.loss_values:
            epochs = range(1, len(self.loss_values) + 1)
            self.line.set_data(epochs, self.loss_values)
            if len(epochs) > 1:
                self.ax.set_xlim(1, max(epochs))
            else:
                self.ax.set_xlim(0.9, 1.1)
            max_loss = max(self.loss_values)
            if max_loss > 0:
                self.ax.set_ylim(0, max_loss * 1.1)
            else:
                self.ax.set_ylim(0, 1)
            self.canvas.draw()
        # Schedule the next update in 500 milliseconds (0.5 seconds)
        self.master.after(500, self.update_plot)

def toggle_data_receiving():
    if gui.receive_live_data_var.get():
        # Start data receiving
        gui.log_message("Starting live data reception.")
        start_data_receiving()
    else:
        # Stop data receiving
        gui.log_message("Stopping live data reception.")
        stop_data_receiving()

def start_data_receiving():
    global data_receiving_thread, data_receiving_active
    if data_receiving_thread is None or not data_receiving_thread.is_alive():
        data_receiving_active = True
        data_receiving_thread = threading.Thread(target=receive_data, daemon=True)
        data_receiving_thread.start()
    else:
        print("Data receiving is already active.")

def stop_data_receiving():
    global data_receiving_active
    data_receiving_active = False  # This will cause the receive_data loop to exit

def parse_data(line):
    try:
        # Example line format:
        # RTC: 2024-09-27 11:02:23.112 UBPM: 100 URR: 20 PSI: 0.76 Voltage: 0.704 RawRadar: FF1E4E2F...
        # Split the line into key-value pairs
        parts = line.strip().split(' ')
        data_dict = {}
        i = 0
        while i < len(parts):
            if parts[i].endswith(':'):
                key = parts[i][:-1]
                if key == 'RTC':
                    # RTC has a space in the value
                    value = parts[i+1] + ' ' + parts[i+2]
                    i += 3
                else:
                    value = parts[i+1]
                    i += 2
                data_dict[key] = value
            else:
                i += 1

        # Extract required fields
        raw_radar_hex = data_dict.get('RawRadar', None)
        bpm = data_dict.get('UBPM', None)
        rr = data_dict.get('URR', None)
        psi = data_dict.get('PSI', None)

        if raw_radar_hex and bpm and rr and psi:
            # Convert hex string to numpy array
            raw_radar_bytes = bytes.fromhex(raw_radar_hex)
            raw_radar_data = np.frombuffer(raw_radar_bytes, dtype=np.uint8)

            # Convert labels to floats
            bpm = float(bpm)
            rr = float(rr)
            psi = float(psi)

            return raw_radar_data, [bpm, rr, psi]
        else:
            print(f"Missing data fields in line: {line}")
            return None, None
    except Exception as e:
        print(f"Error parsing line: {line}, error: {e}")
        return None, None

def receive_data():
    global raw_data_cache, labels_cache, data_receiving_active
    data_receiving_active = True  # Set the flag to True when starting
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        client_socket.connect((HOST, PORT))
        print(f"Connected to server at {HOST}:{PORT}")
    except Exception as e:
        print(f"Could not connect to server: {e}")
        return  # Exit the function

    buffer = ''
    with client_socket:
        while data_receiving_active:
            data = client_socket.recv(4096)
            if not data:
                print("No data received. Connection might be closed.")
                break
            buffer += data.decode('utf-8')
            while '\n' in buffer:
                line, buffer = buffer.split('\n', 1)
                line = line.strip()
                if not line:
                    continue
                raw_radar_data, labels = parse_data(line)
                if raw_radar_data is not None:
                    with data_lock:
                        raw_data_cache.append(raw_radar_data)
                        labels_cache.append(labels)
                        # Check if cache size exceeds threshold
                        if len(raw_data_cache) >= cache_threshold:
                            save_data_to_csv()
                    data_length = len(raw_data_cache)
                    print(f"Received data sample #{data_length}")
                    # Update data count in GUI
                    root.after(0, gui.update_data_count, data_length)
                    root.after(0, gui.update_total_data_points, data_length)
                else:
                    print("Invalid data received, skipping.")
    print("Data receiving stopped.")

def save_data_to_csv():
    global raw_data_cache, labels_cache
    if not raw_data_cache:
        print("No new data to save.")
        return

    # Convert numpy arrays to bytes before calling hex()
    data_dict = {
        'raw_data': [data.tobytes().hex() for data in raw_data_cache],
        'labels': [str(label) for label in labels_cache]
    }
    data_df = pd.DataFrame(data_dict)

    # Append new data to the existing CSV file
    if not os.path.exists(DATA_FILE):
        data_df.to_csv(DATA_FILE, index=False)
    else:
        data_df.to_csv(DATA_FILE, mode='a', header=False, index=False)

    # Clear the cache after saving
    raw_data_cache = []
    labels_cache = []
    print(f"Saved data samples to {DATA_FILE}.")

def preprocess_data(raw_data_list):
    # Find the maximum length
    max_length = max(len(data) for data in raw_data_list)
    # Pad or truncate data to have the same length
    processed_data = []
    for data in raw_data_list:
        if len(data) < max_length:
            # Pad with zeros
            padded_data = np.pad(data, (0, max_length - len(data)), 'constant')
        else:
            # Truncate data
            padded_data = data[:max_length]
        processed_data.append(padded_data)
    # Convert to numpy array
    processed_data = np.array(processed_data)
    # Normalize data
    processed_data = processed_data / 255.0  # Scale to [0,1]
    return processed_data

def normalize_targets(y):
    y_normalized = y.copy()
    y_normalized[:, 0] = y[:, 0] / 200.0  # Normalize BPM (assuming max 200)
    y_normalized[:, 1] = y[:, 1] / 50.0   # Normalize RR (assuming max 50)
    y_normalized[:, 2] = y[:, 2]          # PSI assumed to be between 0 and 1
    return y_normalized

def create_tensorflow_model(input_shape):
    model = models.Sequential()
    model.add(layers.InputLayer(input_shape=(input_shape,)))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(3))  # Output layer for BPM, RR, PSI
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss='mse')
    return model

def train_model():
    global raw_data_list, labels_list
    global training_started, training_paused, training_stopped, model_saved
    global tf_model
    global loss_values, current_epoch
    current_epoch = current_epoch  # Ensure it's initialized
    while True:
        with control_lock:
            ts = training_started
            tp = training_paused
            ms = model_saved
            stop_flag = training_stopped
        if ts:
            with data_lock:
                data_length = len(raw_data_list)
            if data_length >= 100:
                root.after(0, gui.log_message, "Starting training...")
                with data_lock:
                    # Preprocess data
                    X = preprocess_data(raw_data_list)
                    y = np.array(labels_list)
                # Normalize targets
                y = normalize_targets(y)
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
                # Training parameters
                try:
                    num_epochs = int(gui.num_epochs_var.get())
                except ValueError:
                    num_epochs = 10  # Default to 10 if invalid input
                try:
                    batch_size = int(gui.batch_size_var.get())
                except ValueError:
                    batch_size = 32  # Default batch size
                loss_values = []
                # TensorFlow model training
                if tf_model is None:
                    tf_model = create_tensorflow_model(X_train.shape[1])

                # Check for GPU availability and set device
                gpus = tf.config.list_physical_devices('GPU')
                if gui.use_gpu_var.get() and gpus:
                    device = '/GPU:0'
                    root.after(0, gui.log_message, "Using GPU for training.")
                else:
                    device = '/CPU:0'
                    if gui.use_gpu_var.get():
                        root.after(0, gui.log_message, "GPU selected but not available. Using CPU instead.")
                    else:
                        root.after(0, gui.log_message, "Using CPU for training.")

                # Define a custom callback to update GUI and collect loss
                class GUIUpdater(tf.keras.callbacks.Callback):
                    def on_epoch_begin(self, epoch, logs=None):
                        self.epoch_start_time = time.time()
                        self.total_batches = 0

                    def on_train_batch_end(self, batch, logs=None):
                        self.total_batches += 1

                    def on_epoch_end(self, epoch, logs=None):
                        global training_paused, model_saved, current_epoch, training_stopped  # Declare globals
                        epoch_end_time = time.time()
                        epoch_duration = epoch_end_time - self.epoch_start_time
                        iterations_per_second = self.total_batches / epoch_duration if epoch_duration > 0 else 0
                        root.after(0, gui.iterations_per_second.set, f"{iterations_per_second:.2f}")

                        loss = logs.get('loss')
                        val_loss = logs.get('val_loss')
                        loss_values.append(loss)
                        current_epoch = epoch + 1
                        root.after(0, gui.update_current_epoch, current_epoch)
                        root.after(0, gui.log_message, f"Epoch {current_epoch}, Loss: {loss:.4f}, Val Loss: {val_loss:.4f}")
                        root.after(0, gui.update_loss_values, loss_values)
                        # Check for pause or save request
                        with control_lock:
                            if training_paused:
                                self.model.stop_training = True
                            if model_saved:
                                # Save model
                                self.model.save('tf_model.h5')
                                # Save epoch info
                                info = {'epoch': current_epoch}
                                with open('tf_model_info.json', 'w') as f:
                                    json.dump(info, f)
                                model_saved = False
                                root.after(0, gui.log_message, "Model saved.")
                            if training_stopped:
                                self.model.stop_training = True

                gui_updater = GUIUpdater()

                # Add EarlyStopping and ReduceLROnPlateau callbacks
                early_stopping = tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=20,
                    min_delta=0.0001,
                    restore_best_weights=True
                )
                reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=10,
                    min_delta=0.0001
                )

                # Start training
                try:
                    with tf.device(device):
                        tf_model.fit(
                            X_train, y_train,
                            epochs=current_epoch + num_epochs,
                            initial_epoch=current_epoch,
                            batch_size=batch_size,
                            validation_data=(X_test, y_test),
                            callbacks=[gui_updater, early_stopping, reduce_lr],
                            verbose=0
                        )
                    # After training
                    root.after(0, gui.log_message, "Training completed.")
                    # Evaluate on test data
                    test_loss = tf_model.evaluate(X_test, y_test, verbose=0)
                    root.after(0, gui.log_message, f"Test Loss: {test_loss:.4f}")
                except Exception as e:
                    root.after(0, gui.log_message, f"An error occurred during training: {e}")
                finally:
                    # Update current_epoch
                    current_epoch = current_epoch + num_epochs
                    with control_lock:
                        training_started = False
                        training_paused = False
                        if training_stopped:
                            # If training is stopped, reset training variables
                            root.after(0, gui.log_message, "Training has been stopped.")
                            training_stopped = False  # Reset the flag
                            tf_model = None  # Reset the model
                            loss_values = []
                            current_epoch = 0
                            root.after(0, gui.update_current_epoch, current_epoch)
                            root.after(0, gui.update_loss_values, loss_values)
                            root.after(0, gui.iterations_per_second.set, "0")
            else:
                root.after(0, gui.log_message, f"Waiting for more data... Collected {data_length} samples so far.")
                time.sleep(5)  # Check every 5 seconds
        else:
            time.sleep(1)

if __name__ == "__main__":
    # Load existing data from CSV file
    try:
        data_df = pd.read_csv(DATA_FILE)
        # Convert hex strings back to numpy arrays
        raw_data_list = [np.frombuffer(bytes.fromhex(s), dtype=np.uint8) for s in data_df['raw_data']]
        labels_list = [eval(s) for s in data_df['labels']]  # Convert string representation back to list
        print(f"Loaded {len(raw_data_list)} data samples from {DATA_FILE}.")
    except FileNotFoundError:
        raw_data_list = []
        labels_list = []
        print("No existing data file found. You can load one using the 'Load CSV' button.")
    except Exception as e:
        print(f"Error loading data from {DATA_FILE}: {e}")
        raw_data_list = []
        labels_list = []
        print("You can load a CSV file using the 'Load CSV' button.")

    # Check for GPU availability
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"Available GPUs: {gpus}")
    else:
        print("No GPU available. Training will be performed on CPU.")

    # Initialize Tkinter
    root = tk.Tk()
    gui = TrainingGUI(root)
    data_length = len(raw_data_list)
    gui.update_data_count(data_length)
    gui.update_total_data_points(data_length)

    # Start training thread
    training_thread = threading.Thread(target=train_model, daemon=True)
    training_thread.start()

    # Start the Tkinter main loop
    root.mainloop()
