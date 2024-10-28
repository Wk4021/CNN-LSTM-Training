import socket
import threading
import time
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
import sys

# For TensorFlow
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.mixed_precision import experimental as mixed_precision

# For GUI
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk  # For dropdown menu

# For plotting
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# For data saving/loading
import pandas as pd
import json

# Server connection details
HOST = '127.0.0.1'  # Replace with your server's IP address
PORT = 65435        # Replace with the port used by your server script for data clients

# Data file
DATA_FILE = 'data.csv'

# Global variables for data storage
raw_data_cache = []
labels_cache = []
cache_threshold = 100  # Default cache threshold
auto_save_wait_time = 60  # Default auto-save wait time in seconds
last_data_time = time.time()

# Synchronization locks
data_lock = threading.Lock()
control_lock = threading.Lock()

# Control flags
training_started = False
training_paused = False  # Flag to pause training
training_stopped = False  # Flag to stop and reset training
model_saved = False

# Placeholder for model
tf_model = None

# Global variables for training
loss_values = []
val_loss_values = []
current_epoch = 0

# Global variables for data reception
data_receiving_thread = None
data_receiving_active = False

class TrainingGUI:
    def __init__(self, master):
        self.master = master
        master.title("Training GUI")

        # Variable for elapsed time
        self.elapsed_time = tk.StringVar()
        self.elapsed_time.set("00:00:00")

        # Label for elapsed time
        self.label_elapsed_time = tk.Label(master, text="Elapsed Time:")
        self.label_elapsed_time_value = tk.Label(master, textvariable=self.elapsed_time)

        # Add the labels to the GUI layout
        self.label_elapsed_time.grid(row=2, column=6, sticky='e')
        self.label_elapsed_time_value.grid(row=2, column=7, sticky='w')


        # Variables to display
        self.data_count = tk.IntVar()
        self.training_status = tk.StringVar()
        self.training_status.set("Idle")
        self.current_epoch = tk.IntVar()
        self.total_data_points = tk.IntVar()
        self.loss_values = []
        self.val_loss_values = []

        # Number of epochs variable
        self.num_epochs_var = tk.StringVar(value='10')  # Default number of epochs

        # Batch size variable
        self.batch_size_var = tk.StringVar(value='32')  # Default batch size

        # Cache threshold variable
        self.cache_threshold_var = tk.StringVar(value='100')  # Default cache threshold

        # Auto-save wait time variable
        self.auto_save_wait_time_var = tk.StringVar(value='60')  # Default auto-save wait time in seconds

        # Sequence length for LSTM
        self.sequence_length_var = tk.StringVar(value='10')  # Default sequence length

        # Precision selection variable
        self.precision_var = tk.StringVar()
        self.precision_var.set('FP32')  # Default to FP32

        # Iterations per second
        self.iterations_per_second = tk.StringVar()
        self.iterations_per_second.set("0")

        # Current step and total steps
        self.current_step = tk.IntVar()
        self.total_steps = tk.IntVar()

        # Use GPU variable
        self.use_gpu_var = tk.BooleanVar()
        self.use_gpu_var.set(True)  # Default to using GPU

        # Receive Live Data variable
        self.receive_live_data_var = tk.BooleanVar()
        self.receive_live_data_var.set(False)  # Default to not receiving live data

        # Training mode selection variable
        self.training_mode_var = tk.StringVar()
        self.training_mode_var.set('LSTM')  # Default to LSTM

        # Variables for estimated times
        self.epoch_time_remaining = tk.StringVar()
        self.total_time_remaining = tk.StringVar()

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
        self.label_current_step = tk.Label(master, text="Current Step:")
        self.label_current_step_value = tk.Label(master, textvariable=self.current_step)
        self.label_total_steps = tk.Label(master, text="Total Steps:")
        self.label_total_steps_value = tk.Label(master, textvariable=self.total_steps)
        self.label_cache_threshold = tk.Label(master, text="Cache Threshold:")
        self.entry_cache_threshold = tk.Entry(master, textvariable=self.cache_threshold_var)
        self.label_auto_save_wait = tk.Label(master, text="Auto-Save Wait Time (s):")
        self.entry_auto_save_wait = tk.Entry(master, textvariable=self.auto_save_wait_time_var)
        self.label_sequence_length = tk.Label(master, text="Sequence Length:")
        self.entry_sequence_length = tk.Entry(master, textvariable=self.sequence_length_var)
        self.label_precision = tk.Label(master, text="Precision:")

        # Training mode dropdown
        self.label_training_mode = tk.Label(master, text="Training Mode:")
        self.training_mode_dropdown = ttk.Combobox(
            master, textvariable=self.training_mode_var
        )
        self.training_mode_dropdown['values'] = (
            'LSTM', 'CNN', 'FeedForwardNN', 'MLP', 'Regression', 'TreeBased'
        )
        self.training_mode_dropdown.current(0)  # Set default to LSTM

        # Precision selection dropdown
        self.precision_dropdown = ttk.Combobox(master, textvariable=self.precision_var)
        self.precision_dropdown['values'] = ('FP32', 'FP16')
        self.precision_dropdown.current(0)  # Set default to FP32
        self.precision_dropdown.bind('<<ComboboxSelected>>', self.on_precision_change)

        # Buttons
        self.button_start = tk.Button(master, text="Start Training", command=self.start_training)
        self.button_pause = tk.Button(master, text="Pause Training", command=self.pause_training)
        self.button_restart = tk.Button(master, text="Restart Training", command=self.restart_training)
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

        # Variables for estimated times
        self.epoch_time_remaining = tk.StringVar()
        self.total_time_remaining = tk.StringVar()

        # Labels for estimated times
        self.label_epoch_time_remaining = tk.Label(master, text="Epoch Time Remaining:")
        self.label_epoch_time_remaining_value = tk.Label(master, textvariable=self.epoch_time_remaining)
        self.label_total_time_remaining = tk.Label(master, text="Total Time Remaining:")
        self.label_total_time_remaining_value = tk.Label(master, textvariable=self.total_time_remaining)

        # Text widget to display training log
        self.text_log = tk.Text(master, height=15, width=80)
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
        self.label_current_step.grid(row=0, column=6, sticky='e')
        self.label_current_step_value.grid(row=0, column=7, sticky='w')
        self.label_total_steps.grid(row=1, column=6, sticky='e')
        self.label_total_steps_value.grid(row=1, column=7, sticky='w')
        self.label_cache_threshold.grid(row=3, column=0, sticky='e')
        self.entry_cache_threshold.grid(row=3, column=1, sticky='w')
        self.label_auto_save_wait.grid(row=3, column=2, sticky='e')
        self.entry_auto_save_wait.grid(row=3, column=3, sticky='w')
        self.label_sequence_length.grid(row=3, column=4, sticky='e')
        self.entry_sequence_length.grid(row=3, column=5, sticky='w')
        self.label_precision.grid(row=3, column=6, sticky='e')
        self.precision_dropdown.grid(row=3, column=7, sticky='w')
        self.label_epoch_time_remaining.grid(row=2, column=2, sticky='e')
        self.label_epoch_time_remaining_value.grid(row=2, column=3, sticky='w')
        self.label_total_time_remaining.grid(row=2, column=4, sticky='e')
        self.label_total_time_remaining_value.grid(row=2, column=5, sticky='w')

        # Place buttons and checkbox
        self.label_training_mode.grid(row=4, column=0, sticky='e')
        self.training_mode_dropdown.grid(row=4, column=1, sticky='w')
        self.button_start.grid(row=5, column=0)
        self.button_pause.grid(row=5, column=1)
        self.button_restart.grid(row=5, column=2)
        self.button_save.grid(row=5, column=3)
        self.button_force_save_csv.grid(row=5, column=4)
        self.button_load_csv.grid(row=5, column=5)
        self.button_load_model.grid(row=5, column=6)
        self.button_exit.grid(row=5, column=7)
        self.check_use_gpu.grid(row=6, column=2)
        self.check_receive_live_data.grid(row=6, column=3)

        self.text_log.grid(row=7, column=0, columnspan=8)
        self.scrollbar.grid(row=7, column=8, sticky='ns')

        # Matplotlib Figure
        self.figure = Figure(figsize=(5, 4), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_title("Training and Validation Loss")
        self.ax.set_xlabel("Epoch")
        self.ax.set_ylabel("Loss")
        self.line_train, = self.ax.plot([], [], 'r-', label='Training Loss')
        self.line_val, = self.ax.plot([], [], 'b-', label='Validation Loss')
        self.ax.legend()

        self.canvas = FigureCanvasTkAgg(self.figure, master)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=8, column=0, columnspan=8)

        # Start loss plot updater
        self.update_plot()

        # Log initial settings
        self.log_initial_settings()

    def log_initial_settings(self):
        self.log_message("Initial Settings:")
        self.log_message(f"Precision: {self.precision_var.get()}")
        self.log_message(f"Batch Size: {self.batch_size_var.get()}")
        self.log_message(f"Sequence Length: {self.sequence_length_var.get()}")
        self.log_message(f"Training Mode: {self.training_mode_var.get()}")

    def on_precision_change(self, event):
        precision = self.precision_var.get()
        self.log_message(f"Precision changed to: {precision}")
        # Adjust the TensorFlow policy if needed
        if precision == 'FP16':
            policy = mixed_precision.Policy('mixed_float16')
            mixed_precision.set_policy(policy)
        else:
            policy = mixed_precision.Policy('float32')
            mixed_precision.set_policy(policy)

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
        global training_started, training_stopped, training_paused
        with control_lock:
            if not training_started:
                training_started = True
                training_stopped = False
                training_paused = False
                self.training_status.set("Training")
                self.log_message("Training started.")
                self.log_current_settings()
            elif training_paused:
                training_paused = False
                self.training_status.set("Training")
                self.log_message("Training resumed.")
            else:
                self.log_message("Training is already running.")

    def pause_training(self):
        global training_paused
        with control_lock:
            if training_started and not training_paused:
                training_paused = True
                self.training_status.set("Paused")
                self.log_message("Training paused.")
            else:
                self.log_message("Training is not running or already paused.")

    def restart_training(self):
        global training_started, training_stopped, current_epoch, tf_model, loss_values, val_loss_values
        with control_lock:
            training_stopped = True
            training_started = False
            training_paused = False
            current_epoch = 0
            loss_values = []
            val_loss_values = []
            self.current_epoch.set(current_epoch)
            self.training_status.set("Restarted")
            self.log_message("Training has been restarted.")
            
            # Clear the previous model
            tf_model = None
            # Clear the TensorFlow graph and session
            tf.keras.backend.clear_session()

    def save_model(self):
        global tf_model, current_epoch
        if tf_model is not None:
            training_mode = self.training_mode_var.get()
            model_name = f'{training_mode}-Epoch{current_epoch}.h5'
            tf_model.save(model_name)
            # Save epoch info
            info = {'epoch': current_epoch}
            info_filename = f'{training_mode}-Epoch{current_epoch}_info.json'
            with open(info_filename, 'w') as f:
                json.dump(info, f)
            self.log_message(f"Model saved as {model_name}.")
        else:
            self.log_message("No model to save.")

    def force_save_csv(self):
        with data_lock:
            save_data_to_csv()
        self.log_message("Data saved to CSV.")

    def load_csv(self):
        global raw_data_cache, labels_cache
        from tkinter import filedialog
        csv_path = filedialog.askopenfilename(title="Select CSV File", filetypes=(("CSV files", "*.csv"), ("All files", "*.*")))
        if csv_path:
            try:
                data_df = pd.read_csv(csv_path)
                raw_data_cache = [np.frombuffer(bytes.fromhex(s), dtype=np.uint8) for s in data_df['raw_data']]
                labels_cache = [eval(s) for s in data_df['labels']]  # Convert string representation back to list
                self.update_data_count(len(raw_data_cache))
                self.update_total_data_points(len(data_df))
                self.log_message(f"Loaded {len(raw_data_cache)} data samples from {csv_path}.")
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
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.text_log.insert(tk.END, f"{timestamp}: {message}\n")
        self.text_log.see(tk.END)

    def log_current_settings(self):
        self.log_message("Current Training Settings:")
        self.log_message(f"Precision: {self.precision_var.get()}")
        self.log_message(f"Batch Size: {self.batch_size_var.get()}")
        self.log_message(f"Sequence Length: {self.sequence_length_var.get()}")
        self.log_message(f"Number of Epochs: {self.num_epochs_var.get()}")
        self.log_message(f"Training Mode: {self.training_mode_var.get()}")

    def update_data_count(self, count):
        self.data_count.set(count)

    def update_current_epoch(self, epoch):
        self.current_epoch.set(epoch)

    def update_total_data_points(self, total):
        self.total_data_points.set(total)

    def update_loss_values(self, loss_values, val_loss_values):
        self.loss_values = loss_values
        self.val_loss_values = val_loss_values

    def update_iterations_per_second(self, it_per_sec):
        self.iterations_per_second.set(f"{it_per_sec:.2f}")

    def update_current_step(self, current_step, total_steps):
        self.current_step.set(current_step)
        self.total_steps.set(total_steps)

    def update_epoch_time_remaining(self, seconds):
        time_str = time.strftime('%H:%M:%S', time.gmtime(seconds))
        self.epoch_time_remaining.set(time_str)

    def update_total_time_remaining(self, seconds):
        time_str = time.strftime('%H:%M:%S', time.gmtime(seconds))
        self.total_time_remaining.set(time_str)

    def update_window_title(self, elapsed_time, total_time_remaining):
        elapsed_str = time.strftime('%H:%M:%S', time.gmtime(elapsed_time))
        remaining_str = time.strftime('%H:%M:%S', time.gmtime(total_time_remaining))
        self.master.title(f"Training GUI - Elapsed: {elapsed_str} | Remaining: {remaining_str}")

    def update_plot(self):
        # Update the loss plot
        if self.loss_values:
            epochs = range(1, len(self.loss_values) + 1)
            self.line_train.set_data(epochs, self.loss_values)
            self.line_val.set_data(epochs, self.val_loss_values)
            self.ax.set_xlim(1, max(epochs))
            max_loss = max(max(self.loss_values), max(self.val_loss_values))
            if max_loss > 0:
                self.ax.set_ylim(0, max_loss * 1.1)
            else:
                self.ax.set_ylim(0, 1)
            self.canvas.draw()
        # Schedule the next update in 500 milliseconds (0.5 seconds)
        self.master.after(500, self.update_plot)
    
    def update_elapsed_time(self, seconds):
        time_str = time.strftime('%H:%M:%S', time.gmtime(seconds))
        self.elapsed_time.set(time_str)


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
            gui.log_message(f"Missing data fields in line: {line}")
            return None, None
    except Exception as e:
        print(f"Error parsing line: {line}, error: {e}")
        gui.log_message(f"Error parsing line: {line}, error: {e}")
        return None, None

def receive_data():
    global raw_data_cache, labels_cache, data_receiving_active, cache_threshold, last_data_time
    data_receiving_active = True  # Set the flag to True when starting
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        client_socket.connect((HOST, PORT))
        print(f"Connected to server at {HOST}:{PORT}")
        gui.log_message(f"Connected to server at {HOST}:{PORT}")
    except Exception as e:
        print(f"Could not connect to server: {e}")
        gui.log_message(f"Could not connect to server: {e}")
        return  # Exit the function

    buffer = ''
    with client_socket:
        while data_receiving_active:
            data = client_socket.recv(4096)
            if not data:
                print("No data received. Connection might be closed.")
                gui.log_message("No data received. Connection might be closed.")
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
                        # Update last_data_time
                        last_data_time = time.time()
                        # Update data count in GUI
                        data_length = len(raw_data_cache)
                        root.after(0, gui.update_data_count, data_length)
                        # Check if cache size exceeds threshold
                        cache_threshold = int(gui.cache_threshold_var.get())
                        if len(raw_data_cache) >= cache_threshold:
                            save_data_to_csv()
                    # Update total data points in GUI
                    total_data_points = len(raw_data_cache)
                    root.after(0, gui.update_total_data_points, total_data_points)
                    gui.log_message(f"Received data: {len(raw_radar_data)} bytes")
                else:
                    print("Invalid data received, skipping.")
                    gui.log_message("Invalid data received, skipping.")

            # Auto-save if no data received within auto_save_wait_time
            auto_save_wait_time = int(gui.auto_save_wait_time_var.get())
            if time.time() - last_data_time >= auto_save_wait_time:
                with data_lock:
                    if raw_data_cache:
                        save_data_to_csv()
                        last_data_time = time.time()
        print("Data receiving stopped.")
        gui.log_message("Data receiving stopped.")

def save_data_to_csv():
    global raw_data_cache, labels_cache
    if not raw_data_cache:
        print("No new data to save.")
        gui.log_message("No new data to save.")
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
    raw_data_cache.clear()
    labels_cache.clear()
    print(f"Saved data samples to {DATA_FILE}.")
    gui.log_message(f"Saved data samples to {DATA_FILE}.")

def preprocess_data(raw_data_list, labels_list, sequence_length, data_precision):
    # Limit max_length to prevent excessive memory usage
    max_length = max(len(data) for data in raw_data_list)
    max_length = min(max_length, 2060)  # Adjust based on your data
    gui.log_message(f"Max data length set to: {max_length}")

    # Pad or truncate data to have the same length
    processed_data = []
    for data in raw_data_list:
        if len(data) < max_length:
            padded_data = np.pad(data, (0, max_length - len(data)), 'constant')
        else:
            padded_data = data[:max_length]
        # Normalize and convert to the selected precision
        if data_precision == 'FP16':
            padded_data = padded_data.astype(np.float16) / 255.0
        else:
            padded_data = padded_data.astype(np.float32) / 255.0
        processed_data.append(padded_data)
    processed_data = np.array(processed_data)

    # Convert labels to numpy array
    labels_array = np.array(labels_list, dtype=np.float32)
    labels_array = normalize_targets(labels_array)

    # Create sequences
    sequences = []
    sequence_labels = []
    for i in range(len(processed_data) - sequence_length + 1):
        seq = processed_data[i:i+sequence_length]
        sequences.append(seq)
        sequence_labels.append(labels_array[i + sequence_length - 1])

    sequences = np.array(sequences)
    sequence_labels = np.array(sequence_labels)
    gui.log_message(f"Sequences shape: {sequences.shape}")
    gui.log_message(f"Labels shape: {sequence_labels.shape}")

    return sequences, sequence_labels

def normalize_targets(y):
    y_normalized = y.copy()
    y_normalized[:, 0] = y[:, 0] / 200.0  # Normalize BPM (assuming max 200)
    y_normalized[:, 1] = y[:, 1] / 50.0   # Normalize RR (assuming max 50)
    y_normalized[:, 2] = y[:, 2]          # PSI assumed to be between 0 and 1
    return y_normalized

def create_tensorflow_model(input_shape, data_precision, training_mode):
    from tensorflow.keras import mixed_precision
    mixed_precision.set_global_policy(None)
    # Set the global mixed precision policy
    if data_precision == 'FP16':
        mixed_precision.set_global_policy('mixed_float16')
    else:
        mixed_precision.set_global_policy('float32')

    model = models.Sequential()

    if training_mode == 'LSTM':
        model.add(layers.LSTM(64, input_shape=input_shape))
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dense(3))  # Ensure the output layer has 3 units
    elif training_mode == 'CNN':
        model.add(layers.Conv1D(64, kernel_size=3, activation='relu', input_shape=input_shape))
        model.add(layers.GlobalAveragePooling1D())
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dense(3))  # Output layer with 3 units
    elif training_mode == 'FeedForwardNN':
        model.add(layers.Flatten(input_shape=input_shape))
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(3))  # Output layer with 3 units
    elif training_mode == 'MLP':
        model.add(layers.Flatten(input_shape=input_shape))
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(3))  # Output layer with 3 units
    else:
        # Default to LSTM
        model.add(layers.LSTM(64, input_shape=input_shape))
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dense(3))  # Output layer with 3 units

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss='mse')

    return model


def train_model():
    global training_started, training_stopped, training_paused, tf_model, loss_values, val_loss_values, current_epoch
    current_epoch = 0  # Reset epoch count
    while True:
        with control_lock:
            ts = training_started
            tp = training_paused
            stop_flag = training_stopped
        if ts:
            if not tp:
                # Load data using tf.data.Dataset
                try:
                    sequence_length = int(gui.sequence_length_var.get())
                    batch_size = int(gui.batch_size_var.get())
                    num_epochs = int(gui.num_epochs_var.get())
                except ValueError:
                    gui.log_message("Invalid input for sequence length, batch size, or number of epochs.")
                    time.sleep(5)
                    continue

                data_precision = gui.precision_var.get()
                training_mode = gui.training_mode_var.get()

                # Load data from CSV
                try:
                    data_df = pd.read_csv(DATA_FILE)
                    raw_data_list = [np.frombuffer(bytes.fromhex(s), dtype=np.uint8) for s in data_df['raw_data']]
                    labels_list = [eval(s) for s in data_df['labels']]  # Convert string representation back to list
                except Exception as e:
                    root.after(0, gui.log_message, f"Error loading data from CSV: {e}")
                    time.sleep(5)
                    continue
                data_length = len(raw_data_list)
                if data_length >= 100:
                    root.after(0, gui.log_message, "Starting training...")
                    # Preprocess data
                    X, y = preprocess_data(raw_data_list, labels_list, sequence_length, data_precision)
                    # Split data
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

                    # Create dataset
                    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
                    train_dataset = train_dataset.batch(batch_size)
                    train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
                    val_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
                    val_dataset = val_dataset.batch(batch_size)
                    val_dataset = val_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

                    loss_values = []
                    val_loss_values = []

                    # Clear previous model and session before creating a new one
                    tf.keras.backend.clear_session()
                    tf_model = None

                    # TensorFlow model training
                    # Build a new model
                    tf_model = create_tensorflow_model((sequence_length, X.shape[2]), data_precision, training_mode)

                    # Check for GPU availability and set device
                    gpus = tf.config.list_physical_devices('GPU')
                    if gui.use_gpu_var.get() and gpus:
                        try:
                            for gpu in gpus:
                                tf.config.experimental.set_memory_growth(gpu, True)
                            device = '/GPU:0'
                            root.after(0, gui.log_message, "Using GPU for training.")
                        except RuntimeError as e:
                            root.after(0, gui.log_message, f"GPU setup error: {e}")
                            device = '/CPU:0'
                    else:
                        device = '/CPU:0'
                        if gui.use_gpu_var.get():
                            root.after(0, gui.log_message, "GPU selected but not available. Using CPU instead.")
                        else:
                            root.after(0, gui.log_message, "Using CPU for training.")

                    # Define a custom callback to update GUI and collect loss
                    class GUIUpdater(tf.keras.callbacks.Callback):
                        def on_train_begin(self, logs=None):
                            self.training_start_time = time.time()
                            self.total_epochs = self.params.get('epochs', 1)
                            self.total_batches_seen = 0  # Total batches processed across all epochs

                        def on_epoch_begin(self, epoch, logs=None):
                            self.epoch_start_time = time.time()
                            self.batches_in_epoch = 0
                            # Try to get total steps per epoch
                            self.total_steps = self.params.get('steps', None)
                            if self.total_steps is None:
                                # Try to infer total steps from training data
                                if 'samples' in self.params and 'batch_size' in self.params:
                                    self.total_steps = (self.params['samples'] + self.params['batch_size'] - 1) // self.params['batch_size']
                                else:
                                    self.total_steps = 0  # Default to 0 if we can't determine
                            root.after(0, gui.update_current_step, 0, self.total_steps)

                        def on_train_batch_end(self, batch, logs=None):
                            self.batches_in_epoch += 1
                            self.total_batches_seen += 1

                            # Calculate elapsed time
                            elapsed_time = time.time() - self.training_start_time
                            root.after(0, gui.update_elapsed_time, elapsed_time)

                            # Estimate total time remaining
                            total_batches = self.total_epochs * self.total_steps
                            if self.total_batches_seen > 0:
                                time_per_batch = elapsed_time / self.total_batches_seen
                                batches_remaining = total_batches - self.total_batches_seen
                                total_time_remaining = time_per_batch * batches_remaining
                                root.after(0, gui.update_total_time_remaining, total_time_remaining)

                            # Estimate epoch time remaining
                            elapsed_epoch_time = time.time() - self.epoch_start_time
                            if self.batches_in_epoch > 0:
                                time_per_batch_epoch = elapsed_epoch_time / self.batches_in_epoch
                                batches_remaining_epoch = self.total_steps - self.batches_in_epoch
                                epoch_time_remaining = time_per_batch_epoch * batches_remaining_epoch
                                root.after(0, gui.update_epoch_time_remaining, epoch_time_remaining)

                            # Update iterations per second
                            if elapsed_epoch_time > 0:
                                it_per_sec = self.batches_in_epoch / elapsed_epoch_time
                                root.after(0, gui.update_iterations_per_second, it_per_sec)

                            # Update current step
                            current_step = self.batches_in_epoch
                            total_steps = self.total_steps
                            root.after(0, gui.update_current_step, current_step, total_steps)

                            # Check for stop or pause request
                            with control_lock:
                                if training_paused or training_stopped:
                                    self.model.stop_training = True

                        def on_epoch_end(self, epoch, logs=None):
                            global training_stopped, training_paused, current_epoch  # Declare globals
                            epoch_end_time = time.time()
                            epoch_duration = epoch_end_time - self.epoch_start_time
                            # Log epoch duration
                            root.after(0, gui.log_message, f"Epoch {epoch + 1} took {epoch_duration:.2f} seconds")
                            # Update loss values
                            loss = logs.get('loss')
                            val_loss = logs.get('val_loss')
                            loss_values.append(loss)
                            val_loss_values.append(val_loss)
                            current_epoch = epoch + 1
                            root.after(0, gui.update_current_epoch, current_epoch)
                            root.after(0, gui.log_message, f"Epoch {current_epoch}, Loss: {loss:.4f}, Val Loss: {val_loss:.4f}")
                            root.after(0, gui.update_loss_values, loss_values, val_loss_values)
                            # Reset current step and iterations per second
                            root.after(0, gui.update_current_step, 0, self.total_steps)
                            root.after(0, gui.update_iterations_per_second, 0)
                            # Update total time remaining
                            epochs_completed = epoch + 1
                            elapsed_time = time.time() - self.training_start_time
                            time_per_epoch = elapsed_time / epochs_completed
                            epochs_remaining = self.total_epochs - epochs_completed
                            total_time_remaining = time_per_epoch * epochs_remaining
                            root.after(0, gui.update_total_time_remaining, total_time_remaining)
                            # Update window title
                            root.after(0, gui.update_window_title, elapsed_time, total_time_remaining)
                            # Check for stop or pause request
                            with control_lock:
                                if training_paused or training_stopped:
                                    self.model.stop_training = True

                    gui_updater = GUIUpdater()
                    # ReduceLROnPlateau callback
                    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
                        monitor='val_loss',
                        factor=0.5,
                        patience=50,
                        min_delta=0.001
                    )
                    # EarlyStopping callback
                    early_stopping = tf.keras.callbacks.EarlyStopping(
                        monitor='val_loss',
                        patience=75,
                        restore_best_weights=True
                    )

                    # Start training
                    try:
                        with tf.device(device):
                            tf_model.fit(
                                train_dataset,
                                epochs=num_epochs,
                                validation_data=val_dataset,
                                callbacks=[gui_updater, reduce_lr, early_stopping],
                                verbose=0
                            )
                        # After training
                        root.after(0, gui.log_message, "Training completed.")
                        # Evaluate on test data
                        test_loss = tf_model.evaluate(val_dataset, verbose=0)
                        root.after(0, gui.log_message, f"Test Loss: {test_loss:.4f}")
                    except Exception as e:
                        root.after(0, gui.log_message, f"An error occurred during training: {e}")
                    finally:
                        with control_lock:
                            if training_stopped:
                                # If training is stopped, reset training variables
                                root.after(0, gui.log_message, "Training has been stopped.")
                                training_started = False
                                training_stopped = False  # Reset the flag
                                tf_model = None  # Reset the model
                                loss_values = []
                                val_loss_values = []
                                current_epoch = 0
                                root.after(0, gui.update_current_epoch, current_epoch)
                                root.after(0, gui.update_loss_values, loss_values, val_loss_values)
                                root.after(0, gui.iterations_per_second.set, "0")
                                root.after(0, gui.update_current_step, 0, 0)
                                # Clear the previous model and session
                                tf.keras.backend.clear_session()
                            elif training_paused:
                                root.after(0, gui.log_message, "Training has been paused.")
                                training_started = True  # Keep training_started True to allow resuming
                            else:
                                training_started = False  # Training completed normally
                else:
                    root.after(0, gui.log_message, f"Waiting for more data... Collected {data_length} samples so far.")
                    time.sleep(5)  # Check every 5 seconds
            else:
                # Training is paused
                time.sleep(1)
        else:
            time.sleep(1)

if __name__ == "__main__":
    # Load existing data from CSV file
    try:
        data_df = pd.read_csv(DATA_FILE)
        # Convert hex strings back to numpy arrays
        raw_data_cache = [np.frombuffer(bytes.fromhex(s), dtype=np.uint8) for s in data_df['raw_data']]
        labels_cache = [eval(s) for s in data_df['labels']]  # Convert string representation back to list
        print(f"Loaded {len(raw_data_cache)} data samples from {DATA_FILE}.")
    except FileNotFoundError:
        raw_data_cache = []
        labels_cache = []
        print("No existing data file found. You can load one using the 'Load CSV' button.")
    except Exception as e:
        print(f"Error loading data from {DATA_FILE}: {e}")
        raw_data_cache = []
        labels_cache = []
        print("You can load a CSV file using the 'Load CSV' button.")

    # Check for GPU availability and enable memory growth
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Available GPUs: {gpus}")
        except RuntimeError as e:
            print(f"Error setting up GPU memory growth: {e}")
    else:
        print("No GPU available. Training will be performed on CPU.")

    # Initialize Tkinter
    root = tk.Tk()
    gui = TrainingGUI(root)
    data_length = len(raw_data_cache)
    gui.update_data_count(data_length)
    gui.update_total_data_points(data_length)

    # Start training thread
    training_thread = threading.Thread(target=train_model, daemon=True)
    training_thread.start()

    # Start the Tkinter main loop
    root.mainloop()
