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
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model

# Set NumPy print options to avoid scientific notation
np.set_printoptions(suppress=True)

# Import necessary RL components
from collections import deque
import random

# Import graphviz for network visualization
from graphviz import Digraph
from PIL import Image, ImageTk

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
            augmented_X.append(np.pad(X[i], (shift, 0), mode='constant')[:-shift])
        elif shift < 0:
            augmented_X.append(np.pad(X[i], (0, -shift), mode='constant')[-shift:])
        else:
            augmented_X.append(X[i])

        augmented_y.append(y[i])  # Keep labels the same for augmented samples

    return np.array(augmented_X), np.array(augmented_y)

# GUI Application
class TrainingApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Training Application with DQN and Enhanced Neuron Visualizer")

        # Variables
        self.num_epochs_var = tk.IntVar(value=100)
        self.batch_size_var = tk.IntVar(value=64)
        self.error_threshold_var = tk.DoubleVar(value=10.0)
        self.learning_rate_var = tk.DoubleVar(value=0.001)
        self.save_interval_var = tk.IntVar(value=10)
        self.lr_schedule_var = tk.StringVar(value="")
        self.model_type_var = tk.IntVar(value=1)  # Default to DQN
        self.current_epoch = 0
        self.total_epochs = 0
        self.training = False
        self.paused = False
        self.stop_training_flag = False
        self.lr_schedule = {}
        self.training_thread = None

        # Initialize metric lists
        self.loss_values = []
        self.val_loss_values = []

        # Variables for Label Selection
        self.train_bpm_var = tk.BooleanVar(value=True)
        self.train_rr_var = tk.BooleanVar(value=False)
        self.train_psi_var = tk.BooleanVar(value=False)

        # Initialize model
        self.model = None
        
        # Load data
        self.load_data()

        # Build GUI
        self.build_gui()

        # Experience replay buffer for DQN
        self.replay_buffer = deque(maxlen=2000)
        self.gamma = 0.95  # Discount factor

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

        # Select only the labels chosen by the user
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

        # Normalize the data
        self.state_size = self.X_train.shape[1]
        self.action_size = len(selected_indices)

        # Check dimensions after stacking
        print(f"X_train shape: {self.X_train.shape}, y_train shape: {self.y_train.shape}")
        print(f"X_val shape: {self.X_val.shape}, y_val shape: {self.y_val.shape}")
        print(f"X_test shape: {self.X_test.shape}, y_test shape: {self.y_test.shape}")

    # Function to get selected label indices
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
        # Configure grid layout
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

        # Learning rate input
        tk.Label(self.master, text="Learning Rate:").grid(row=3, column=0, sticky='e')
        tk.Entry(self.master, textvariable=self.learning_rate_var).grid(row=3, column=1, sticky='we')

        # Save model every N epochs
        tk.Label(self.master, text="Save Model Every N Epochs:").grid(row=4, column=0, sticky='e')
        tk.Entry(self.master, textvariable=self.save_interval_var).grid(row=4, column=1, sticky='we')

        # Learning rate schedule input
        tk.Label(self.master, text="Learning Rate Schedule:").grid(row=5, column=0, sticky='e')
        tk.Entry(self.master, textvariable=self.lr_schedule_var).grid(row=5, column=1, columnspan=3, sticky='we')
        tk.Label(self.master, text="Format: epoch:lr,epoch:lr").grid(row=5, column=4, sticky='w')

        # Model type selection (hidden or disabled if only DQN)
        tk.Label(self.master, text="Model Type:").grid(row=6, column=0, sticky='e')
        model_options = [("DQN", 1)]
        for i, (text, value) in enumerate(model_options):
            tk.Radiobutton(self.master, text=text, variable=self.model_type_var, value=value).grid(row=6, column=1 + i, sticky='w')

        # Label selection
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

        # Live Neuron Visualizer Label
        tk.Label(self.master, text="Neuron Activations:").grid(row=8, column=0, sticky='nw')

        # Console output
        tk.Label(self.master, text="Console Output:").grid(row=8, column=3, sticky='nw')
        self.console = tk.Text(self.master, height=10)
        self.console.grid(row=9, column=3, columnspan=3, sticky='nsew')

        # Training Metrics Graph
        self.figure = Figure(figsize=(6, 4), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_title("Training Metrics")
        self.ax.set_xlabel("Epoch")
        self.ax.set_ylabel("Average Reward")
        self.line_loss, = self.ax.plot([], [], label='Average Reward')
        self.ax.legend()

        self.canvas = FigureCanvasTkAgg(self.figure, master=self.master)
        self.canvas.get_tk_widget().grid(row=10, column=3, columnspan=3, sticky='nsew')

        # Neuron Activation Graph
        self.neuron_figure = Figure(figsize=(6, 4), dpi=100)
        self.neuron_ax = self.neuron_figure.add_subplot(111)
        self.neuron_ax.set_title("Neuron Activations")
        self.neuron_ax.set_xlabel("Neuron Index")
        self.neuron_ax.set_ylabel("Activation")
        self.neuron_canvas = FigureCanvasTkAgg(self.neuron_figure, master=self.master)
        self.neuron_canvas.get_tk_widget().grid(row=9, column=0, rowspan=2, columnspan=3, sticky='nsew')

        # Network Architecture Visualization
        tk.Label(self.master, text="Network Architecture:").grid(row=11, column=0, sticky='nw')
        self.network_canvas = tk.Label(self.master)
        self.network_canvas.grid(row=12, column=0, rowspan=2, columnspan=6, sticky='nsew')

        # Generate and display the network architecture
        self.display_network_architecture()

        # Configure row and column weights for resizing
        for i in range(13):
            self.master.rowconfigure(i, weight=1)
        for i in range(6):
            self.master.columnconfigure(i, weight=1)

    def display_network_architecture(self):
        if self.model is None:
            return

        # Generate network graph using graphviz
        dot = Digraph()
        for i, layer in enumerate(self.model.layers):
            layer_name = f"{layer.__class__.__name__}_{i}"
            dot.node(layer_name, label=f"{layer.__class__.__name__}\n{layer.output_shape}")
            if i > 0:
                prev_layer_name = f"{self.model.layers[i - 1].__class__.__name__}_{i - 1}"
                dot.edge(prev_layer_name, layer_name)

        # Save and display the graph
        dot.render('network_architecture', format='png', cleanup=True)
        image = Image.open('network_architecture.png')
        image = image.resize((800, 300), Image.LANCZOS)
        photo = ImageTk.PhotoImage(image)
        self.network_canvas.configure(image=photo)
        self.network_canvas.image = photo  # Keep a reference

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

            # Reload data to apply label selection
            self.load_data()

            self.training_thread = threading.Thread(target=self.train_dqn)
            self.training_thread.start()
        elif self.paused:
            self.paused = False
            self.stop_training_flag = False
            self.training_thread = threading.Thread(target=self.train_dqn)
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
        self.batch_size_var.set(64)
        self.error_threshold_var.set(10.0)
        self.learning_rate_var.set(0.001)
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
        self.train_rr_var.set(False)
        self.train_psi_var.set(False)

        # Clear metrics
        self.loss_values = []
        self.val_loss_values = []

        # Reset model
        self.model = None

        # Clear console
        self.console.delete('1.0', tk.END)

        # Reset progress bar
        self.progress_var.set(0)

        # Clear graphs
        self.ax.clear()
        self.ax.set_title("Training Metrics")
        self.ax.set_xlabel("Epoch")
        self.ax.set_ylabel("Average Reward")
        self.ax.legend()
        self.canvas.draw()

        self.neuron_ax.clear()
        self.neuron_ax.set_title("Neuron Activations")
        self.neuron_ax.set_xlabel("Neuron Index")
        self.neuron_ax.set_ylabel("Activation")
        self.neuron_canvas.draw()

        # Clear network architecture
        self.network_canvas.configure(image='')
        self.network_canvas.image = None

        # Reset time estimates
        self.time_per_epoch_var.set("00:00:00")
        self.total_time_var.set("00:00:00")

        # Update GUI elements
        self.update_learning_rate_display()

        self.print_to_console("Application reset. Ready to start a new training session.")

    def train_dqn(self):
        self.print_to_console("DQN Training started...")
        batch_size = self.batch_size_var.get()
        epochs = self.num_epochs_var.get()
        error_threshold = self.error_threshold_var.get()

        # Build DQN model if not already loaded
        if self.model is None:
            self.build_dqn_model()
            # Display network architecture
            self.display_network_architecture()

        # Initialize variables
        epsilon = 1.0  # Exploration rate
        epsilon_min = 0.01
        epsilon_decay = 0.995

        # Prepare data as environment
        state_space = self.X_train
        action_space = self.y_train

        # TensorBoard callback
        log_dir = "logs/fit/" + time.strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0, profile_batch=0)

        total_steps = epochs * len(state_space)
        current_step = 0

        for e in range(epochs):
            if self.stop_training_flag:
                break

            self.current_epoch = e + 1

            # Adjust learning rate if in schedule
            if (e + 1) in self.lr_schedule:
                new_lr = self.lr_schedule[e + 1]
                tf.keras.backend.set_value(self.model.optimizer.learning_rate, new_lr)
                self.learning_rate_var.set(new_lr)
                self.update_learning_rate_display()
                self.print_to_console(f"Learning rate adjusted to {new_lr} at epoch {e + 1}")

            # Shuffle the data at each epoch
            indices = np.arange(len(state_space))
            np.random.shuffle(indices)
            state_space = state_space[indices]
            action_space = action_space[indices]

            total_reward = 0

            start_time = time.time()

            for i in range(len(state_space)):
                if self.stop_training_flag:
                    break

                state = state_space[i]
                state = np.reshape(state, [1, self.state_size])

                # Epsilon-greedy action selection
                if np.random.rand() <= epsilon:
                    # Random action
                    action = np.random.uniform(low=0, high=1, size=(1, self.action_size))
                else:
                    # Predict action
                    action = self.model.predict(state, verbose=0)

                # Calculate reward (negative MSE loss)
                target = action_space[i]
                reward = -np.mean((action - target) ** 2)
                total_reward += reward

                # Next state is the same in this setup
                next_state = state
                done = True  # Each step is terminal in this setup

                # Store experience in replay buffer
                self.replay_buffer.append((state, action, reward, next_state, done))

                # Perform experience replay
                if len(self.replay_buffer) > batch_size:
                    self.replay(batch_size, tensorboard_callback)

                # Update progress bar
                current_step += 1
                progress = (current_step / total_steps) * 100
                self.update_progress(progress)

            # Update epsilon
            if epsilon > epsilon_min:
                epsilon *= epsilon_decay

            # Update console
            avg_reward = total_reward / len(state_space)
            self.print_to_console(f"Epoch {e+1}/{epochs} - Epsilon: {epsilon:.4f} - Avg Reward: {avg_reward:.4f}")

            # Save model every N epochs
            if (e + 1) % self.save_interval_var.get() == 0:
                model_name = f"Models/DQN_Model_Epoch_{e+1}.h5"
                self.model.save(model_name)
                self.print_to_console(f"Model saved as {model_name}")

            # Update graphs
            self.loss_values.append(avg_reward)
            self.update_graph(range(1, len(self.loss_values) + 1))

            # Update neuron activations
            self.update_neuron_visualization()

            # Update time estimates
            epoch_time = time.time() - start_time
            self.update_time_per_epoch(epoch_time)
            total_time_remaining = epoch_time * (epochs - (e + 1))
            self.update_total_time(total_time_remaining)

        if not self.stop_training_flag:
            # Save final model
            final_model_path = "Models/DQN_FinalModel.h5"
            self.model.save(final_model_path)
            self.print_to_console(f"Final model saved as {final_model_path}")
            self.print_to_console("DQN Training completed.")
        else:
            self.print_to_console("Training was stopped before completion.")

        self.training = False

    def build_dqn_model(self):
        # Simple DQN model
        self.model = tf.keras.Sequential([
            layers.Dense(128, input_dim=self.state_size, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(self.action_size)
        ])
        self.model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate_var.get()))

    def replay(self, batch_size, tensorboard_callback):
        minibatch = random.sample(self.replay_buffer, batch_size)
        states = []
        targets_f = []
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                # Predict future reward
                target = reward + self.gamma * np.amax(self.model.predict(next_state, verbose=0)[0])
            target_f = self.model.predict(state, verbose=0)
            target_f[0] = target
            states.append(state[0])
            targets_f.append(target_f[0])
        states = np.array(states)
        targets_f = np.array(targets_f)
        # Fit the model using TensorBoard callback
        self.model.fit(states, targets_f, epochs=1, verbose=0, callbacks=[tensorboard_callback])

    def update_progress(self, value):
        self.progress_var.set(value)
        self.master.update_idletasks()

    def update_graph(self, epochs):
        self.ax.clear()
        self.ax.set_title("Training Metrics")
        self.ax.set_xlabel("Epoch")
        self.ax.set_ylabel("Average Reward")
        self.ax.plot(epochs, self.loss_values, label='Average Reward')
        self.ax.legend()
        self.canvas.draw()

    def update_neuron_visualization(self):
        if self.model is None:
            return

        # Get activations for a sample input
        sample_input = self.X_test[0]
        sample_input = np.reshape(sample_input, [1, self.state_size])

        # Create a model that outputs the activations
        layer_outputs = [layer.output for layer in self.model.layers]
        activation_model = tf.keras.models.Model(inputs=self.model.input, outputs=layer_outputs)
        activations = activation_model.predict(sample_input, verbose=0)

        # Visualize activations on the network graph
        dot = Digraph()
        for i, layer in enumerate(self.model.layers):
            layer_name = f"{layer.__class__.__name__}_{i}"
            # Get activations
            activation = activations[i][0]
            # Format activations
            activation_str = ', '.join([f"{act:.2f}" for act in activation])
            dot.node(layer_name, label=f"{layer.__class__.__name__}\nActivation:\n{activation_str}")
            if i > 0:
                prev_layer_name = f"{self.model.layers[i - 1].__class__.__name__}_{i - 1}"
                dot.edge(prev_layer_name, layer_name)

        # Save and display the graph
        dot.render('network_activation', format='png', cleanup=True)
        image = Image.open('network_activation.png')
        image = image.resize((800, 300), Image.ANTIALIAS)
        photo = ImageTk.PhotoImage(image)
        self.neuron_canvas.get_tk_widget().destroy()  # Remove previous canvas
        self.neuron_canvas = tk.Label(self.master, image=photo)
        self.neuron_canvas.image = photo  # Keep a reference
        self.neuron_canvas.grid(row=9, column=0, rowspan=2, columnspan=3, sticky='nsew')

    def update_time_per_epoch(self, epoch_time):
        time_str = time.strftime('%H:%M:%S', time.gmtime(epoch_time))
        self.time_per_epoch_var.set(time_str)
        self.master.update_idletasks()

    def update_total_time(self, total_time):
        time_str = time.strftime('%H:%M:%S', time.gmtime(total_time))
        self.total_time_var.set(time_str)
        self.master.update_idletasks()

    def print_to_console(self, message):
        self.console.insert(tk.END, f"{message}\n")
        self.console.see(tk.END)
        self.master.update_idletasks()

    def update_learning_rate_display(self):
        # Force update of the learning rate Entry widget
        self.master.update_idletasks()

if __name__ == "__main__":
    root = tk.Tk()
    app = TrainingApp(root)
    root.mainloop()
