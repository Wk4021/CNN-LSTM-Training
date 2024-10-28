import tkinter as tk
from tkinter import filedialog
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import socket
import time
import queue
import numpy as np
import re  # For regular expressions

# Import the machine learning library, assuming TensorFlow/Keras
from tensorflow.keras.models import load_model

class RadarApp:
    def __init__(self, master):
        self.master = master
        master.title("Radar Data Monitoring")

        # Placeholder for the loaded models
        self.bpm_model = None
        self.rr_model = None
        self.psi_model = None

        # Create Load Model buttons
        self.load_bpm_model_button = tk.Button(master, text="Load BPM Model", command=lambda: self.load_model('BPM'))
        self.load_bpm_model_button.pack()

        self.load_rr_model_button = tk.Button(master, text="Load RR Model", command=lambda: self.load_model('RR'))
        self.load_rr_model_button.pack()

        self.load_psi_model_button = tk.Button(master, text="Load PSI Model", command=lambda: self.load_model('PSI'))
        self.load_psi_model_button.pack()

        # Labels to show the loaded model file names
        self.bpm_model_label = tk.Label(master, text="BPM Model: Not Loaded")
        self.bpm_model_label.pack()

        self.rr_model_label = tk.Label(master, text="RR Model: Not Loaded")
        self.rr_model_label.pack()

        self.psi_model_label = tk.Label(master, text="PSI Model: Not Loaded")
        self.psi_model_label.pack()

        # Labels to show the current readings
        self.bpm_label = tk.Label(master, text="Current BPM: N/A", font=("Helvetica", 14))
        self.bpm_label.pack()

        self.rr_label = tk.Label(master, text="Current RR: N/A", font=("Helvetica", 14))
        self.rr_label.pack()

        self.psi_label = tk.Label(master, text="Current PSI: N/A", font=("Helvetica", 14))
        self.psi_label.pack()

        # Create a figure for plotting
        self.figure = Figure(figsize=(8, 6), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_title('BPM, RR, and PSI over Time')
        self.ax.set_xlabel('Time (s)')
        self.ax.set_ylabel('Value')
        self.bpm_line, = self.ax.plot([], [], label='BPM')
        self.rr_line, = self.ax.plot([], [], label='RR')
        self.psi_line, = self.ax.plot([], [], label='PSI')
        self.ax.legend()

        # Create a canvas and add the figure to it
        self.canvas = FigureCanvasTkAgg(self.figure, master=master)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack()

        # Data for plotting
        self.time_data = []
        self.bpm_data = []
        self.rr_data = []
        self.psi_data = []
        self.start_time = None  # To track the start time of plotting

        # Heartbeat animation canvas
        self.heart_canvas = tk.Canvas(master, width=200, height=200)
        self.heart_canvas.pack()

        # Create a heart shape on the canvas
        self.heart = self.heart_canvas.create_oval(50, 50, 150, 150, fill='red')

        # Heartbeat animation variables
        self.heart_scale = 1.0
        self.heartbeat_interval = 1.0  # Time interval for heartbeats (in seconds)
        self.heartbeat_direction = 1

        # Start server thread
        self.data_queue = queue.Queue()
        self.prediction_queue = queue.Queue()
        self.server_thread = threading.Thread(target=self.server_thread_func, daemon=True)
        self.server_thread.start()

        # Start prediction processing thread
        self.prediction_thread = threading.Thread(target=self.prediction_thread_func, daemon=True)
        self.prediction_thread.start()

        # Schedule the GUI to update periodically
        self.update_plot()
        self.animate_heartbeat()

    def load_model(self, model_type):
        # Open a file dialog to select a model file
        model_path = filedialog.askopenfilename(title=f"Select {model_type} Model File", filetypes=[("Model Files", "*.h5")])
        if model_path:
            # Load the appropriate model
            if model_type == 'BPM':
                self.bpm_model = load_model(model_path)
                self.bpm_model_label.config(text=f"BPM Model: {model_path.split('/')[-1]}")  # Show file name in label
                print(f"BPM Model loaded from {model_path}")
            elif model_type == 'RR':
                self.rr_model = load_model(model_path)
                self.rr_model_label.config(text=f"RR Model: {model_path.split('/')[-1]}")  # Show file name in label
                print(f"RR Model loaded from {model_path}")
            elif model_type == 'PSI':
                self.psi_model = load_model(model_path)
                self.psi_model_label.config(text=f"PSI Model: {model_path.split('/')[-1]}")  # Show file name in label
                print(f"PSI Model loaded from {model_path}")

    def server_thread_func(self):
        # Start a server socket listening on port 65432
        HOST = ''  # Listen on all interfaces
        PORT = 65432
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((HOST, PORT))
            s.listen()
            print(f"Server listening on port {PORT}")
            while True:
                conn, addr = s.accept()
                threading.Thread(target=self.handle_client, args=(conn, addr), daemon=True).start()

    def handle_client(self, conn, addr):
        print(f"Connected by {addr}")
        buffer = ''
        with conn:
            while True:
                try:
                    data = conn.recv(4096)
                    if not data:
                        break
                    # Decode data to string
                    buffer += data.decode('utf-8', errors='ignore')
                    # Split data into lines
                    lines = buffer.split('\n')
                    buffer = lines[-1]  # Save incomplete line
                    for line in lines[:-1]:
                        # Process each complete line
                        self.data_queue.put(line)
                except ConnectionResetError:
                    print(f"Connection lost from {addr}")
                    break
                except Exception as e:
                    print(f"Error in handle_client: {e}")
                    break

    def prediction_thread_func(self):
        predictions = []  # To hold predictions within the second
        last_time = time.time()  # To track the time for averaging
        while True:
            try:
                # Get data from the queue
                line = self.data_queue.get(timeout=1)
                # Process the line
                print(f"Received line: {repr(line)}")  # For debugging

                # Split the line by whitespace
                parts = line.strip().split()
                print(f"Parts: {parts}")  # For debugging

                if len(parts) < 5:
                    print("Line does not have enough parts. Skipping.")
                    continue

                # The radar data starts at index 4
                radar_data_parts = parts[4:]

                # Join the radar data parts into a single string
                raw_radar_data_hex = ''.join(radar_data_parts)

                # Remove any non-hex characters
                raw_radar_data_hex = re.sub(r'[^0-9A-Fa-f]', '', raw_radar_data_hex)
                print(f"Raw radar data hex: {repr(raw_radar_data_hex)}")  # For debugging

                # Ensure the length is even
                if len(raw_radar_data_hex) % 2 != 0:
                    print("Hex string length is odd. Trimming the last character.")
                    raw_radar_data_hex = raw_radar_data_hex[:-1]

                # Convert hex to bytes
                try:
                    raw_radar_data_bytes = bytes.fromhex(raw_radar_data_hex)
                except ValueError as ve:
                    print(f"Failed to convert hex data to bytes: {ve}")
                    continue

                print(f"Received data bytes length: {len(raw_radar_data_bytes)}")  # For debugging

                # Prepare data for model input
                data_array = np.frombuffer(raw_radar_data_bytes, dtype=np.uint8).astype('float32')
                data_array = data_array.reshape((1, -1, 1))  # Shape: (batch_size, timesteps, features)

                bpm_pred, rr_pred, psi_pred = None, None, None

                # Make predictions if models are loaded
                if self.bpm_model:
                    bpm_pred = self.bpm_model.predict(data_array)[0][0]
                if self.rr_model:
                    rr_pred = self.rr_model.predict(data_array)[0][0]
                if self.psi_model:
                    psi_pred = self.psi_model.predict(data_array)[0][0]

                # Print the predictions to the console
                print(f"Predicted BPM: {bpm_pred}, Predicted RR: {rr_pred}, Predicted PSI: {psi_pred}")

                predictions.append((bpm_pred, rr_pred, psi_pred))

            except queue.Empty:
                pass
            except Exception as e:
                print(f"Error in prediction thread: {e}")

            # Every second, average the predictions and add to the queue
            current_time = time.time()
            if current_time - last_time >= 1.0:
                if predictions:
                    # Calculate average BPM, RR, and PSI
                    bpm_preds = [p[0] for p in predictions if p[0] is not None]
                    rr_preds = [p[1] for p in predictions if p[1] is not None]
                    psi_preds = [p[2] for p in predictions if p[2] is not None]

                    avg_bpm = sum(bpm_preds) / len(bpm_preds) if bpm_preds else 0
                    avg_rr = sum(rr_preds) / len(rr_preds) if rr_preds else 0
                    avg_psi = sum(psi_preds) / len(psi_preds) if psi_preds else 0

                    # Update GUI labels
                    self.bpm_label.config(text=f"Current BPM: {avg_bpm:.2f}")
                    self.rr_label.config(text=f"Current RR: {avg_rr:.2f}")
                    self.psi_label.config(text=f"Current PSI: {avg_psi:.2f}")

                    print(f"Average BPM: {avg_bpm:.2f}, Average RR: {avg_rr:.2f}, Average PSI: {avg_psi:.2f}")

                    # Add to the queue for plotting
                    self.prediction_queue.put((current_time, avg_bpm, avg_rr, avg_psi))

                # Reset the predictions list and the timer for the next second
                predictions = []
                last_time = current_time

    def update_plot(self):
        try:
            while True:
                # Get the averaged predictions from the queue
                timestamp, avg_bpm, avg_rr, avg_psi = self.prediction_queue.get_nowait()

                # Update data
                if self.start_time is None:
                    self.start_time = timestamp
                elapsed_time = timestamp - self.start_time

                # Append new point for the current second
                self.time_data.append(elapsed_time)
                self.bpm_data.append(avg_bpm)
                self.rr_data.append(avg_rr)
                self.psi_data.append(avg_psi)

                # Keep only the last 10 seconds of data
                while self.time_data and (elapsed_time - self.time_data[0]) > 10:
                    # Remove data points older than 10 seconds
                    self.time_data.pop(0)
                    self.bpm_data.pop(0)
                    self.rr_data.pop(0)
                    self.psi_data.pop(0)

                # Update plot data with the averaged points
                self.bpm_line.set_data(self.time_data, self.bpm_data)
                self.rr_line.set_data(self.time_data, self.rr_data)
                self.psi_line.set_data(self.time_data, self.psi_data)

                # Adjust x-axis limits to show the last 10 seconds
                self.ax.set_xlim(max(0, elapsed_time - 10), elapsed_time)

                # Adjust y-axis limits based on data
                self.ax.relim()
                self.ax.autoscale_view()

                # Redraw the canvas
                self.canvas.draw()
        except queue.Empty:
            pass
        except Exception as e:
            print(f"Error in update_plot: {e}")
        finally:
            # Schedule next update even if no predictions are made
            if not self.time_data:
                elapsed_time = (time.time() - self.start_time) if self.start_time else 0
                self.time_data.append(elapsed_time)
                self.bpm_data.append(0)
                self.rr_data.append(0)
                self.psi_data.append(0)

                # Keep the graph moving forward even without predictions
                self.bpm_line.set_data(self.time_data, self.bpm_data)
                self.rr_line.set_data(self.time_data, self.rr_data)
                self.psi_line.set_data(self.time_data, self.psi_data)

                # Adjust x-axis limits and y-axis limits
                self.ax.set_xlim(max(0, elapsed_time - 10), elapsed_time)
                self.ax.relim()
                self.ax.autoscale_view()
                self.canvas.draw()

        # Schedule next update
        self.master.after(1000, self.update_plot)

    def animate_heartbeat(self):
        bpm_text = self.bpm_label.cget("text").split(":")[1].strip()
        try:
            # Try to convert the BPM to a float
            bpm = float(bpm_text)  # Parse BPM from the label if it's a number
        except ValueError:
            bpm = None  # If parsing fails or the label contains "N/A", set BPM to None

        if bpm is None:
            # If BPM is not available or valid, stop the animation
            self.heartbeat_direction = 1
            self.heart_scale = 1.0
            self.heart_canvas.scale(self.heart, 100, 100, self.heart_scale, self.heart_scale)
        else:
            # Continue animation when BPM is available
            if self.heart_scale >= 1.2:
                self.heartbeat_direction = -1
            elif self.heart_scale <= 1.0:
                self.heartbeat_direction = 1

            # Adjust heart scale based on BPM
            self.heartbeat_interval = 60 / bpm

            # Apply the heart scaling
            self.heart_scale += 0.01 * self.heartbeat_direction
            self.heart_canvas.scale(self.heart, 100, 100, self.heart_scale, self.heart_scale)

            # Schedule the next heartbeat animation frame
            self.master.after(int(self.heartbeat_interval * 1000 / 60), self.animate_heartbeat)

if __name__ == "__main__":
    root = tk.Tk()
    app = RadarApp(root)
    root.mainloop()
