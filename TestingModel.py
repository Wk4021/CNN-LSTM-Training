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
from datetime import datetime
import sys
import os
import re  # For regular expressions

# Import the machine learning library, assuming TensorFlow/Keras
from tensorflow.keras.models import load_model

class RadarApp:
    def __init__(self, master):
        self.master = master
        master.title("Radar Data Monitoring")
        
        self.model = None  # Placeholder for the loaded model
        
        # Create Load Model button
        self.load_model_button = tk.Button(master, text="Load Model", command=self.load_model)
        self.load_model_button.pack()
        
        # Create a figure for plotting
        self.figure = Figure(figsize=(8, 6), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_title('BPM and RR over Time')
        self.ax.set_xlabel('Time (s)')
        self.ax.set_ylabel('Value')
        self.bpm_line, = self.ax.plot([], [], label='BPM')
        self.rr_line, = self.ax.plot([], [], label='RR')
        self.ax.legend()
        
        # Create a canvas and add the figure to it
        self.canvas = FigureCanvasTkAgg(self.figure, master=master)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack()
        
        # Data for plotting
        self.time_data = []
        self.bpm_data = []
        self.rr_data = []
        self.start_time = None  # To track the start time of plotting
        
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
    
    def load_model(self):
        # Open a file dialog to select a model file
        model_path = filedialog.askopenfilename(title="Select Model File", filetypes=[("Model Files", "*.h5")])
        if model_path:
            # Load the model
            self.model = load_model(model_path)
            print(f"Model loaded from {model_path}")
            print(f"Model input shape: {self.model.input_shape}")
    
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
        predictions = []
        last_time = time.time()
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
                
                if self.model:
                    # Prepare data for model input
                    # Adjust expected_input_size to match your model's input size
                    expected_input_shape = self.model.input_shape  # e.g., (None, 1030, 1)
                    print(f"Expected input shape: {expected_input_shape}")  # For debugging
                    
                    # Reshape data to match expected input shape
                    # For input shape (None, 1030, 1), we need to reshape data to (1, 1030, 1)
                    data_array = np.frombuffer(raw_radar_data_bytes, dtype=np.uint8)
                    data_array = data_array.astype('float32') / 255.0
                    data_array = data_array.reshape((1, -1, 1))  # Shape: (batch_size, timesteps, features)
                    
                    # Verify the reshaped data matches the expected input shape
                    if data_array.shape[1:] != expected_input_shape[1:]:
                        print(f"Reshaped data shape {data_array.shape} does not match expected input shape {expected_input_shape}")
                        continue
                    
                    # Make prediction
                    prediction = self.model.predict(data_array)
                    # Assume prediction is an array with BPM and RR
                    bpm_pred = prediction[0][0]
                    rr_pred = prediction[0][1]
                    
                    # Print the predictions to the console
                    print(f"Predicted BPM: {bpm_pred}, Predicted RR: {rr_pred}")
                    
                    # Add to predictions list with timestamp
                    predictions.append((time.time(), bpm_pred, rr_pred))
                else:
                    print("Model not loaded yet.")
            except queue.Empty:
                pass
            except Exception as e:
                print(f"Error in prediction thread: {e}")
            # Check if a second has passed
            current_time = time.time()
            if current_time - last_time >= 1.0:
                # Average the predictions made in the last second
                bpm_preds = [p[1] for p in predictions if p[0] >= last_time]
                rr_preds = [p[2] for p in predictions if p[0] >= last_time]
                if bpm_preds and rr_preds:
                    avg_bpm = sum(bpm_preds) / len(bpm_preds)
                    avg_rr = sum(rr_preds) / len(rr_preds)
                    # Add to prediction queue
                    self.prediction_queue.put((current_time, avg_bpm, avg_rr))
                last_time = current_time
                # Clear old predictions
                predictions = [p for p in predictions if p[0] >= last_time]
    
    def update_plot(self):
        try:
            while True:
                # Get averaged predictions
                timestamp, avg_bpm, avg_rr = self.prediction_queue.get_nowait()
                # Update data
                if self.start_time is None:
                    self.start_time = timestamp
                elapsed_time = timestamp - self.start_time
                self.time_data.append(elapsed_time)
                self.bpm_data.append(avg_bpm)
                self.rr_data.append(avg_rr)
                
                # Keep only the last 10 seconds of data
                while self.time_data and (elapsed_time - self.time_data[0]) > 10:
                    # Remove data points older than 10 seconds
                    self.time_data.pop(0)
                    self.bpm_data.pop(0)
                    self.rr_data.pop(0)
                
                # Update plot data
                self.bpm_line.set_data(self.time_data, self.bpm_data)
                self.rr_line.set_data(self.time_data, self.rr_data)
                
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
        # Schedule next update
        self.master.after(1000, self.update_plot)

if __name__ == "__main__":
    root = tk.Tk()
    app = RadarApp(root)
    root.mainloop()
