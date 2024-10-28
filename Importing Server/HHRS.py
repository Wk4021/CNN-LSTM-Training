import tkinter as tk
from tkinter import ttk
from tkinter import scrolledtext
import threading
import time
from datetime import datetime, timedelta
import socket
import matplotlib
import csv
import sys
import os
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from collections import deque
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Ensure TkAgg backend is used
matplotlib.use("TkAgg")

# Server configuration
HOST = '0.0.0.0'  # Listen on all network interfaces
RADAR_PORT = 65432      # Port for radar data
HEART_RATE_PORT = 65433   # Port for heart rate data

# Global variables for data storage
# Using deque for efficient append and pop operations
HEART_RATE_HISTORY_MAX_SECONDS = 10  # Keep heart rate data from the last 10 seconds
heart_rate_history = deque()  # Stores {'RTC_dt': datetime, 'BPM': int}

# Locks for thread-safe data access
heart_rate_data_lock = threading.Lock()

# Output file
output_file_name = 'HugoHR.CSV'

class CSVFileHandler(FileSystemEventHandler):
    def __init__(self, app):
        self.app = app

    def on_modified(self, event):
        if event.src_path == os.path.abspath(self.app.csv_file_name):
            self.app.error_messages.append("CSV file modified externally. Reloading data.")
            self.app.load_existing_data_count()

    def on_deleted(self, event):
        if event.src_path == os.path.abspath(self.app.csv_file_name):
            self.app.error_messages.append("CSV file deleted externally.")
            self.app.total_data_points = 0
            self.test_sizes = []
            self.update_labels()

    def on_created(self, event):
        if event.src_path == os.path.abspath(self.app.csv_file_name):
            self.app.error_messages.append("CSV file created externally. Reloading data.")
            self.app.load_existing_data_count()

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        # Set up the window
        self.title("Data Collection GUI")
        self.geometry("800x600")
        
        # Make the window resizable
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)
        
        # Main frame
        self.main_frame = tk.Frame(self)
        self.main_frame.grid(sticky='nsew')
        self.main_frame.rowconfigure(6, weight=1)  # For the error log
        self.main_frame.columnconfigure(0, weight=1)
        
        # Initialize variables
        self.total_data_points = 0
        self.current_test_data_points = 0
        self.heart_rate_data = []  # list to hold heart rate data for graph (timestamp, BPM)
        self.data_cache = []  # cache to hold data before saving to CSV
        self.receiving_bpm = False
        self.receiving_radar = False
        self.error_messages = []
        self.csv_file_name = output_file_name
        self.cache_lock = threading.Lock()
        self.last_radar_data_time = None  # To track last radar data time
        self.test_sizes = []  # List to store sizes of each test
        
        # GUI elements
        # Total data points label
        self.total_data_label = tk.Label(self.main_frame, text="Total data points: 0")
        self.total_data_label.grid(row=0, column=0, sticky='w', padx=5, pady=2)
        
        # Current test data points label
        self.current_test_data_label = tk.Label(self.main_frame, text="Data points in current test: 0")
        self.current_test_data_label.grid(row=1, column=0, sticky='w', padx=5, pady=2)
        
        # Number of tests label
        self.test_count_label = tk.Label(self.main_frame, text="Number of tests: 0")
        self.test_count_label.grid(row=2, column=0, sticky='w', padx=5, pady=2)
        
        # Test sizes label
        self.test_sizes_label = tk.Label(self.main_frame, text="Test sizes: []")
        self.test_sizes_label.grid(row=3, column=0, sticky='w', padx=5, pady=2)
        
        # BPM and Radar status indicators
        status_frame = tk.Frame(self.main_frame)
        status_frame.grid(row=4, column=0, columnspan=2, sticky='we', padx=5, pady=5)
        status_frame.columnconfigure(1, weight=1)
        status_frame.columnconfigure(3, weight=1)
        
        # BPM status
        self.bpm_status_label = tk.Label(status_frame, text="BPM Status:")
        self.bpm_status_label.grid(row=0, column=0, sticky='e')
        self.bpm_status_indicator = tk.Canvas(status_frame, width=20, height=20)
        self.bpm_status_indicator.grid(row=0, column=1, sticky='w')
        self.bpm_status_indicator.create_oval(2, 2, 18, 18, fill='red')
        
        # Radar status
        self.radar_status_label = tk.Label(status_frame, text="Radar Status:")
        self.radar_status_label.grid(row=0, column=2, sticky='e')
        self.radar_status_indicator = tk.Canvas(status_frame, width=20, height=20)
        self.radar_status_indicator.grid(row=0, column=3, sticky='w')
        self.radar_status_indicator.create_oval(2, 2, 18, 18, fill='red')
        
        # Heart rate tracking graph
        self.figure = Figure(figsize=(6, 3), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_title("Heart Rate (Last 10 seconds)")
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("BPM")
        self.line, = self.ax.plot([], [])
        self.canvas = FigureCanvasTkAgg(self.figure, self.main_frame)
        self.canvas.get_tk_widget().grid(row=5, column=0, columnspan=2, sticky='nsew', padx=5, pady=5)
        
        # Allow the graph to expand
        self.main_frame.rowconfigure(5, weight=1)
        
        # Error logging area
        self.error_label = tk.Label(self.main_frame, text="Error Log:")
        self.error_label.grid(row=6, column=0, sticky='w', padx=5, pady=2)
        self.error_text = scrolledtext.ScrolledText(self.main_frame, wrap=tk.WORD)
        self.error_text.grid(row=7, column=0, columnspan=2, sticky='nsew', padx=5, pady=5)
        
        # Allow the error log to expand
        self.main_frame.rowconfigure(7, weight=1)
        
        # Start updating the GUI
        self.after(1000, self.update_gui)
        
        # Start server threads
        threading.Thread(target=self.start_heart_rate_server, daemon=True).start()
        threading.Thread(target=self.start_radar_server, daemon=True).start()
        
        # Set up file monitoring
        self.setup_file_monitoring()
        
        # Load existing data count if any
        self.load_existing_data_count()
        # Update labels
        self.update_labels()
        
    def setup_file_monitoring(self):
        self.event_handler = CSVFileHandler(self)
        self.observer = Observer()
        self.observer.schedule(self.event_handler, path=os.path.dirname(os.path.abspath(self.csv_file_name)), recursive=False)
        self.observer.start()
    
    def update_labels(self):
        self.total_data_label.config(text=f"Total data points: {self.total_data_points}")
        self.current_test_data_label.config(text=f"Data points in current test: {self.current_test_data_points}")
        self.test_count_label.config(text=f"Number of tests: {len(self.test_sizes)}")
        self.test_sizes_label.config(text=f"Test sizes: {self.test_sizes}")
    
    def load_existing_data_count(self):
        self.total_data_points = 0
        self.test_sizes = []
        current_test_data_points = 0
        if os.path.exists(self.csv_file_name):
            with open(self.csv_file_name, 'r', newline='') as csvfile:
                reader = csv.reader(csvfile)
                header = next(reader, None)  # Read header if present
                for row in reader:
                    if not row:
                        # Blank line indicates end of test
                        if current_test_data_points > 0:
                            self.test_sizes.append(current_test_data_points)
                            current_test_data_points = 0
                    else:
                        self.total_data_points += 1
                        current_test_data_points += 1
                # Add the last test if not ended with a blank line
                if current_test_data_points > 0:
                    self.test_sizes.append(current_test_data_points)
        else:
            # File does not exist, create it and write header
            with open(self.csv_file_name, 'w', newline='') as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow(['RawRadar', 'BPM'])  # Write CSV header
        # Update labels
        self.update_labels()
    
    def update_gui(self):
        # Update labels
        self.update_labels()
        
        # Update BPM status indicator
        if self.receiving_bpm:
            self.bpm_status_indicator.create_oval(2, 2, 18, 18, fill='green')
        else:
            self.bpm_status_indicator.create_oval(2, 2, 18, 18, fill='red')
        
        # Update Radar status indicator
        if self.receiving_radar:
            self.radar_status_indicator.create_oval(2, 2, 18, 18, fill='green')
        else:
            self.radar_status_indicator.create_oval(2, 2, 18, 18, fill='red')
        
        # Update heart rate graph
        self.update_heart_rate_graph()
        
        # Update error messages
        self.update_error_log()
        
        # Reset the receiving data flags
        self.receiving_bpm = False
        self.receiving_radar = False
        
        # Check if radar data has stopped coming in
        if self.last_radar_data_time is not None:
            time_since_last_radar = time.time() - self.last_radar_data_time
            if time_since_last_radar > 2:  # Threshold in seconds
                # Consider the test ended
                self.end_test()
        
        # Schedule the next update
        self.after(1000, self.update_gui)
    
    def end_test(self):
        with self.cache_lock:
            if self.data_cache:
                # Save data to CSV
                self.save_data_to_csv()
                # Add a blank line to CSV
                try:
                    with open(self.csv_file_name, 'a', newline='') as csvfile:
                        csv_writer = csv.writer(csvfile)
                        csv_writer.writerow([])
                except Exception as e:
                    self.error_messages.append(f"Error adding blank line to CSV: {e}")
    
                # Append current test data points to test_sizes
                self.test_sizes.append(self.current_test_data_points)
                # Reset current test data points
                self.current_test_data_points = 0
                # Reset data cache
                self.data_cache = []
            else:
                # No data to save, just reset current test data points
                self.current_test_data_points = 0
        # Reset last radar data time
        self.last_radar_data_time = None
        # Update labels
        self.update_labels()
    
    def update_heart_rate_graph(self):
        # Remove data older than 10 seconds
        current_time = time.time()
        self.heart_rate_data = [(t, bpm) for t, bpm in self.heart_rate_data if current_time - t <= 10]
        
        # Extract times and bpm
        times = [t - current_time + 10 for t, bpm in self.heart_rate_data]  # shift time to display last 10 seconds
        bpms = [bpm for t, bpm in self.heart_rate_data]
        
        # Update the plot
        self.ax.clear()
        self.ax.set_title("Heart Rate (Last 10 seconds)")
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("BPM")
        self.ax.set_xlim(0, 10)
        if bpms:
            self.ax.set_ylim(min(bpms) - 10, max(bpms) + 10)
        else:
            self.ax.set_ylim(0, 100)
        if times:
            self.ax.plot(times, bpms)
        self.canvas.draw()
    
    def update_error_log(self):
        # Append any new error messages
        if self.error_messages:
            for msg in self.error_messages:
                self.error_text.insert(tk.END, msg + '\n')
            self.error_text.see(tk.END)
            self.error_messages = []
    
    def start_heart_rate_server(self):
        heart_rate_server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        heart_rate_server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        heart_rate_server_socket.bind((HOST, HEART_RATE_PORT))
        heart_rate_server_socket.listen()
        self.error_messages.append(f"Heart rate server listening on {HOST}:{HEART_RATE_PORT}")
        while True:
            client_sock, client_addr = heart_rate_server_socket.accept()
            client_handler = threading.Thread(
                target=self.handle_heart_rate_connection,
                args=(client_sock, client_addr),
                daemon=True
            )
            client_handler.start()
    
    def start_radar_server(self):
        radar_server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        radar_server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        radar_server_socket.bind((HOST, RADAR_PORT))
        radar_server_socket.listen()
        self.error_messages.append(f"Radar server listening on {HOST}:{RADAR_PORT}")
        while True:
            client_sock, client_addr = radar_server_socket.accept()
            client_handler = threading.Thread(
                target=self.handle_radar_connection,
                args=(client_sock, client_addr),
                daemon=True
            )
            client_handler.start()
    
    def handle_heart_rate_connection(self, client_socket, client_address):
        self.error_messages.append(f"Accepted heart rate connection from {client_address}")
        buffer = ''
        with client_socket:
            while True:
                try:
                    data = client_socket.recv(1024)  # Buffer size
                    if not data:
                        break
                    decoded_data = data.decode('utf-8')
                    buffer += decoded_data
                    while '\n' in buffer:
                        line, buffer = buffer.split('\n', 1)
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            # Assuming data format: RTC,BPM
                            parts = line.split(',')
                            if len(parts) == 2:
                                rtc_str, bpm_str = parts
                                bpm = int(bpm_str)
                                # Parse RTC
                                try:
                                    rtc_dt = datetime.strptime(rtc_str.strip(), '%Y-%m-%d %H:%M:%S.%f')
                                except ValueError:
                                    rtc_dt = datetime.strptime(rtc_str.strip(), '%Y-%m-%d %H:%M:%S')
                                with heart_rate_data_lock:
                                    # Append to heart rate history
                                    heart_rate_history.append({'RTC_dt': rtc_dt, 'BPM': bpm})
                                    # Clean up old data
                                    while heart_rate_history and (rtc_dt - heart_rate_history[0]['RTC_dt']).total_seconds() > HEART_RATE_HISTORY_MAX_SECONDS:
                                        heart_rate_history.popleft()
                                # Update heart rate data for the graph
                                current_time = time.time()
                                self.heart_rate_data.append((current_time, bpm))
                                # Set the flag indicating that we are receiving BPM data
                                self.receiving_bpm = True
                            else:
                                self.error_messages.append(f"Invalid heart rate data received: {line}")
                        except ValueError as e:
                            self.error_messages.append(f"Invalid heart rate data: {line}, error: {e}")
                except Exception as e:
                    self.error_messages.append(f"Error in heart rate connection: {e}")
                    break
    
    def handle_radar_connection(self, client_socket, client_address):
        self.error_messages.append(f"Accepted radar connection from {client_address}")
        buffer = ''
        with client_socket:
            while True:
                try:
                    data = client_socket.recv(1024)  # Buffer size
                    if not data:
                        break
                    decoded_data = data.decode('utf-8')
                    buffer += decoded_data
                    while '\n' in buffer:
                        line, buffer = buffer.split('\n', 1)
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            # Assuming data format: RTC RawRadarData
                            # Split the line into timestamp and data
                            parts = line.split(' ')
                            if len(parts) >= 3:
                                # Recombine the first two parts for the timestamp
                                rtc_part = ' '.join(parts[:2])
                                raw_data = ' '.join(parts[2:])
                                rtc_part = rtc_part.strip()
                                raw_data = raw_data.strip()
                                # Parse RTC
                                try:
                                    radar_rtc_dt = datetime.strptime(rtc_part, '%Y-%m-%d %H:%M:%S.%f')
                                except ValueError:
                                    radar_rtc_dt = datetime.strptime(rtc_part, '%Y-%m-%d %H:%M:%S')
                                # Set the flag indicating that we are receiving Radar data
                                self.receiving_radar = True
                                # Update last radar data time
                                self.last_radar_data_time = time.time()
                                # Process radar data
                                self.process_radar_data(radar_rtc_dt, raw_data, line)
                            else:
                                self.error_messages.append(f"Invalid radar data format: {line}")
                        except ValueError as e:
                            self.error_messages.append(f"Invalid radar data: {line}, error: {e}")
                except Exception as e:
                    self.error_messages.append(f"Error in radar connection: {e}")
                    break
    
    def process_radar_data(self, radar_rtc_dt, raw_data, line):
        with heart_rate_data_lock:
            if heart_rate_history:
                # Find the heart rate data point with minimal time difference
                min_time_diff = None
                closest_heart_rate_data = None
                time_diff_direction = ''
                for hr_data in heart_rate_history:
                    time_diff = (hr_data['RTC_dt'] - radar_rtc_dt).total_seconds()
                    abs_time_diff = abs(time_diff)
                    if (min_time_diff is None) or (abs_time_diff < min_time_diff):
                        min_time_diff = abs_time_diff
                        closest_heart_rate_data = hr_data
                        time_diff_direction = 'earlier' if time_diff < 0 else 'later'
                if closest_heart_rate_data:
                    bpm = closest_heart_rate_data['BPM']
                    # Store data in cache
                    with self.cache_lock:
                        self.data_cache.append([raw_data, bpm])
                        self.current_test_data_points += 1
                        self.total_data_points += 1
                    # Log the time difference if it exceeds threshold
                    if min_time_diff > 0.15:  # threshold in seconds
                        # Print the line and the time difference, and what was replaced
                        self.error_messages.append(f"Radar data line: {line}")
                        self.error_messages.append(f"Large time difference ({min_time_diff:.3f} s, {time_diff_direction}) between radar and heart rate data.")
                        self.error_messages.append(f"Radar RTC: {radar_rtc_dt}, Closest Heart Rate RTC: {closest_heart_rate_data['RTC_dt']}")
                        self.error_messages.append(f"Used BPM: {bpm} for Radar data.")
                else:
                    # No heart rate data available
                    bpm = ''
                    self.error_messages.append("No heart rate data available to match with radar data.")
                    # Store radar data with empty BPM
                    with self.cache_lock:
                        self.data_cache.append([raw_data, bpm])
                        self.current_test_data_points +=1
                        self.total_data_points += 1
            else:
                # No heart rate data available
                bpm = ''
                self.error_messages.append("No heart rate data available to match with radar data.")
                # Store radar data with empty BPM
                with self.cache_lock:
                    self.data_cache.append([raw_data, bpm])
                    self.current_test_data_points +=1
                    self.total_data_points += 1
        # Update labels
        self.update_labels()
    
    def save_data_to_csv(self):
        try:
            with open(self.csv_file_name, 'a', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                for row in self.data_cache:
                    if row:
                        csv_writer.writerow(row)
        except Exception as e:
            self.error_messages.append(f"Error saving data to CSV: {e}")

    def on_closing(self):
        # Stop the file observer
        self.observer.stop()
        self.observer.join()
        self.destroy()

if __name__ == "__main__":
    app = App()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()
