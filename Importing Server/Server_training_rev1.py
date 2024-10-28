import socket
import threading
import time
from datetime import datetime

# Server configuration
HOST = '0.0.0.0'  # Listen on all network interfaces
RADAR_PORT = 65432      # Port for radar data
PRESSURE_PORT = 65433   # Port for pressure data
DATA_CLIENT_PORT = 65435  # Port for data clients (training script)

# Global variables for data storage
last_graph_data = {'RTC': None, 'PSI': None, 'Voltage': None}
last_radar_data = {'RTC': None, 'UBPM': None, 'URR': None, 'RawRadar': None}

# Locks for thread-safe data access
graph_data_lock = threading.Lock()
radar_data_lock = threading.Lock()

# List to keep track of connected data clients
data_clients = []

def handle_pressure_connection(client_socket, client_address):
    global last_graph_data
    print(f"Accepted pressure connection from {client_address}")
    buffer = ''
    with client_socket:
        while True:
            data = client_socket.recv(1024)  # Buffer size
            if not data:
                break
            buffer += data.decode('utf-8')
            while '\n' in buffer:
                line, buffer = buffer.split('\n', 1)
                line = line.strip()
                if not line:
                    continue
                try:
                    parts = line.split(',')
                    if len(parts) == 3:
                        # Data format: RTC,PSI,Voltage
                        real_time_clock_str, pressure_str, voltage_str = parts
                        pressure_psi = float(pressure_str)
                        voltage = float(voltage_str)
                        with graph_data_lock:
                            last_graph_data['RTC'] = real_time_clock_str.strip()
                            last_graph_data['PSI'] = pressure_psi
                            last_graph_data['Voltage'] = voltage
                        # Attempt to sync and print data
                        sync_and_print_data()
                    else:
                        print(f"Invalid pressure data received: {line}")
                except ValueError as e:
                    print(f"Invalid pressure data: {line}, error: {e}")

def handle_radar_connection(client_socket, client_address):
    global last_radar_data
    print(f"Accepted radar connection from {client_address}")
    buffer = ''
    with client_socket:
        while True:
            data = client_socket.recv(1024)  # Buffer size
            if not data:
                break
            buffer += data.decode('utf-8')
            while '\n' in buffer:
                line, buffer = buffer.split('\n', 1)
                line = line.strip()
                if not line:
                    continue
                try:
                    # Extract RTC separately (first 23 characters)
                    rtc_part = line[:23]  # Assuming RTC has a fixed length of 23 characters
                    remaining_data = line[24:].strip()
                    
                    parts = remaining_data.split(' ', 3)
                    if len(parts) == 3:
                        # Data format: UBPM URR RawRadarData
                        ubpm_str, urr_str, raw_data = parts
                        
                        # Check if UBPM and URR are both zero
                        if ubpm_str == "0" and urr_str == "0":
                            # Skip further processing and don't print anything if both UBPM and URR are zero
                            continue
                        
                        with radar_data_lock:
                            last_radar_data['RTC'] = rtc_part
                            last_radar_data['UBPM'] = ubpm_str.strip()
                            last_radar_data['URR'] = urr_str.strip()
                            last_radar_data['RawRadar'] = raw_data.strip()
                        # Attempt to sync and print data
                        sync_and_print_data()
                    else:
                        print(f"Invalid radar data received: {line}")
                except ValueError as e:
                    print(f"Invalid radar data: {line}, error: {e}")

def sync_and_print_data():
    global last_graph_data, last_radar_data, data_clients
    with graph_data_lock, radar_data_lock:
        if last_graph_data['RTC'] and last_radar_data['RTC']:
            try:
                # Check if both RTC timestamps are complete (i.e., they include time)
                if len(last_graph_data['RTC']) < 23 or len(last_radar_data['RTC']) < 23:
                    print(f"Incomplete RTC data. Pressure RTC: {last_graph_data['RTC']}, Radar RTC: {last_radar_data['RTC']}")
                    return

                # Parse RTC timestamps with milliseconds
                graph_rtc_dt = datetime.strptime(last_graph_data['RTC'], '%Y-%m-%d %H:%M:%S.%f')
                radar_rtc_dt = datetime.strptime(last_radar_data['RTC'], '%Y-%m-%d %H:%M:%S.%f')

                # Calculate time difference in milliseconds
                time_diff_ms = abs((graph_rtc_dt - radar_rtc_dt).total_seconds() * 1000)

                if time_diff_ms < 100:  # If the difference is less than 100 milliseconds
                    output_line = f"RTC: {last_radar_data['RTC']} UBPM: {last_radar_data['UBPM']} URR: {last_radar_data['URR']} " \
                                  f"PSI: {last_graph_data['PSI']} Voltage: {last_graph_data['Voltage']} " \
                                  f"RawRadar: {last_radar_data['RawRadar']}"
                    print(output_line)

                    # Send data to connected data clients
                    for client in data_clients.copy():
                        try:
                            client.sendall((output_line + '\n').encode('utf-8'))
                        except Exception as e:
                            print(f"Error sending data to client: {e}")
                            data_clients.remove(client)

                    # Reset data after successful print
                    last_graph_data = {'RTC': None, 'PSI': None, 'Voltage': None}
                    last_radar_data = {'RTC': None, 'UBPM': None, 'URR': None, 'RawRadar': None}
                else:
                    # RTCs are not synchronized within threshold
                    print(f"RTC times differ by {time_diff_ms:.3f} milliseconds")
                    last_graph_data['RTC'] = None
                    last_radar_data['RTC'] = None
            except ValueError as e:
                print(f"Error parsing RTC timestamps: {e}")
                print(f"Pressure RTC: {last_graph_data['RTC']}")
                print(f"Radar RTC: {last_radar_data['RTC']}")
                # Reset RTC data to avoid repeated errors
                last_graph_data['RTC'] = None
                last_radar_data['RTC'] = None

def start_pressure_server():
    pressure_server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    pressure_server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    pressure_server_socket.bind((HOST, PRESSURE_PORT))
    pressure_server_socket.listen()
    print(f"Pressure server listening on {HOST}:{PRESSURE_PORT}")
    while True:
        client_sock, client_addr = pressure_server_socket.accept()
        client_handler = threading.Thread(
            target=handle_pressure_connection,
            args=(client_sock, client_addr),
            daemon=True
        )
        client_handler.start()

def start_radar_server():
    radar_server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    radar_server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    radar_server_socket.bind((HOST, RADAR_PORT))
    radar_server_socket.listen()
    print(f"Radar server listening on {HOST}:{RADAR_PORT}")
    while True:
        client_sock, client_addr = radar_server_socket.accept()
        client_handler = threading.Thread(
            target=handle_radar_connection,
            args=(client_sock, client_addr),
            daemon=True
        )
        client_handler.start()

def handle_data_client(client_socket, client_address):
    global data_clients
    print(f"Accepted data client connection from {client_address}")
    with client_socket:
        data_clients.append(client_socket)
        try:
            while True:
                # Keep the connection open
                time.sleep(1)
        except Exception as e:
            print(f"Data client {client_address} disconnected: {e}")
        finally:
            if client_socket in data_clients:
                data_clients.remove(client_socket)

def start_data_client_server():
    data_server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    data_server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    data_server_socket.bind((HOST, DATA_CLIENT_PORT))
    data_server_socket.listen()
    print(f"Data client server listening on {HOST}:{DATA_CLIENT_PORT}")
    while True:
        client_sock, client_addr = data_server_socket.accept()
        client_handler = threading.Thread(
            target=handle_data_client,
            args=(client_sock, client_addr),
            daemon=True
        )
        client_handler.start()

if __name__ == "__main__":
    # Start all servers
    pressure_server_thread = threading.Thread(target=start_pressure_server, daemon=True)
    radar_server_thread = threading.Thread(target=start_radar_server, daemon=True)
    data_client_server_thread = threading.Thread(target=start_data_client_server, daemon=True)
    pressure_server_thread.start()
    radar_server_thread.start()
    data_client_server_thread.start()

    # Keep the script running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Server shutting down.")
