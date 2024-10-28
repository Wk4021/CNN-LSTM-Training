import time
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.common.by import By
import socket
import threading

# Server configuration
HOST = '10.54.9.92'
#'10.54.9.92'  # Replace with your server IP
PORT = 65433         # Replace with your server port

# URL to open
URL = 'https://pulsoid.net/widget/view/0950e10c-7d15-4b7a-87b9-6a0f58f631c8'

# Create a TCP/IP socket
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

def connect_to_server():
    try:
        client_socket.connect((HOST, PORT))
        print(f"Connected to server at {HOST}:{PORT}")
    except Exception as e:
        print(f"Could not connect to server: {e}")
        client_socket.close()
        exit(1)

def start_browser():
    # Set up Chrome options for headless browsing
    chrome_options = Options()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--disable-gpu')
    # Optional: Uncomment if you have issues with SSL certificates
    # chrome_options.add_argument('--ignore-certificate-errors')

    # Path to chromedriver executable if not in PATH
    # driver = webdriver.Chrome(executable_path='/path/to/chromedriver', options=chrome_options)
    driver = webdriver.Chrome(options=chrome_options)
    return driver

def get_heart_rate(driver):
    try:
        # Locate the heart rate element using the new method
        heart_rate_element = driver.find_element(By.ID, 'heartRate')
        # Extract the text content
        heart_rate_text = heart_rate_element.text.strip()
        # Extract numeric value
        heart_rate = ''.join(filter(str.isdigit, heart_rate_text))
        if heart_rate:
            return int(heart_rate)
        else:
            return None
    except NoSuchElementException:
        print("Heart rate element not found on the page.")
        return None

def main():
    connect_to_server()
    driver = start_browser()
    driver.get(URL)
    print("Webpage loaded.")
    try:
        while True:
            heart_rate = get_heart_rate(driver)
            if heart_rate is not None:
                # Get current UTC time
                #utc_time = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
                utc_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                # Format data as "UTC,BPM" (comma-separated)
                data_to_send = f"{utc_time},{heart_rate}\n"
                try:
                    client_socket.sendall(data_to_send.encode('utf-8'))
                    #print(f"Sent data: {data_to_send.strip()}")
                except Exception as e:
                    print(f"Error sending data: {e}")
                    client_socket.close()
                    break
            else:
                print("Failed to get heart rate. Skipping this iteration.")
            time.sleep(0.1)  # Sleep for 100 milliseconds
    except KeyboardInterrupt:
        print("Script interrupted by user.")
    finally:
        driver.quit()
        client_socket.close()
        print("Browser and socket closed.")

if __name__ == "__main__":
    main()