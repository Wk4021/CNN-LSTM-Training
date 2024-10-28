import numpy as np
import tensorflow as tf
from tkinter import Tk, Button, Frame, filedialog, Label, Entry, Text, Scrollbar
from tensorflow.keras.models import load_model

# Function to convert hex to bytes
def hex_to_bytes(hex_str):
    return bytes.fromhex(hex_str)

# Function to load a pre-trained model
def load_model():
    global model
    model_path = filedialog.askopenfilename(title="Select a trained model", filetypes=(("H5 files", "*.h5"),))
    if model_path:
        model = tf.keras.models.load_model(model_path)
        model_name_label.config(text=f'Model Loaded: {model_path}')
        print(f'Model {model_path} loaded successfully.')
        print(model.input_shape)  # Add this line after loading the model


# Function to predict from RawRadar hex input
def predict_from_hex():
    global model
    if not model:
        print("Please load a model first.")
        return
    
    raw_hex = hex_input.get("1.0", "end-1c").strip()
    if not raw_hex:
        print("Please enter RawRadar hex data.")
        return

    try:
        # Convert hex to bytes
        byte_data = hex_to_bytes(raw_hex)

        # Convert bytes to a numpy array and reshape it for the model
        byte_array = np.frombuffer(byte_data, dtype=np.uint8).reshape(1, 1030, 1)  # Adjusted shape


        # Make predictions
        predictions = model.predict(byte_array.astype(np.float32))
        bpm, rr, psi = predictions[0][0], predictions[0][1], predictions[0][2]

        # Print the predictions
        print(f'Predicted BPM: {bpm:.2f}, RR: {rr:.2f}, PSI: {psi:.2f}')

    except Exception as e:
        print(f"Error in prediction: {e}")

# GUI Setup using tkinter
def setup_gui():
    global model_name_label, hex_input, model

    model = None

    root = Tk()
    root.title("Radar Data Predictor")

    frame = Frame(root)
    frame.pack(pady=10)

    # Load Model Button
    load_model_button = Button(frame, text="Load Model", command=load_model)
    load_model_button.pack()

    # Input for Raw Radar hex data
    Label(frame, text="Enter RawRadar Hex Data:").pack()
    hex_input = Text(frame, height=10, width=50)
    hex_input.pack()

    # Predict Button
    predict_button = Button(frame, text="Predict", command=predict_from_hex)
    predict_button.pack()

    # Model name label
    model_name_label = Label(frame, text="Model Loaded: None")
    model_name_label.pack(pady=5)

    root.mainloop()

if __name__ == '__main__':
    setup_gui()
