from tensorflow.keras.models import load_model

model_path = r"R:\OHG\Cardiac_Respiratory_Phantom\Blood Pressure Processing\Models\34_model.h5"
try:
    model = load_model(model_path)
    print("Model loaded successfully")
except OSError as e:
    print(f"Failed to load model: {e}")
