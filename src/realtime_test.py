import sounddevice as sd
import numpy as np
from scipy.fft import fft
import joblib
import os

# Path to your model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "../model/classifier.pkl")

# Check if the file exists
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")

# Load the model
model = joblib.load(MODEL_PATH)

LABELS = {
    0: "Normal",
    1: "Asthma",
    2: "Heart Failure",
    3: "COPD",
    4: "Pleural Effusion"
}

fs = 22050
duration = 5  # seconds

print("Recording respiratory sound for 5 seconds...")
audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
sd.wait()

audio = audio.flatten()

fft_vals = np.abs(fft(audio))
fft_vals = fft_vals[:len(fft_vals)//2]

features = [
    np.mean(fft_vals),
    np.std(fft_vals),
    np.max(fft_vals)
]

prediction = model.predict([features])[0]
print("Predicted Condition:", LABELS[prediction])
