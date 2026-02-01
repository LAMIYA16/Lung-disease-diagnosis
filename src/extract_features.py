import os
import librosa
import numpy as np
from scipy.fft import fft

# -------- PATH SETUP (WINDOWS SAFE) --------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIO_PATH = os.path.join(BASE_DIR, "data", "audio")

print("BASE_DIR:", BASE_DIR)
print("AUDIO_PATH:", AUDIO_PATH)

if not os.path.exists(AUDIO_PATH):
    print("❌ audio folder NOT FOUND")
    exit()

files = os.listdir(AUDIO_PATH)
print("Files found:", len(files))

X = []
y = []

LABEL_MAP = {
    "N": 0,
    "Normal": 0,
    "Asthma": 1,
    "Heart Failure": 2,
    "COPD": 3,
    "Pleural Effusion": 4
}

def extract_fft_features(signal):
    fft_vals = np.abs(fft(signal))
    fft_vals = fft_vals[:len(fft_vals)//2]
    return [
        np.mean(fft_vals),
        np.std(fft_vals),
        np.max(fft_vals)
    ]

processed = 0

for file in files:
    if not file.endswith(".wav"):
        continue

    print("Processing:", file)

    try:
        disease = file.split("_")[1].split(",")[0].strip()
    except:
        print("❌ Filename parsing failed:", file)
        continue

    if disease not in LABEL_MAP:
        print("❌ Disease not in label map:", disease)
        continue

    file_path = os.path.join(AUDIO_PATH, file)

    try:
        signal, sr = librosa.load(file_path, sr=22050)
    except Exception as e:
        print("❌ Audio load failed:", file, e)
        continue

    features = extract_fft_features(signal)

    X.append(features)
    y.append(LABEL_MAP[disease])
    processed += 1

print("✅ Processed files:", processed)

X = np.array(X)
y = np.array(y)

np.save(os.path.join(BASE_DIR, "data", "X.npy"), X)
np.save(os.path.join(BASE_DIR, "data", "y.npy"), y)

print("🎉 Feature extraction completed successfully")
print("Total samples:", len(X))
