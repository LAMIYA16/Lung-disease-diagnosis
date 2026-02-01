Real-Time Respiratory Disease Classification Using Lung Sound Audio
📌 Project Overview

This project aims to classify respiratory diseases using lung sound recordings and machine learning techniques. The system is capable of identifying Normal and abnormal respiratory conditions such as Asthma, COPD, Heart Failure, and Pleural Effusion.
It also supports real-time audio input using a microphone.

The project is implemented using Python and executed entirely through VS Code.

🎯 Objectives

To extract meaningful features from lung sound audio

To classify respiratory sounds into Normal and disease categories

To perform real-time respiratory sound classification

To build a machine learning–based decision support system (not a diagnostic replacement)

🧠 Dataset Description

Dataset consists of .wav audio files

Disease labels are embedded in the filename

Example filename:

BP20_N,E W,P L L R,22,M.wav

Disease Labels Used
Label in Filename	Meaning
N	Normal
Asthma	Asthma
Heart Failure	Heart Failure
COPD	Chronic Obstructive Pulmonary Disease
Pleural Effusion	Pleural Effusion
🧱 Project Folder Structure
Respiratory_Audio_Classification/
│
├── data/
│   ├── audio/           # All .wav files
│   ├── X.npy            # Extracted features
│   └── y.npy            # Labels
│
├── src/
│   ├── extract_features.py
│   ├── train_model.py
│   └── realtime_test.py
│
├── model/
│   └── classifier.pkl
│
├── requirements.txt
└── README.md

⚙️ Technologies Used

Python 3.10

NumPy

SciPy

Librosa

Scikit-learn

SoundDevice

Joblib

VS Code

🔧 Installation

Clone or download the project folder

Open the folder in VS Code

Install dependencies:

pip install -r requirements.txt

▶️ How to Run the Project (IMPORTANT ORDER)
1️⃣ Place Dataset

Copy all .wav audio files into:

data/audio/

2️⃣ Feature Extraction

This step extracts FFT-based features and saves them.

python src/extract_features.py


Expected output:

Feature extraction completed successfully
Total samples: XXX

3️⃣ Train the Model

This trains a multi-class SVM classifier.

python src/train_model.py


Expected output:

Accuracy: 0.xx
Classification Report:


Model is saved as:

model/classifier.pkl

4️⃣ Real-Time Prediction

Records live audio using microphone and predicts condition.

python src/realtime_test.py


Output example:

Predicted Condition: Normal

🧠 Machine Learning Methodology

Feature Extraction: Fast Fourier Transform (FFT)

Classifier: Support Vector Machine (SVM)

Classification Type: Multi-class classification

Input: Lung sound audio

Output: Respiratory condition

🎤 Real-Time Audio Support

Records 5 seconds of audio

Extracts FFT features

Classifies respiratory condition instantly
