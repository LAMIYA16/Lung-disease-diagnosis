import os
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import sys

# --------------------------
# Determine project root
# --------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # one level above src/
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "model")

# Ensure model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

# --------------------------
# File paths
# --------------------------
X_FILE = os.path.join(DATA_DIR, "X.npy")
Y_FILE = os.path.join(DATA_DIR, "y.npy")
MODEL_FILE = os.path.join(MODEL_DIR, "classifier.pkl")

# --------------------------
# Check if data files exist
# --------------------------
if not os.path.isfile(X_FILE) or not os.path.isfile(Y_FILE):
    print(f"Error: Could not find X.npy or y.npy in '{DATA_DIR}'.")
    print(f"Expected files:\n - {X_FILE}\n - {Y_FILE}")
    sys.exit(1)

# --------------------------
# Load data
# --------------------------
X = np.load(X_FILE)
y = np.load(Y_FILE)

# --------------------------
# Split data
# --------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --------------------------
# Train SVM model
# --------------------------
model = SVC(kernel="rbf", probability=True)
model.fit(X_train, y_train)

# --------------------------
# Evaluate model
# --------------------------
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# --------------------------
# Save model
# --------------------------
joblib.dump(model, MODEL_FILE)
print(f"Model saved as '{MODEL_FILE}'")
