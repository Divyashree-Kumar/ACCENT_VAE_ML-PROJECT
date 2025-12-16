from pathlib import Path
import numpy as np
import joblib

# backend/app
BASE_DIR = Path(__file__).resolve().parent

# Load scaler and VAE+SVM model saved by train_api_model.py
scaler = joblib.load(BASE_DIR / "vae_scaler.pkl")
svm_vae = joblib.load(BASE_DIR / "vae_accent_model.pkl")

# These are your accent classes (same as in README)
ACCENT_LABELS = ["ES", "FR", "GE", "IT", "UK", "US"]


def predict_from_features(features: np.ndarray) -> str:
    """
    Take a 12‑D MFCC feature vector and return the predicted accent label.
    """
    # Ensure numpy array and shape (1, 12)
    x = np.asarray(features, dtype=float).reshape(1, -1)

    # Scale features with the trained scaler
    x_scaled = scaler.transform(x)

    # Predict label (strings, since y_vae_train was string labels)
    pred = svm_vae.predict(x_scaled)[0]

    # Just return as string
    return str(pred)
