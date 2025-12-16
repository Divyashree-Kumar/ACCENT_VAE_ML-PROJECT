from pathlib import Path
import numpy as np
import joblib

BASE_DIR = Path(__file__).resolve().parent

# Load scaler and VAE+SVM model saved by your training script
scaler = joblib.load(BASE_DIR / "vae_scaler.pkl")
svm_vae = joblib.load(BASE_DIR / "vae_accent_model.pkl")


def predict_from_features(features: np.ndarray) -> str:
    """
    Take a 12‑D MFCC feature vector and return the predicted accent label.
    """
    x = np.asarray(features, dtype=float).reshape(1, -1)

    # scale
    x_scaled = scaler.transform(x)

    # predict string label (y_train were strings)
    pred = svm_vae.predict(x_scaled)[0]
    return str(pred)
