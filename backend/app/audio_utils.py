import numpy as np
import librosa
from io import BytesIO


def extract_mfcc_12_from_bytes(audio_bytes: bytes, sr: int = 16000) -> np.ndarray:
    """
    Load raw audio bytes and return a 12‑dimensional MFCC feature vector
    (e.g., mean over time of the 12 MFCC coefficients).
    Adjust to match how you created the features in your training code.
    """
    # Load audio from bytes
    audio_buffer = BytesIO(audio_bytes)
    y, sr = librosa.load(audio_buffer, sr=sr)

    # Compute 12 MFCCs over time
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=12)

    # Aggregate over time dimension, e.g. by taking the mean
    features = mfcc.mean(axis=1)

    return features
