from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from .model import predict_from_features
from .audio_utils import extract_mfcc_12_from_bytes


app = FastAPI()

# Allow your static HTML (opened from file or localhost) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # You can restrict this later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def read_root():
    return {"message": "Accent VAE API is running"}


@app.post("/predict-accent")
async def predict_accent(file: UploadFile = File(...)):
    """
    Receive an uploaded audio file, extract 12-D MFCC features,
    run the VAE+classifier pipeline, and return the predicted accent.
    """
    # Read uploaded audio into memory
    audio_bytes = await file.read()

    # Convert raw bytes -> 12 MFCC features (np.ndarray or list-like)
    features = extract_mfcc_12_from_bytes(audio_bytes)

    # Get accent label (string) from your trained pipeline
    accent_label = predict_from_features(features)

    # Frontend expects: accent, confidence, filename
    return {
        "accent": accent_label,
        "confidence": 1.0,      # replace with real confidence later if you add it
        "filename": file.filename,
    }
