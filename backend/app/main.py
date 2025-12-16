from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from .model import predict_from_features
from .audio_utils import extract_mfcc_12_from_bytes

app = FastAPI()

# CORS so index.html can call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # later you can restrict this to your frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"message": "Accent VAE API is running"}


@app.post("/predict-accent")
async def predict_accent(file: UploadFile = File(...)):
    # Read uploaded audio
    audio_bytes = await file.read()

    # Extract 12‑D MFCC features from the audio bytes
    features = extract_mfcc_12_from_bytes(audio_bytes)

    # Get model prediction (e.g., accent label and/or probabilities)
    result = predict_from_features(features)

    return {"prediction": result}
