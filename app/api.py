from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import torch
import joblib
import librosa
import numpy as np
import os
from app.model1 import HybridModel
from app.model2 import FinalModel
from app.model3 import FinalModel3

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Define BASE_DIR properly
# -------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # -> DLProj/app
PROJECT_ROOT = os.path.dirname(BASE_DIR)                # -> DLProj

# Load models and scaler at startup
model_paths = {
    "HybridModel": os.path.join(PROJECT_ROOT, "model_checkpoints", "best_model.pth"),
    "FinalModel": os.path.join(PROJECT_ROOT, "model_2_checkpoints", "best_model.pth"),
    "FinalModel3": os.path.join(PROJECT_ROOT, "model_3_checkpoints", "best_model.pth"),
}
scaler_path = os.path.join(PROJECT_ROOT, "standard_scaler.pkl")
scaler = joblib.load(scaler_path)

models = {
    "HybridModel": HybridModel(input_dim=186, lstm_hidden_dim=64, transformer_dim=64, dropout=0.5),
    "FinalModel": FinalModel(num_blocks=4, num_classes=1, embed_size=32, heads=4, forward_expansion=4, dropout=0.5),
    "FinalModel3": FinalModel3(num_blocks=4, num_classes=1, dropout_rate=0.5),
}
for name, model in models.items():
    checkpoint = torch.load(model_paths[name], map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()


def extract_features(audio, sample_rate):
    features = []
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
    features.append(np.mean(mfcc.T, axis=0))
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
    features.append(np.mean(mfcc_delta.T, axis=0))
    features.append(np.mean(mfcc_delta2.T, axis=0))
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sample_rate)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    features.append(np.mean(mel_spec_db.T, axis=0))
    chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
    features.append(np.mean(chroma.T, axis=0))
    spec_contrast = librosa.feature.spectral_contrast(y=audio, sr=sample_rate)
    features.append(np.mean(spec_contrast.T, axis=0))
    return np.concatenate(features)


def reshape_for_models(features, model_type):
    if model_type in ["FinalModel", "FinalModel3"]:
        features = features[:, :147]
        reshaped = features.reshape(1, 7, 7, 3)
        reshaped = np.transpose(reshaped, (0, 3, 1, 2))
        return reshaped
    return features.reshape(1, -1)


def predict_gender(model, scaler, audio, sample_rate, model_type):
    features = extract_features(audio, sample_rate)
    features_scaled = scaler.transform(features.reshape(1, -1))
    features_tensor = torch.tensor(features_scaled, dtype=torch.float32)
    features_tensor = reshape_for_models(features_tensor, model_type)
    with torch.no_grad():
        output = model(features_tensor)
        prediction = output.item()
    if model_type == "HybridModel":
        return "Male" if prediction > 0.5 else "Female"
    elif model_type in ["FinalModel", "FinalModel3"]:
        return "Female" if prediction > 0.5 else "Male"


from pydub import AudioSegment

# ⚠️ Don't hardcode Windows ffmpeg path!
# On Render, ffmpeg will be installed via apt-get in the build step
# so no need to set AudioSegment.converter at all
# (locally, just add ffmpeg to PATH)
# AudioSegment.converter = "ffmpeg"

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    ext = file.filename.lower().split(".")[-1]
    audio_bytes = await file.read()
    import io

    supported_exts = ["wav", "mp3", "flac", "ogg", "m4a", "weba"]
    if ext not in supported_exts:
        return {"error": "Please upload a WAV, MP3, FLAC, OGG, M4A, or WEBA audio file."}
    try:
        if ext in ["m4a", "weba"]:
            audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format=ext)
            wav_io = io.BytesIO()
            audio.export(wav_io, format="wav")
            wav_io.seek(0)
            audio_data, sample_rate = librosa.load(wav_io, sr=None, mono=True)
        elif ext == "wav":
            try:
                audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format="wav")
                wav_io = io.BytesIO()
                audio.export(wav_io, format="wav")
                wav_io.seek(0)
                audio_data, sample_rate = librosa.load(wav_io, sr=None, mono=True)
            except Exception:
                audio_data, sample_rate = librosa.load(io.BytesIO(audio_bytes), sr=None, mono=True)
        else:
            audio_data, sample_rate = librosa.load(io.BytesIO(audio_bytes), sr=None, mono=True)
    except Exception as e:
        return {"error": f"Audio file could not be read: {str(e)}"}

    predictions = []
    for model_name, model in models.items():
        gender = predict_gender(model, scaler, audio_data, sample_rate, model_name)
        predictions.append(gender)

    final_prediction = max(set(predictions), key=predictions.count)
    return {"predictions": predictions, "final": final_prediction}
