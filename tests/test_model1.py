import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import numpy as np
import joblib
import librosa
import sounddevice as sd
import wavio
from app.model1 import HybridModel
from app.model2 import FinalModel
from app.model3 import FinalModel3

def load_model(model_path, model_class, *args, **kwargs):
    model = model_class(*args, **kwargs)
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    return model

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

def record_audio(duration, filename):
    print(f"Recording for {duration} seconds...")
    samplerate = 44100
    audio = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=1, dtype='float64')
    sd.wait()
    wavio.write(filename, audio, samplerate, sampwidth=3)
    print(f"Recording saved to {filename}")

def predict_gender(model, scaler, audio_file, model_type):
    audio, sample_rate = librosa.load(audio_file, sr=None)
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



def main():
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    # model_paths = {
    #     "HybridModel": 'model_checkpoints/best_model.pth',
    #     "FinalModel": 'model_2_checkpoints/best_model.pth',
    #     "FinalModel3": 'model_3_checkpoints/best_model.pth',
    # }
    model_paths = {
        "HybridModel": os.path.join(BASE_DIR, 'model_checkpoints', 'best_model.pth'),
        "FinalModel": os.path.join(BASE_DIR, 'model_2_checkpoints', 'best_model.pth'),
        "FinalModel3": os.path.join(BASE_DIR, 'model_3_checkpoints', 'best_model.pth'),
    }

    models = {
        "HybridModel": load_model(model_paths["HybridModel"], HybridModel, input_dim=186, lstm_hidden_dim=64, transformer_dim=64, dropout=0.5),
        "FinalModel": load_model(model_paths["FinalModel"], FinalModel, num_blocks=4, num_classes=1, embed_size=32, heads=4, forward_expansion=4, dropout=0.5),
        "FinalModel3": load_model(model_paths["FinalModel3"], FinalModel3, num_blocks=4, num_classes=1, dropout_rate=0.5),
    }

    # try:
    #     scaler = joblib.load('standard_scaler.pkl')
    # except FileNotFoundError:
    #     print("Scaler file not found. Please ensure that 'standard_scaler.pkl' exists.")
    #     return
    try:
        scaler = joblib.load(os.path.join(BASE_DIR, 'standard_scaler.pkl'))
    except FileNotFoundError:
        print("Scaler file not found. Please ensure that 'standard_scaler.pkl' exists.")
        return

    audio_file = "recording.wav"
    audio_file = os.path.join(BASE_DIR, "recording.wav")
    record_audio(duration=5, filename=audio_file)

    predictions = []

    for model_name, model in models.items():
        gender = predict_gender(model, scaler, audio_file, model_name)
        predictions.append(gender)
        print(f"Predicted Gender from {model_name}: {gender}")
    
    # Majority voting
    final_prediction = max(set(predictions), key=predictions.count)
    print(f"\nFinal Gender Prediction (Majority Vote): {final_prediction}")

if __name__ == '__main__':
    main()