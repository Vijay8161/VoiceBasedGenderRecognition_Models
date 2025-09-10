import os
import pandas as pd
import librosa
import numpy as np
import psutil
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import random

CLIPS_DIR = 'C:/Users/vijay/DLProj/data/cv-corpus-18.0-delta-2024-06-14/en/clips'
TSV_FILE = 'C:/Users/vijay/DLProj/data/cv-corpus-18.0-delta-2024-06-14/en/other.tsv'
OUTPUT_DIR = './data/preprocessed'
CHECKPOINT_FILE = 'C:/Users/vijay/DLProj/app/processing_checkpoint.txt'

def get_system_resources():
    memory_info = psutil.virtual_memory()
    available_memory = memory_info.available
    num_cores = cpu_count()
    return available_memory, num_cores

def calculate_batch_size(available_memory, file_size_estimate=200 * 1024):
    batch_size = available_memory // (file_size_estimate * 2)
    return max(1, int(batch_size))
    # print("Trying to load TSV file from:", TSV_FILE)
def load_labels(tsv_file, limit=7000):
    data = pd.read_csv(tsv_file, sep='\t', nrows=limit)
    file_gender_map = dict(zip(data['path'], data['gender']))
    return file_gender_map

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

def add_noise(audio, noise_factor=0.005):
    noise = np.random.randn(len(audio))
    return audio + noise_factor * noise

def adjust_volume(audio, factor=1.5):
    return audio * factor

def random_gain(audio, gain_range=(0.8, 1.2)):
    gain = random.uniform(*gain_range)
    return audio * gain

def reverse_audio(audio):
    return audio[::-1]

def speed_change(audio, speed_factor=0.4):
    indices = np.round(np.arange(0, len(audio), speed_factor))
    indices = indices[indices < len(audio)].astype(int)
    return audio[indices]

import time

#actual method
# def preprocess_file(file_name, gender, augment=False):
#     try:
#         start = time.time()
#         file_path = os.path.join(CLIPS_DIR, file_name)
#         audio, sample_rate = librosa.load(file_path, sr=None)

#         if augment:
#             augmentations = [
#                 add_noise,
#                 adjust_volume,
#                 random_gain,
#                 reverse_audio,
#                 speed_change,
#             ]
#             for aug in random.sample(augmentations, k=random.randint(1, 3)):
#                 audio = aug(audio)

#         features = extract_features(audio, sample_rate)
#         took = time.time() - start
#         if took > 3:
#             print(f"  Slow file: {file_name} took {took:.2f}s to process")
#         return features, gender
#     except Exception as e:
#         print(f" Error processing {file_name}: {e}")
#         return None, None

#change 1
# def preprocess_file(file_name, gender, augment=False):
#     try:
#         file_path = os.path.join(CLIPS_DIR, file_name)
#         audio, sample_rate = librosa.load(file_path, sr=None)
#         if augment:
#             augmentations = [
#                 add_noise,
#                 adjust_volume,
#                 random_gain,
#                 reverse_audio,
#                 speed_change,
#             ]
#             for aug in random.sample(augmentations, k=random.randint(1, 3)):
#                 audio = aug(audio)
#         features = extract_features(audio, sample_rate)
#         return features, gender
#     except Exception as e:
#         print(f"Error processing {file_name}: {e}")
#         return None, None

def preprocess_file(file_name, gender, augment=False):
    try:
        start = time.time()
        file_path = os.path.join(CLIPS_DIR, file_name)
        audio, sample_rate = librosa.load(file_path, sr=None)

        # Skip very short audio files
        if len(audio) < 2048 or librosa.get_duration(y=audio, sr=sample_rate) < 1.0:
            raise ValueError(f"Audio too short for processing: {file_name}")

        if augment:
            original_audio = audio.copy()
            augmentations = [
                add_noise,
                adjust_volume,
                random_gain,
                reverse_audio,
                speed_change
            ]
            selected_augs = random.sample(augmentations, k=random.randint(1, 1)) # random.randint(1, 3)

            for aug in selected_augs:
                try:
                    audio = aug(audio)
                except Exception as e:
                    print(f"  Failed augmentation {aug.__name__} on {file_name}: {e}")
                    audio = original_audio  # fallback to original if augmentation fails
                    break

            # Check again after augmentation
            if len(audio) < 2048 or librosa.get_duration(y=audio, sr=sample_rate) < 1.0:
                raise ValueError(f"Audio too short after augmentation: {file_name}")

        features = extract_features(audio, sample_rate)
        took = time.time() - start
        if took > 3:
            print(f"  Slow file: {file_name} took {took:.2f}s to process")

        return features, gender

    except Exception as e:
        print(f"Error processing {file_name} (augment={augment}): {e}")
        return None, None

def save_data(X, y, output_file):
    np.savez(output_file, X=X, y=y)
    print(f"Data saved to {output_file}")

def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r') as f:
            last_processed_idx = int(f.read().strip())
    else:
        last_processed_idx = 0
    return last_processed_idx

def save_checkpoint(last_processed_idx):
    with open(CHECKPOINT_FILE, 'w') as f:
        f.write(str(last_processed_idx))

import time

def get_adjusted_pool_size():
    target_cpu_usage = 90
    current_cpu_usage = psutil.cpu_percent(interval=1)
    available_cpu_usage = max(target_cpu_usage - current_cpu_usage, 10)
    max_pool_size = cpu_count()
    adjusted_pool_size = int((available_cpu_usage / 100) * max_pool_size)
    return max(1, min(adjusted_pool_size, max_pool_size))

def main():
    file_gender_map = load_labels(TSV_FILE)
    last_processed_idx = load_checkpoint()
    available_memory, _ = get_system_resources()
    batch_size = calculate_batch_size(available_memory)
    print(f"Calculated batch size: {batch_size}")
    file_names = list(file_gender_map.keys())
    genders = list(file_gender_map.values())
    total_files = min(15000, len(file_names))
    print(f"Total files to be processed: {total_files}")

    # while last_processed_idx < total_files:
    while last_processed_idx < total_files:
        pool_size = get_adjusted_pool_size()
        print(f"Adjusted pool size: {pool_size}")

        with Pool(processes=pool_size) as pool:
            end_idx = min(last_processed_idx + batch_size, total_files)
            batch_data = list(zip(file_names[last_processed_idx:end_idx], genders[last_processed_idx:end_idx]))
            timestamp = int(time.time())
            batch_id = f"batch_{timestamp}"

            with tqdm(total=len(batch_data), desc="Processing Augmented Features") as pbar_augmented:
                augmented_features = [
                    pool.apply_async(
                        preprocess_file,
                        (file_name, gender, True),
                        callback=lambda _: pbar_augmented.update(1),
                    )
                    for file_name, gender in batch_data
                ]
                augmented_features = [a.get() for a in augmented_features]

            with tqdm(total=len(batch_data), desc="Processing Regular Features") as pbar_regular:
                regular_features = [
                    pool.apply_async(
                        preprocess_file,
                        (file_name, gender, False),
                        callback=lambda _: pbar_regular.update(1),
                    )
                    for file_name, gender in batch_data
                ]
                regular_features = [r.get() for r in regular_features]

            # with tqdm(total=len(batch_data), desc="Processing Augmented Features") as pbar_augmented:
            #     augmented_features = [
            #         pool.apply_async(
            #             preprocess_file,
            #             (file_name, gender, True),
            #             callback=lambda _: pbar_augmented.update(1),
            #         )
            #         for file_name, gender in batch_data
            #     ]
            #     augmented_features = [a.get() for a in augmented_features]

            X_regular = np.array([item for item, _ in regular_features if item is not None])
            y_regular = np.array([gender for _, gender in regular_features if gender is not None])
            X_augmented = np.array([item for item, _ in augmented_features if item is not None])
            y_augmented = np.array([gender for _, gender in augmented_features if gender is not None])
            # X_augmented = np.array([item for item, _ in regular_features if item is not None])
            # y_augmented = np.array([gender for _, gender in regular_features if gender is not None])
            # print(X_regular)
            # print(y_regular)
            if not os.path.exists(OUTPUT_DIR):
                os.makedirs(OUTPUT_DIR)

            save_data(X_regular, y_regular, os.path.join(OUTPUT_DIR, f"features_{batch_id}.npz"))
            save_data(X_augmented, y_augmented, os.path.join(OUTPUT_DIR, f"augmented_features_{batch_id}.npz"))

            last_processed_idx = end_idx
            save_checkpoint(last_processed_idx)
            print(f"Processed {end_idx}/{total_files} files.")

if __name__ == "__main__":
    main()
