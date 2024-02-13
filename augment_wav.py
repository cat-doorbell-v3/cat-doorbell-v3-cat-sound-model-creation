import os

import librosa
import numpy as np
import soundfile as sf


def add_noise(data, noise_factor=0.005):
    noise = np.random.randn(len(data))
    augmented_data = data + noise_factor * noise
    return augmented_data.astype(type(data[0]))


def shift_pitch(data, sampling_rate, pitch_factor=5):
    return librosa.effects.pitch_shift(y=data, sr=sampling_rate, n_steps=pitch_factor)


def change_speed(data, speed_factor=1.25):
    return np.interp(np.arange(0, len(data), speed_factor), np.arange(0, len(data)), data)


def augment_and_save(file_path, target_dir, sampling_rate):
    data, _ = librosa.load(file_path, sr=sampling_rate)
    noise_data = add_noise(data)
    pitch_shifted_data = shift_pitch(data, sampling_rate)
    speed_changed_data = change_speed(data)

    base_name = os.path.splitext(os.path.basename(file_path))[0]
    sf.write(os.path.join(target_dir, f"{base_name}_original.wav"), data, sampling_rate)
    sf.write(os.path.join(target_dir, f"{base_name}_noise.wav"), noise_data, sampling_rate)
    sf.write(os.path.join(target_dir, f"{base_name}_pitch.wav"), pitch_shifted_data, sampling_rate)
    sf.write(os.path.join(target_dir, f"{base_name}_speed.wav"), speed_changed_data, sampling_rate)


def copy_and_augment_directory(source_dir, target_dir, sampling_rate=22050):
    for root, dirs, files in os.walk(source_dir):
        rel_path = os.path.relpath(root, source_dir)
        current_target_dir = os.path.join(target_dir, rel_path)
        os.makedirs(current_target_dir, exist_ok=True)

        for file in files:
            if file.lower().endswith('.wav'):
                file_path = os.path.join(root, file)
                augment_and_save(file_path, current_target_dir, sampling_rate)


# Usage
source_dir = './CAT_SOUND_DB_SAMPLES_WAV'
target_dir = './CAT_SOUND_DB_SAMPLES_AUGMENTED'
copy_and_augment_directory(source_dir, target_dir)
