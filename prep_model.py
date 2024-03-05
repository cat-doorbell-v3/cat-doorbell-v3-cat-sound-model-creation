"""
This changes .wav files to conform to:
- 30ms window frame
- 20ms window stride
- 16KHz sample rate
- 16-bit signed PCM data
- single channel (mono)

"""
import random
import shutil
import sys

import librosa
import numpy as np
import soundfile as sf
from pydub import AudioSegment
from pydub.silence import detect_nonsilent


def conform_process_audio_file(file_path):
    # Load the file (automatically resampled to 16KHz and converted to mono)
    data, sr = librosa.load(file_path, sr=16000, mono=True)

    # Ensure the data is in 16-bit signed PCM format
    data = (data * 32767).astype('int16')

    # Save the processed file in place
    sf.write(file_path, data, sr, subtype='PCM_16')
    print(f"Processed {file_path}")


def conform_process_directory(directory):
    # Walk through the directory
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.wav'):
                file_path = os.path.join(root, file)

                # Process the file
                conform_process_audio_file(file_path)


def trim_initial_silence(wav_path, min_silence_len=1000, silence_thresh=-40):
    """
    Removes initial silence from a WAV file and overwrites the original file with the trimmed audio.

    Args:
    wav_path (str): Path to the WAV file.
    min_silence_len (int): Minimum length of a silence to be considered for trimming (in milliseconds).
    silence_thresh (int): The upper bound for what's considered silence (in dB). Default is -40 dB.
    """
    sound = AudioSegment.from_wav(wav_path)
    non_silents = detect_nonsilent(sound, min_silence_len=min_silence_len, silence_thresh=silence_thresh)

    if non_silents:
        start_trim = non_silents[0][0]
        trimmed_sound = sound[start_trim:]
        trimmed_sound.export(wav_path, format="wav")
        print(f"Trimmed and saved {wav_path}")


def trim_process_directory(input_dir, min_silence_len=1000, silence_thresh=-40):
    """
    Processes all WAV files in a directory and its subdirectories to remove initial silence, in place.

    Args:
    input_dir (str): Directory containing WAV files to process in place.
    min_silence_len (int): Minimum length of silence in milliseconds for trimming.
    silence_thresh (int): Silence threshold in dB.
    """
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".wav"):
                wav_path = os.path.join(root, file)
                trim_initial_silence(wav_path, min_silence_len, silence_thresh)


def add_noise(y, noise_level=0.005):
    noise = np.random.randn(len(y))
    return y + noise_level * noise


def time_stretch(y, rate=0.8):
    return librosa.effects.time_stretch(y, rate=rate)


def random_crop(y, sr, duration=1):
    if len(y) > sr * duration:
        start = np.random.randint(len(y) - sr * duration)
        return y[start:start + sr * duration]
    return y


def change_volume(y, volume_change_dB=-6):
    return librosa.db_to_amplitude(volume_change_dB) * y


def pitch_shift(y, sr, n_steps=2.5):
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)


def augment_audio(file_path, output_dir):
    y, sr = librosa.load(file_path)

    # List of augmentation functions to randomly choose from
    augmentations = [add_noise, time_stretch, random_crop, pitch_shift]
    random.shuffle(augmentations)

    # Apply two random augmentations
    for aug in augmentations[:2]:
        y = aug(y, sr)

    # Apply volume change (not random)
    y = change_volume(y)

    # Save the augmented audio
    base, ext = os.path.splitext(os.path.basename(file_path))
    output_file_path = os.path.join(output_dir, f"{base}_augmented{ext}")
    sf.write(output_file_path, y, sr)
    print(f"Augmented file saved to {output_file_path}")


def augment_process_directory(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.wav'):
                file_path = os.path.join(root, file)
                augment_audio(file_path, root)


import os
import glob


def count_wav_files_in_subdirs(parent_dir):
    # Check if the specified path is indeed a directory
    if not os.path.isdir(parent_dir):
        print(f"The provided path '{parent_dir}' is not a directory.")
        return

    # Iterate through all items in the parent directory
    for subdir in os.listdir(parent_dir):
        subdir_path = os.path.join(parent_dir, subdir)

        # Check if the current item is a directory
        if os.path.isdir(subdir_path):
            # Use glob to count .wav files in the current subdirectory
            wav_files = glob.glob(os.path.join(subdir_path, '*.wav'))
            num_wav_files = len(wav_files)

            # Print the subdirectory and the count of .wav files
            print(f"Subdirectory: '{subdir}', .wav files: {num_wav_files}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: script_name.py <directory>")
        sys.exit(1)

    directory = sys.argv[1]

    print("First, count the number of files in the directory")
    count_wav_files_in_subdirs(directory)

    print(f"Conforming directory {directory}")
    conform_process_directory(directory)

    print(f"Trimming directory {directory}")
    trim_process_directory(directory)

    print(f"Augmenting directory {directory}")
    augment_process_directory(directory)

    print("Re-count the files")
    count_wav_files_in_subdirs(directory)

    parent_dir = os.path.abspath(os.path.join(directory, os.pardir))
    base_dir = os.path.basename(os.path.normpath(directory))
    zip_file_path = os.path.join(parent_dir, base_dir)
    shutil.make_archive(zip_file_path, 'zip', parent_dir, base_dir)
    print(f"Created zip file: {zip_file_path}.zip")

    print("Done")
