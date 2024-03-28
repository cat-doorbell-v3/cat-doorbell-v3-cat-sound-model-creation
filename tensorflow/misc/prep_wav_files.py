"""
This changes .wav files to conform to:
- 30ms window frame
- 20ms window stride
- 16KHz sample rate
- 16-bit signed PCM data
- single channel (mono)

"""
import glob
import os
import wave

import librosa
import soundfile as sf
import tensorflow as tf
from pydub import AudioSegment

YAMNET_MODEL_DIR = '/Users/tennis/sound-library/yamnet/archive'
YAMNET_CAT_SCORE_THRESHOLD = 0.8  # 80% certainty
YAMNET_CAT_MEOW_SOUND_INDEX = 78


def keep_meow_sounds(wav_files_directory):
    yamnet_model = tf.saved_model.load(YAMNET_MODEL_DIR)
    infer = yamnet_model.signatures["serving_default"]

    # Go through each file in the directory
    for filename in os.listdir(wav_files_directory):
        if filename.endswith('.wav'):
            file_path = os.path.join(wav_files_directory, filename)

            # Load and preprocess the audio file
            audio = load_audio_file(file_path)

            # Prepare the audio tensor according to the model's expected input
            audio_tensor = tf.constant(audio, dtype=tf.float32)

            # Run inference
            output = infer(waveform=audio_tensor)

            # Extract scores from the output
            scores = output['output_0'].numpy()[0]

            if scores[YAMNET_CAT_MEOW_SOUND_INDEX] < YAMNET_CAT_SCORE_THRESHOLD:
                #  Not a meow, so remove the file
                os.remove(file_path)


def load_audio_file(file_path, target_sr=16000):
    # Load an audio file as a floating point time series
    audio, sr = librosa.load(file_path, sr=target_sr)
    return audio

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


def find_wav_files(directory):
    """Recursively finds all .wav files in the given directory and its subdirectories."""
    wav_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".wav"):
                wav_files.append(os.path.join(root, file))
    return wav_files


def get_wav_duration(wav_file):
    """Returns the duration of a .wav file in seconds."""
    with wave.open(wav_file, 'rb') as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        duration = frames / float(rate)
        return duration


def analyze_wav_files(directory):
    """Finds all .wav files in the directory, then reports the min, max, and average duration."""
    wav_files = find_wav_files(directory)
    if not wav_files:
        print("No .wav files found in the directory.")
        return

    durations = [get_wav_duration(file) for file in wav_files]
    min_duration = min(durations)
    max_duration = max(durations)
    avg_duration = sum(durations) / len(durations)

    print(f"Total .wav files found: {len(wav_files)}")
    print(f"Minimum duration: {min_duration} seconds")
    print(f"Maximum duration: {max_duration} seconds")
    print(f"Average duration: {avg_duration} seconds")


def split_wav_files(directory_path):
    # Ensure the provided path exists and is a directory
    if not os.path.isdir(directory_path):
        raise ValueError("The provided path is not a directory or does not exist.")

    # Iterate through all files in the directory
    for filename in os.listdir(directory_path):
        if filename.endswith('.wav'):
            full_path = os.path.join(directory_path, filename)

            # Load the .wav file
            audio = AudioSegment.from_wav(full_path)

            # Calculate the number of full 1-second chunks
            num_chunks = len(audio) // 1000

            # Split and save 1-second chunks
            for i in range(num_chunks):
                chunk = audio[i * 1000:(i + 1) * 1000]
                chunk_name = f"{filename[:-4]}-{i}.wav"
                chunk_path = os.path.join(directory_path, chunk_name)
                chunk.export(chunk_path, format="wav")

            # Delete the original file
            os.remove(full_path)

if __name__ == "__main__":
    directory = '/tmp/youtube_downloads'

    print("First, count the number of files in the directory")
    count_wav_files_in_subdirs(directory)

    print(f"Analyzing directory {directory}")
    analyze_wav_files(directory)

    print(f"Conforming directory {directory}")
    conform_process_directory(directory)

    print(f"Splitting files in {directory}")
    split_wav_files(directory)

    print("Keep only meows")
    keep_meow_sounds(directory)

    print("Re-count the files")
    count_wav_files_in_subdirs(directory)

    print(f"Re-analyze directory {directory}")
    analyze_wav_files(directory)

    print("Done")
