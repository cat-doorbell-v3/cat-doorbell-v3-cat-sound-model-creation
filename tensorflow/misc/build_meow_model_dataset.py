"""
Google:
 - The input for this script was downloaded from here: https://research.google.com/audioset/download.html
 - The dataset was: http://storage.googleapis.com/us_audioset/youtube_corpus/v1/features/features.tar.gz

YamNet:
 - https://www.kaggle.com/models/google/yamnet/frameworks/tensorFlow2

Todo:
 - Pull/untar the audioset dataset from Google
 - Pull/unzip the dataset from YamNet
"""
import os
import random
import subprocess

import librosa
import tensorflow as tf
from pydub import AudioSegment

TFRECORD_FILES_PATTERN = '/tmp/audioset/*/*.tfrecord'
CAT_OUTPUT_DIR = '/tmp/cat-doorbell-model/meow'
UNKNOWN_OUTPUT_DIR = '/tmp/cat-doorbell-model/unknown'
YAMNET_MODEL_DIR = '/Users/tennis/sound-library/yamnet/archive'

SAMPLE_COUNT = 1000

"""
Indices per this file: http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv
"""
GOOGLE_CAT_ALL_SOUND_INDEX = 81
GOOGLE_CAT_PURR_SOUND_INDEX = 82
GOOGLE_CAT_MEOW_SOUND_INDEX = 83
GOOGLE_CAT_HISS_SOUND_INDEX = 84
GOOGLE_CAT_CATERWAUL_SOUND_INDEX = 85

GOOGLE_CAT_SOUND_INDICES = {
    GOOGLE_CAT_ALL_SOUND_INDEX,
    GOOGLE_CAT_PURR_SOUND_INDEX,
    GOOGLE_CAT_MEOW_SOUND_INDEX,
    GOOGLE_CAT_HISS_SOUND_INDEX,
    GOOGLE_CAT_CATERWAUL_SOUND_INDEX
}

"""
Indices per this file: https://github.com/tensorflow/models/blob/master/research/audioset/yamnet/yamnet_class_map.csv
"""

YAMNET_CAT_SCORE_THRESHOLD = 0.8  # 80% certainty

YAMNET_CAT_ALL_SOUND_INDEX = 76
YAMNET_CAT_PURR_SOUND_INDEX = 77
YAMNET_CAT_MEOW_SOUND_INDEX = 78
YAMNET_CAT_HISS_SOUND_INDEX = 79
YAMNET_CAT_CATERWAUL_SOUND_INDEX = 80

YAMNET_CAT_SOUND_INDICES = {
    YAMNET_CAT_ALL_SOUND_INDEX,
    YAMNET_CAT_PURR_SOUND_INDEX,
    YAMNET_CAT_MEOW_SOUND_INDEX,
    YAMNET_CAT_HISS_SOUND_INDEX,
    YAMNET_CAT_CATERWAUL_SOUND_INDEX
}

FEATURE_DESCRIPTION = {
    'video_id': tf.io.FixedLenFeature([], tf.string),
    'start_time_seconds': tf.io.FixedLenFeature([], tf.float32),
    'end_time_seconds': tf.io.FixedLenFeature([], tf.float32),
}


def load_audio_file(file_path, target_sr=16000):
    # Load an audio file as a floating point time series
    audio, sr = librosa.load(file_path, sr=target_sr)
    return audio


def remove_any_cat_sounds_using_yamnet(wav_files_directory):
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

            # Check if any of the cat sound indices exceed the threshold
            for cat_sound_index in YAMNET_CAT_SOUND_INDICES:
                if scores[cat_sound_index] >= YAMNET_CAT_SCORE_THRESHOLD:
                    # Found a cat sound, so remove the file
                    os.remove(file_path)


def keep_meow_sounds_using_yamnet(wav_files_directory):
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


def is_google_meow(entry):
    def py_func(entry):
        example = tf.train.Example()
        example.ParseFromString(entry.numpy())
        for key, feature in example.features.feature.items():
            if key == 'labels' and GOOGLE_CAT_MEOW_SOUND_INDEX in feature.int64_list.value:
                return True
        return False

    return tf.py_function(py_func, [entry], Tout=tf.bool)


def is_google_cat(entry):
    def py_func(entry):
        example = tf.train.Example()
        example.ParseFromString(entry.numpy())
        labels_feature = example.features.feature.get('labels')
        if labels_feature:
            # Convert label indices to a set for efficient intersection check
            label_indices = set(labels_feature.int64_list.value)
            # Check if there is any intersection between label indices and cat sound indices
            if label_indices.intersection(GOOGLE_CAT_SOUND_INDICES):
                return True
        return False

    # Use tf.py_function to wrap the Python function
    return tf.py_function(py_func, [entry], Tout=tf.bool)


def download_audio_segment(video_id, start_time, end_time, output_filename):
    # Calculate duration
    duration = end_time - start_time

    # Construct yt-dlp command to get the direct audio URL
    yt_dlp_cmd = ['yt-dlp', '-f', 'bestaudio', '--get-url', f'https://www.youtube.com/watch?v={video_id}']

    # Execute yt-dlp command and capture output
    process = subprocess.run(yt_dlp_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    audio_url = process.stdout.strip()

    # Construct ffmpeg command to download the specified segment and convert it to the desired format
    """
    This changes .wav files to conform to:
    - 30ms window frame
    - 20ms window stride
    - 16KHz sample rate
    - 16-bit signed PCM data
    - single channel (mono)
    """
    ffmpeg_cmd = [
        'ffmpeg', '-loglevel', 'quiet', '-ss', str(start_time), '-t', str(duration), '-i',
        audio_url, '-vn', '-ar', '16000', '-ac', '1', '-acodec', 'pcm_s16le', '-f', 'wav', output_filename
    ]

    # Execute ffmpeg command
    subprocess.run(ffmpeg_cmd)


def extract_audio_files(dataset, output_dir, feature_description):
    """
    Extracts audio files from a given dataset and saves them to the specified output directory.
    If a file with the intended name already exists, appends a dash and 4 random characters to the filename.

    Args:
        dataset: A tf.data.Dataset object containing records to process.
        output_dir: The directory where output audio files will be saved.
        feature_description: A dictionary describing the features in the dataset records for parsing.
    """

    for raw_record in dataset:
        # Parse the raw record based on the provided feature description
        parsed_record = tf.io.parse_single_example(raw_record, feature_description)

        # Decode the necessary fields from the parsed record
        video_id = parsed_record['video_id'].numpy().decode('utf-8')
        start_time_seconds = parsed_record['start_time_seconds'].numpy()
        end_time_seconds = parsed_record['end_time_seconds'].numpy()

        # Construct the initial output filename
        output_filename = f"{output_dir}/{video_id}.wav"

        # Check if the file already exists
        if os.path.exists(output_filename):
            continue

        # Download the audio segment
        download_audio_segment(video_id, start_time_seconds, end_time_seconds, output_filename)


def count_wav_files(directory):
    files_in_directory = os.listdir(directory)
    wav_files = [file for file in files_in_directory if file.endswith(".wav")]
    return len(wav_files)


# Example usage
directory_path = '/path/to/your/directory'
count_wav_files(directory_path)

def main():
    print("Checking if output directories exist...")
    # Ensure output directories exist
    os.makedirs(CAT_OUTPUT_DIR, exist_ok=True)
    os.makedirs(UNKNOWN_OUTPUT_DIR, exist_ok=True)

    print("Loading dataset...")

    # Use Python's glob to find TFRecord files
    tfrecord_files = tf.io.gfile.glob(TFRECORD_FILES_PATTERN)
    print(f"Found {len(tfrecord_files)} TFRecord files")

    print("shuffle to randomize order of files")
    random.shuffle(tfrecord_files)

    print("Creating raw tf.data.Dataset")
    dataset_raw = tf.data.TFRecordDataset(tfrecord_files)

    print("Counting raw tf.data.Dataset")
    dataset_raw_size = sum(1 for _ in dataset_raw)

    print(f"Raw tf.data.Dataset has {dataset_raw_size} records")

    print("Creating Meow tf.data.Dataset")
    dataset_cat = dataset_raw.filter(lambda x: is_google_meow(x)).take(SAMPLE_COUNT)

    print("Counting Meow tf.data.Dataset")
    dataset_cat_size = sum(1 for _ in dataset_cat)

    print(f"Meow tf.data.Dataset has {dataset_cat_size} records")

    print("Creating Unknown tf.data.Dataset")
    dataset_unknown = dataset_raw.filter(lambda x: not is_google_cat(x)).take(dataset_cat_size)

    print("Counting Unknown tf.data.Dataset")
    dataset_unknown_size = sum(1 for _ in dataset_unknown)

    print(f"Unknown tf.data.Dataset has {dataset_unknown_size} records")

    print("Extracting 10-second audio files for meow")
    extract_audio_files(dataset_cat, CAT_OUTPUT_DIR, FEATURE_DESCRIPTION)

    print("Extracting 10-second audio files for unknown")
    extract_audio_files(dataset_unknown, UNKNOWN_OUTPUT_DIR, FEATURE_DESCRIPTION)

    print("Splitting CAT audio files into 1-second chunks")
    split_wav_files(CAT_OUTPUT_DIR)

    print("Splitting Unknown audio files into 1-second chunks")
    split_wav_files(UNKNOWN_OUTPUT_DIR)

    print("Keeping meows in 1-second files using YAMNET")
    keep_meow_sounds_using_yamnet(CAT_OUTPUT_DIR)

    print("Removing cat sounds in Unknown files using YAMNET")
    remove_any_cat_sounds_using_yamnet(UNKNOWN_OUTPUT_DIR)

    print("Counting.wav files in CAT directory")
    chunked_meow_file_count = count_wav_files(CAT_OUTPUT_DIR)
    print(f"Found {chunked_meow_file_count} files in CAT directory")

    print("Counting.wav files in UNKNOWN directory")
    chunked_unknown_file_count = count_wav_files(UNKNOWN_OUTPUT_DIR)
    print(f"Found {chunked_unknown_file_count} files in UNKNOWN directory")

    print("Done")


if __name__ == '__main__':
    import time

    start_time = time.time()
    main()
    end_time = time.time()
    elapsed_time = end_time - start_time
    # Convert elapsed time to hours and minutes
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, _ = divmod(remainder, 60)
    print(f"Total elapsed time: {int(hours)} hours and {int(minutes)} minutes")
