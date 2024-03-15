import os
import random
import subprocess

import tensorflow as tf

# Google AudioSet extract sample parameters
#
TFRECORD_FILES_PATTERN = '/tmp/audioset/*/*.tfrecord'
CAT_OUTPUT_DIR = '/tmp/cat-doorbell-model-test/meow'
NOT_CAT_OUTPUT_DIR = '/tmp/cat-doorbell-model-test/other'
SAMPLE_COUNT = 1000

"""
Indices per this file: http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv
"""
CAT_CAT_SOUND_INDEX = 81
CAT_PURR_SOUND_INDEX = 82
CAT_MEOW_SOUND_INDEX = 83
CAT_HISS_SOUND_INDEX = 84
CAT_CATERWAUL_SOUND_INDEX = 85

CAT_SOUND_INDICES = {
    CAT_CAT_SOUND_INDEX,
    CAT_PURR_SOUND_INDEX,
    CAT_MEOW_SOUND_INDEX,
    CAT_HISS_SOUND_INDEX,
    CAT_CATERWAUL_SOUND_INDEX
}

FEATURE_DESCRIPTION = {
    'video_id': tf.io.FixedLenFeature([], tf.string),
    'start_time_seconds': tf.io.FixedLenFeature([], tf.float32),
    'end_time_seconds': tf.io.FixedLenFeature([], tf.float32),
}


def is_meow(entry):
    def py_func(entry):
        example = tf.train.Example()
        example.ParseFromString(entry.numpy())
        for key, feature in example.features.feature.items():
            if key == 'labels' and CAT_MEOW_SOUND_INDEX in feature.int64_list.value:
                return True
        return False

    return tf.py_function(py_func, [entry], Tout=tf.bool)


def is_cat(entry):
    def py_func(entry):
        example = tf.train.Example()
        example.ParseFromString(entry.numpy())
        labels_feature = example.features.feature.get('labels')
        if labels_feature:
            # Convert label indices to a set for efficient intersection check
            label_indices = set(labels_feature.int64_list.value)
            # Check if there is any intersection between label indices and cat sound indices
            if label_indices.intersection(CAT_SOUND_INDICES):
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


def main():
    print("Checking if output directories exist...")
    # Ensure output directories exist
    os.makedirs(CAT_OUTPUT_DIR, exist_ok=True)
    os.makedirs(NOT_CAT_OUTPUT_DIR, exist_ok=True)

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
    dataset_cat = dataset_raw.filter(lambda x: is_meow(x)).take(SAMPLE_COUNT)

    print("Counting Meow tf.data.Dataset")
    dataset_cat_size = sum(1 for _ in dataset_cat)

    print(f"Meow tf.data.Dataset has {dataset_cat_size} records")

    print("Creating NOT Cat tf.data.Dataset")
    dataset_not_cat = dataset_raw.filter(lambda x: not is_cat(x)).take(dataset_cat_size)

    print("Counting NOT Cat tf.data.Dataset")
    dataset_not_cat_size = sum(1 for _ in dataset_not_cat)

    print(f"NOT Cat tf.data.Dataset has {dataset_not_cat_size} records")

    print("Extracting audio files for dataset_cat")
    extract_audio_files(dataset_cat, CAT_OUTPUT_DIR, FEATURE_DESCRIPTION)

    print("Extracting audio files for dataset_not_cat")
    extract_audio_files(dataset_not_cat, NOT_CAT_OUTPUT_DIR, FEATURE_DESCRIPTION)

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
