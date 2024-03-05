"""
This changes .wav files to conform to:
- 30ms window frame
- 20ms window stride
- 16KHz sample rate
- 16-bit signed PCM data
- single channel (mono)

"""
import os

import librosa
import soundfile as sf


def process_audio_file(input_path, output_path):
    # Load the file (automatically resampled to 16KHz and converted to mono)
    data, sr = librosa.load(input_path, sr=16000, mono=True)

    # Ensure the data is in 16-bit signed PCM format
    data = (data * 32767).astype('int16')

    # Save the processed file
    sf.write(output_path, data, sr, subtype='PCM_16')


def process_directory(input_dir, output_dir):
    # Walk through the input directory
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.wav'):
                input_path = os.path.join(root, file)

                # Construct the output path
                relative_path = os.path.relpath(input_path, input_dir)
                output_path = os.path.join(output_dir, relative_path)

                # Ensure the output directory exists
                os.makedirs(os.path.dirname(output_path), exist_ok=True)

                # Process the file
                print(f"Processing {input_path}...")
                process_audio_file(input_path, output_path)


# Example usage
input_dir = '/Users/tennis/sound-library/raw-audio'
output_dir = '/Users/tennis/cat-doorbell-model-test/not_cat'
process_directory(input_dir, output_dir)
