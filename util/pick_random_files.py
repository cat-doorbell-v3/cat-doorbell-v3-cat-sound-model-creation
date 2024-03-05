import glob
import os
import random
import shutil


def pick_random_wav_files(source_dir, destination_dir, file_count=50):
    # Check if the destination directory exists, if not, create it
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    # Use glob to find all .wav files within the source directory, including subdirectories
    wav_files = glob.glob(os.path.join(source_dir, '**', '*.wav'), recursive=True)

    # If there are fewer files than requested, adjust the file_count
    file_count = min(len(wav_files), file_count)

    # Randomly select 'file_count' files
    selected_files = random.sample(wav_files, file_count)

    # Copy the selected files to the destination directory
    for file in selected_files:
        shutil.copy(file, destination_dir)
        print(f"Copied: {file} to {destination_dir}")


# Example usage
source_directory = '/Users/tennis/sound-library/data_speech_commands_v0.02'
destination_directory = '/Users/tennis/sound-library/raw-audio'
pick_random_wav_files(source_directory, destination_directory)
