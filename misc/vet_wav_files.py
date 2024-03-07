import os
import subprocess

# Path to the directory where .wav files are located
# Please update this path to the directory containing your .wav files.
wav_files_directory = '/tmp/model-test/cat'
keep_directory = '/tmp/keep'


# Function to find, play .wav files and move kept files to a specific directory
def find_and_play_wav_files(directory, keep_dir):
    # Create the keep directory if it does not exist
    if not os.path.exists(keep_dir):
        os.makedirs(keep_dir)

    # List all .wav files in the given directory
    wav_files = [f for f in os.listdir(directory) if f.endswith('.wav')]
    total_files = len(wav_files)

    # Iterate over each .wav file
    for count, filename in enumerate(wav_files, 1):
        # Full path to the file
        file_path = os.path.join(directory, filename)
        print(f"Playing {filename} ({count} of {total_files})...")
        # Play the .wav file using afplay
        subprocess.run(['afplay', file_path])
        # Ask the user if they want to keep the file, default is 'no'
        keep = input("Do you want to keep this .wav file? [y/N]: ") or 'n'
        # If user chooses 'n' or just hits return, delete the file
        if keep.lower() == 'n':
            os.remove(file_path)
            print(f"{filename} has been deleted.")
        elif keep.lower() == 'y':
            # Move the file to the keep directory
            keep_path = os.path.join(keep_dir, filename)
            os.rename(file_path, keep_path)
            print(f"{filename} has been moved to {keep_dir}.")


# Run the function with the path to your directory
find_and_play_wav_files(wav_files_directory, keep_directory)
