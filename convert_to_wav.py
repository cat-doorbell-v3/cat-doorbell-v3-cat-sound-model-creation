import os
import shutil

from pydub import AudioSegment


def copy_and_convert_directory(source_dir, target_dir):
    """
    Copy the directory structure from source to target and convert all MP3 files to WAV.

    Parameters:
    - source_dir: The path to the source directory.
    - target_dir: The path to the target directory where you want to copy and convert the files.
    """
    for root, dirs, files in os.walk(source_dir):
        # Construct the path to the destination directory
        dest_dir = os.path.join(target_dir, os.path.relpath(root, source_dir))

        # Create the destination directory
        os.makedirs(dest_dir, exist_ok=True)

        for file in files:
            # Full path for source and destination files
            src_file_path = os.path.join(root, file)
            dest_file_path = os.path.join(dest_dir, file)

            # If the file is an MP3, convert it to WAV
            if src_file_path.lower().endswith('.mp3'):
                audio = AudioSegment.from_mp3(src_file_path)
                dest_file_path = dest_file_path.rsplit('.', 1)[0] + '.wav'  # Change file extension to .wav
                audio.export(dest_file_path, format="wav")
            else:
                # If not an MP3 file, just copy the file as is
                shutil.copy2(src_file_path, dest_file_path)


# Example usage:
source_dir = './CAT_SOUND_DB_SAMPLES'
target_dir = './CAT_SOUND_DB_SAMPLES_WAV'
copy_and_convert_directory(source_dir, target_dir)
