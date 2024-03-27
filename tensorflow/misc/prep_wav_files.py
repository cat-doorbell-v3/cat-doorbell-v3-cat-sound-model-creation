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

    print("Re-count the files")
    count_wav_files_in_subdirs(directory)

    print(f"Re-analyze directory {directory}")
    analyze_wav_files(directory)

    print("Done")
