import os

from pydub import AudioSegment
from pydub.silence import detect_nonsilent


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


if __name__ == "__main__":
    directory = '/tmp/google-audioset-samples/meow'

    print(f"Trimming directory {directory}")
    trim_process_directory(directory)
