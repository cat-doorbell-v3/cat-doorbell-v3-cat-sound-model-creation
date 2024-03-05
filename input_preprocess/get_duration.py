import os
import wave


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


# Example usage
if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python script.py <directory>")
    else:
        directory_name = sys.argv[1]
        analyze_wav_files(directory_name)
