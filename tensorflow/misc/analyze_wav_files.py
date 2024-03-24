import glob
import os
import wave


def analyze_wav(file_path):
    try:
        with wave.open(file_path, 'rb') as wav_file:
            # Extract basic parameters
            n_channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            frame_rate = wav_file.getframerate()
            n_frames = wav_file.getnframes()
            comp_type = wav_file.getcomptype()
            comp_name = wav_file.getcompname()

            # Ensure 16-bit signed PCM data, mono channel
            if sample_width != 2 or n_channels != 1:
                print(f"The file {file_path} must be 16-bit signed PCM data, mono channel. Skipping...")
                return None  # Skip files that do not meet criteria

            return (frame_rate, n_channels, sample_width * 8, n_frames, comp_type, comp_name)
    except wave.Error as e:
        print(f"Error processing {file_path}: {e}")
        return None


def analyze_directory(directory_path):
    wav_files = glob.glob(os.path.join(directory_path, '*.wav'))
    for file_path in wav_files:
        result = analyze_wav(file_path)
        if result:
            frame_rate, n_channels, sample_width, n_frames, comp_type, comp_name = result
            print(
                f"{os.path.basename(file_path)}: Sample Rate={frame_rate} Hz, Channels={n_channels}, Sample Width="
                f"{sample_width} bits, Frames={n_frames}, Compression={comp_type} ({comp_name})")


if __name__ == "__main__":
    directory_path = "/Users/tennis/sound-library/vetted-meows"
    analyze_directory(directory_path)
