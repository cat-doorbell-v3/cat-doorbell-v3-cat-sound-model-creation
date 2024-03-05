import os
import wave

import librosa

import constants
import utils

"""
A common choice for speech processing tasks at an 8,000 Hz sampling rate is 
an FFT window size of 256 or 512 samples. These sizes offer a good trade-off 
between frequency resolution and time resolution for the lower sampling rate 
and are efficient for FFT computations (being powers of two). The specific 
choice can depend on the nature of your audio data and the requirements of 
your application:
FFT_WINDOW_SIZE = 512

For most speech processing tasks at an 8,000 Hz sampling rate with an FFT 
window size of 512, a hop length of 128 or 256 samples is a practical choice. 
It balances the need for temporal resolution to capture the dynamics of speech 
with the computational efficiency necessary for processing. The specific choice 
should be based on your application's requirements, with 128 samples offering 
higher overlap and potentially smoother feature extraction and 256 samples 
reducing computational demands while providing adequate temporal resolution.
HOP_LENGTH = 256


AUDIO_CONSTANTS = {
    "kMaxAudioSampleSize": None,
    "kAudioSampleFrequency": constants.SAMPLING_RATE,
    "kFeatureSize": None,  # ???
    "kFeatureCount": None,
    "kFeatureElementCount": None,
    "kFeatureStrideMs": None,
    "kFeatureDurationMs": None
}
print(f"Calculating average duration of files in {wav_dir}...")
average_duration = get_audio_durations_and_average(wav_dir)

print(f"Get sampling rates from {wav_dir}...")
sampling_rates = walk_directory_for_wav_sampling_rates(wav_dir)

print(f"Get bit depths from {wav_dir}...")
bit_depths_found = walk_directory_for_wavs(wav_dir)

bit_depth = 0
for bit_depth, files in bit_depths_found.items():
    print(f"Found {len(files)} file(s) with a bit depth of {bit_depth} bits:")

DEPTH_BITS = bit_depth
total_samples = average_duration * sampling_rates[0]
AUDIO_CONSTANTS["kFeatureCount"] = int(1 + (total_samples - FFT_WINDOW_SIZE) / HOP_LENGTH)
AUDIO_CONSTANTS["kFeatureStrideMs"] = int((HOP_LENGTH / sampling_rates[0]) * 1000)
AUDIO_CONSTANTS["kFeatureDurationMs"] = int((FFT_WINDOW_SIZE / sampling_rates[0]) * 1000)
DURATION_SECONDS = 1
BYTES_PER_SAMPLE = DEPTH_BITS / 8
AUDIO_CONSTANTS["kMaxAudioSampleSize"] = int((AUDIO_SAMPLING_FREQUENCY / 1000) * AUDIO_CONSTANTS["kFeatureDurationMs"])
AUDIO_CONSTANTS["kFeatureElementCount"] = int(AUDIO_CONSTANTS["kFeatureSize"] * AUDIO_CONSTANTS["kFeatureCount"])
"""


def get_directory_wav_sampling_rates(root_dir):
    """
    Walks through a directory and its subdirectories to find .wav files and compile a list of unique sampling rates.

    :param root_dir: The root directory to start the walk from.
    """
    unique_sampling_rates = set()  # Use a set to store unique sampling rates

    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".wav"):
                file_path = os.path.join(subdir, file)
                try:
                    with wave.open(file_path, 'r') as wav_file:
                        frame_rate = wav_file.getframerate()
                        unique_sampling_rates.add(frame_rate)  # Add the frame rate to the set
                except wave.Error as e:
                    print(f"Error reading {file_path}: {e}")

    # Convert the set to a sorted list to print the sampling rates in order
    sorted_sampling_rates = sorted(list(unique_sampling_rates))
    print(f"Unique Sampling Rates: {sorted_sampling_rates}")
    return sorted_sampling_rates


def get_bit_depth(wav_file_path):
    try:
        with wave.open(wav_file_path, 'r') as wav_file:
            sample_width_bytes = wav_file.getsampwidth()
            bit_depth = sample_width_bytes * 8  # Convert bytes to bits
            return bit_depth
    except wave.Error:
        print(f"Error reading {wav_file_path}. It might not be a valid WAV file.")
        return None


def get_all_bit_depth(directory_path):
    bit_depths = {}
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith(".wav"):
                full_path = os.path.join(root, file)
                bit_depth = get_bit_depth(full_path)
                if bit_depth:
                    if bit_depth in bit_depths:
                        bit_depths[bit_depth].append(full_path)
                    else:
                        bit_depths[bit_depth] = [full_path]

    # Check if bit_depths contains only one entry
    if len(bit_depths) == 1:
        # Return just the bit depth value
        return list(bit_depths.keys())[0]

    return bit_depths


def get_audio_durations(directory_path):
    """
    Walks through a directory, calculates the duration of all .wav files,
    and computes the minimum, maximum, and average durations.

    Args:
    directory_path (str): Path to the directory to search for .wav files.

    Returns:
    None
    """
    durations = []

    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.lower().endswith('.wav'):
                file_path = os.path.join(root, file)
                y, sr = librosa.load(file_path, sr=None)  # Load with the original sampling rate
                duration = librosa.get_duration(y=y, sr=sr)
                durations.append(duration)

    if durations:
        min_duration = min(durations)
        max_duration = max(durations)
        average_duration = sum(durations) / len(durations)
    else:
        print("No .wav files found in the directory.")
        return None, None, None

    return min_duration, max_duration, average_duration


def generate_cpp_definitions(feature_size, sample_rate, target_duration, fft_window_size, hop_length):
    total_samples = target_duration * sample_rate
    feature_count = int(1 + (total_samples - fft_window_size) / hop_length)
    feature_duration_ms = int((fft_window_size / sample_rate) * 1000)
    return {
        "kMaxAudioSampleSize": int((sample_rate / 1000) * feature_duration_ms),
        "kAudioSampleFrequency": sample_rate,
        "kFeatureSize": feature_size,
        "kFeatureCount": feature_count,
        "kFeatureElementCount": int(feature_size * feature_count),
        "kFeatureStrideMs": int((hop_length / sample_rate) * 1000),
        "kFeatureDurationMs": feature_duration_ms,
    }


def generate_header_file(label_map, audio_constants, output_file="cat_sound_model.h"):
    labels = list(label_map.keys())
    category_count = len(labels)

    with open(output_file, "w") as f:
        f.write("#ifndef CAT_SOUND_MODEL_H_\n")
        f.write("#define CAT_SOUND_MODEL_H_\n\n")

        # Write audio constants
        for key, value in audio_constants.items():
            f.write(f"constexpr int {key} = {value};\n")
        f.write("\n")

        # Write category labels
        f.write(f"constexpr int kCategoryCount = {category_count};\n")
        f.write("constexpr const char* kCategoryLabels[kCategoryCount] = {\n")
        for label in labels:
            f.write(f'    "{label}",\n')
        f.write("};\n\n")

        f.write("#endif  // CAT_SOUND_MODEL_H_\n")


def main():
    utils.remove_directories([constants.MODEL_DATASET])

    utils.unzip_file(constants.MODEL_DATASET_ZIP, '/tmp')

    bit_depths = get_all_bit_depth(constants.MODEL_DATASET_PATH)
    print(f"Found {bit_depths} bit depths")

    sample_rates = get_directory_wav_sampling_rates(constants.MODEL_DATASET_PATH)
    print(f"Found {len(sample_rates)} sample rate")
    for s in sample_rates:
        print(f"Sample Rate: {s}")

    min_duration, max_duration, avg_duration = get_audio_durations(constants.MODEL_DATASET_PATH)
    print(f"Min Duration: {min_duration:.2f} seconds")
    print(f"Max Duration: {max_duration:.2f} seconds")
    print(f"Average Duration: {avg_duration:.2f} seconds")

    audio_cc_constants = generate_cpp_definitions(constants.FEATURE_SIZE,
                                                  constants.SAMPLING_RATE,
                                                  avg_duration,
                                                  constants.N_FFT,
                                                  constants.HOP_LENGTH)

    generate_header_file(constants.DATASET_LABEL_MAP, audio_cc_constants)


if __name__ == '__main__':
    main()
