import sys

import librosa
import numpy as np


def wav_to_c_array(wav_path, output_path, n_mels=49, hop_length=512):
    # Load the wav file
    y, sr = librosa.load(wav_path, sr=None)

    # Convert to Mel spectrogram
    # Note: You can adjust n_mels and hop_length as needed
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length)
    S_dB = librosa.power_to_db(S, ref=np.max)

    # Dimensions of the spectrogram
    height, width = S_dB.shape

    # Normalize and quantize
    S_normalized = (S_dB - S_dB.min()) / (S_dB.max() - S_dB.min())
    S_quantized = np.round(255 * S_normalized) - 128
    S_quantized = S_quantized.astype(np.int8)

    # Generate C array
    with open(output_path, 'w') as f:
        f.write(f"const int g_no_micro_f9643d42_nohash_4_width = {width};\n")
        f.write(f"const int g_no_micro_f9643d42_nohash_4_height = {n_mels};\n")
        f.write("const signed char g_no_micro_f9643d42_nohash_4_data[] = {\n")
        for i, val in enumerate(S_quantized.flatten()):
            f.write(f"{val}, ")
            if (i + 1) % 10 == 0:  # New line every 10 values for readability
                f.write("\n")
        f.write("};\n")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <input_wav_path> <output_c_path>")
        sys.exit(1)

    wav_path = sys.argv[1]
    output_path = sys.argv[2]
    wav_to_c_array(wav_path, output_path)
