import os
import pickle  # For saving the features to a file

import librosa


def extract_mfcc_features(audio_file, n_mfcc=13, n_fft=2048, hop_length=512):
    """
    Extract MFCC features from an audio file.

    Parameters:
    - audio_file: Path to the audio file.
    - n_mfcc: Number of MFCC features to extract.
    - n_fft: Length of the FFT window.
    - hop_length: Number of samples between successive frames.

    Returns:
    - mfcc_features: MFCC features of the audio file.
    """
    audio, sample_rate = librosa.load(audio_file)
    mfcc_features = librosa.feature.mfcc(y=audio,
                                         sr=sample_rate,
                                         n_mfcc=n_mfcc,
                                         n_fft=n_fft,
                                         hop_length=hop_length)
    return mfcc_features.T  # Transpose to have time steps in rows and features in columns


def process_directory(directory):
    """
    Process a directory, extracting MFCC features from all WAV files.

    Parameters:
    - directory: The directory to process.
    """
    features = {}

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.wav'):
                file_path = os.path.join(root, file)
                mfcc_features = extract_mfcc_features(file_path)
                features[file_path] = mfcc_features

    return features


def save_features(features, output_file='mfcc_features.pkl'):
    """
    Save the extracted features to a file.

    Parameters:
    - features: The dictionary of features to save.
    - output_file: The path to the output file.
    """
    with open(output_file, 'wb') as f:
        pickle.dump(features, f)


# Example usage
directory = './CAT_SOUND_DB_SAMPLES_AUGMENTED'
features = process_directory(directory)
save_features(features, 'mfcc_features.pkl')

print(f"Extracted features for {len(features)} files.")
