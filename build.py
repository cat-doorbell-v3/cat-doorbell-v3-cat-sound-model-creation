import shutil
import wave
import zipfile

import librosa
import numpy as np
import soundfile as sf
import tensorflow as tf
from keras.callbacks import EarlyStopping
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.utils import to_categorical

MODEL_NAME = "cat_sound_model.tflite"

AUDIO_SAMPLING_FREQUENCY = 8000

"""
Why 13?: The choice of 13 as the feature size is a conventional practice that 
dates back to early speech recognition research. It has been found empirically 
that the first 12 MFCCs (along with the energy, making it 13) capture most of 
the relevant information about the spectral envelope of the audio signal for 
many tasks, including speech recognition and music information retrieval. 
Additional coefficients can provide diminishing returns or even introduce 
noise for certain applications."""
FEATURE_SIZE = 13

"""
A common choice for speech processing tasks at an 8,000 Hz sampling rate is 
an FFT window size of 256 or 512 samples. These sizes offer a good trade-off 
between frequency resolution and time resolution for the lower sampling rate 
and are efficient for FFT computations (being powers of two). The specific 
choice can depend on the nature of your audio data and the requirements of 
your application:
"""
FFT_WINDOW_SIZE = 512

"""
For most speech processing tasks at an 8,000 Hz sampling rate with an FFT 
window size of 512, a hop length of 128 or 256 samples is a practical choice. 
It balances the need for temporal resolution to capture the dynamics of speech 
with the computational efficiency necessary for processing. The specific choice 
should be based on your application's requirements, with 128 samples offering 
higher overlap and potentially smoother feature extraction and 256 samples 
reducing computational demands while providing adequate temporal resolution.
"""
HOP_LENGTH = 256

LABEL_MAP = {
    'brushing': 0, 'isolation': 1, 'food': 2
}


def unzip_file(zip_file_path, extract_to_path):
    """
    Unzips a ZIP file to a specified location.

    Args:
    zip_file_path (str): The path to the ZIP file.
    extract_to_path (str): The destination directory where the contents will be extracted.

    Returns:
    None
    """
    # Ensure the destination directory exists, create if it doesn't
    if not os.path.exists(extract_to_path):
        os.makedirs(extract_to_path)

    # Open the zip file
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        # Extract all the contents into the destination directory
        zip_ref.extractall(extract_to_path)
    print(f"File extracted to {extract_to_path}")


def remove_directories(dir_list):
    """
    Remove a list of directories located in the /tmp directory.

    Parameters:
    - dir_list (list of str): A list of directory names to be removed.

    Returns:
    - None
    """
    base_path = "/tmp"
    for dir_name in dir_list:
        dir_path = os.path.join(base_path, dir_name)
        try:
            shutil.rmtree(dir_path)
            print(f"Directory {dir_path} removed successfully.")
        except FileNotFoundError:
            print(f"Directory {dir_path} does not exist.")
        except Exception as e:
            print(f"Error removing directory {dir_path}: {e}")


# Example usage:
# remove_directories(['sounds', 'temp_data', 'unused_files'])
import os


def organize_meows(directory):
    """
    Organizes .wav files in the specified directory into subdirectories based on their emission context.

    Parameters:
    - directory (str): The path to the directory containing the .wav files.

    Returns:
    - None
    """
    # Define the base path and subdirectories
    base_path = "/tmp/dataset"
    contexts = {'B': 'brushing', 'I': 'isolation', 'F': 'food'}

    # Create the base directory if it doesn't exist
    os.makedirs(base_path, exist_ok=True)

    # Create subdirectories for each context
    for context in contexts.values():
        os.makedirs(os.path.join(base_path, context), exist_ok=True)

    # Move files based on the emission context
    for filename in os.listdir(directory):
        if filename.endswith('.wav'):
            context_code = filename.split('_')[0]  # Extract context code from file name
            target_dir = contexts.get(context_code)  # Determine target directory
            if target_dir:
                os.rename(
                    os.path.join(directory, filename),
                    os.path.join(base_path, target_dir, filename)
                )
                # print(f"Moved {filename} to {target_dir}.")


def get_audio_durations_and_average(directory_path):
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
        print(f"Min Duration: {min_duration:.2f} seconds")
        print(f"Max Duration: {max_duration:.2f} seconds")
        print(f"Average Duration: {average_duration:.2f} seconds")
    else:
        print("No .wav files found in the directory.")
        return None, None, None

    return min_duration, max_duration, average_duration


def walk_directory_for_wav_sampling_rates(root_dir):
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


def walk_directory_for_wav_bit_depth(directory_path):
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


def add_noise(data, noise_factor=0.005):
    noise = np.random.randn(len(data))
    augmented_data = data + noise_factor * noise
    return augmented_data.astype(type(data[0]))


def shift_pitch(data, sampling_rate, pitch_factor=5):
    return librosa.effects.pitch_shift(y=data, sr=sampling_rate, n_steps=pitch_factor)


def change_speed(data, speed_factor=1.25):
    return np.interp(np.arange(0, len(data), speed_factor), np.arange(0, len(data)), data)


def time_stretch(data, rate=1.25):
    return librosa.effects.time_stretch(data, rate=rate)


def dynamic_range_compression(data, compression_factor=0.5):
    # Ensure data is in float format
    data = data.astype(float)
    # Apply compression
    return np.sign(data) * np.log1p(compression_factor * np.abs(data))


def augment_and_save(file_path, sampling_rate):
    data, _ = librosa.load(file_path, sr=sampling_rate)

    # Existing augmentations
    noise_data = add_noise(data)
    pitch_shifted_data = shift_pitch(data, sampling_rate)
    speed_changed_data = change_speed(data)

    # New augmentations
    time_stretched_data = time_stretch(data)
    compressed_data = dynamic_range_compression(data)

    base_name = os.path.splitext(os.path.basename(file_path))[0]
    dir_name = os.path.dirname(file_path)

    # Save files
    sf.write(os.path.join(dir_name, f"{base_name}_original.wav"), data, sampling_rate)
    sf.write(os.path.join(dir_name, f"{base_name}_noise.wav"), noise_data, sampling_rate)
    sf.write(os.path.join(dir_name, f"{base_name}_pitch.wav"), pitch_shifted_data, sampling_rate)
    sf.write(os.path.join(dir_name, f"{base_name}_speed.wav"), speed_changed_data, sampling_rate)
    sf.write(os.path.join(dir_name, f"{base_name}_timestretch.wav"), time_stretched_data, sampling_rate)
    sf.write(os.path.join(dir_name, f"{base_name}_compressed.wav"), compressed_data, sampling_rate)


def augment_wav_files(source_dir, sampling_rate):
    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.lower().endswith('.wav'):
                file_path = os.path.join(root, file)
                augment_and_save(file_path, sampling_rate)


def extract_mfcc_features(audio_file, n_mfcc=FEATURE_SIZE, n_fft=FFT_WINDOW_SIZE, hop_length=HOP_LENGTH):
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


def process_directory_for_mfcc_features(directory):
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


def normalize_features(features):
    normalized_features = []
    for feature in features:
        if feature.size == 0:  # Skip empty features if there are any
            continue
        scaler = StandardScaler()
        normalized_feature = scaler.fit_transform(feature)  # Normalize each feature array
        normalized_features.append(normalized_feature)
    return normalized_features


def pad_features(features, max_length):
    """
    Pads or trims the feature arrays so that they all have the same length.

    Args:
    features (list): A list of feature arrays.
    max_length (int): The length to which the feature arrays will be padded or trimmed.

    Returns:
    np.array: An array of features adjusted to have the same length.
    """
    padded_features = []

    for feature in features:
        if len(feature) < max_length:
            # Pad the feature array if it's shorter than the max_length
            padding = ((0, max_length - len(feature)), (0, 0))
            feature_padded = np.pad(feature, padding, 'constant', constant_values=0)
        elif len(feature) > max_length:
            # Trim the feature array if it's longer than the max_length
            feature_padded = feature[:max_length]
        else:
            # If the feature is already the correct length, use it as is
            feature_padded = feature
        padded_features.append(feature_padded)

    return np.array(padded_features)


def representative_dataset_gen(X_train_cnn):
    for input_value in X_train_cnn[:100]:  # Assuming you're using a subset of your training data
        yield [np.expand_dims(input_value, axis=0).astype(np.float32)]


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


def extract_features_and_labels(mfcc_data, label_map):
    """
    Extracts features and labels from the mfcc_data dictionary.

    Parameters:
    - mfcc_data: Dictionary with file paths as keys and MFCC features as values.
    - label_map: Dictionary mapping label names to numerical values.

    Returns:
    - features: NumPy array of MFCC features.
    - labels: NumPy array of numerical labels.
    """
    features = []
    labels = []

    for file_path, mfcc_features in mfcc_data.items():
        # Assuming the label is in the fourth position of the path split
        label_name = file_path.split('/')[3]
        label = label_map.get(label_name)

        if label is not None:
            features.append(mfcc_features)
            labels.append(label)

    features = np.array(features, dtype=object)
    labels = np.array(labels)

    return features, labels


def train_and_select_best_model(X_train, y_train, input_shape, num_classes, n_splits=5, patience=4):
    """
    Trains models using K-Fold cross-validation and returns the best model based on validation accuracy.

    Parameters:
    - X_train: Training data features.
    - y_train: Training data labels.
    - input_shape: Shape of the input data (excluding the batch size).
    - num_classes: Number of unique classes in the dataset.
    - n_splits: Number of folds for K-Fold cross-validation.
    - patience: Number of epochs with no improvement after which training will be stopped.

    Returns:
    - best_model: The Keras model with the highest validation accuracy.
    """

    # Define early stopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=patience)

    # Define the KFold cross-validator
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    best_score = -np.inf
    best_model = None

    fold_no = 1
    for train, val in kf.split(X_train, y_train):
        # Initialize the model
        model = Sequential([
            Flatten(input_shape=input_shape),
            Dense(64, activation='relu', kernel_regularizer=l1_l2(l1=0.01, l2=0.01)),
            Dropout(0.3),
            Dense(num_classes, activation='softmax')
        ])

        # Compile the model
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # Fit the model
        model.fit(X_train[train], y_train[train],
                  batch_size=32,
                  epochs=30,
                  verbose=0,
                  validation_data=(X_train[val], y_train[val]),
                  callbacks=[early_stopping])

        # Evaluate the model
        val_loss, val_accuracy = model.evaluate(X_train[val], y_train[val], verbose=0)
        print(f'Fold {fold_no} - Loss: {val_loss:.2f} - Accuracy: {val_accuracy * 100:.2f}%')

        # Update the best model if current model is better
        if val_accuracy > best_score:
            best_score = val_accuracy
            best_model = model

        fold_no += 1

    # Optionally save the best model to a file
    best_model.save('/tmp/best_model.keras')

    print(f'Best model selected from fold {fold_no} with accuracy: {best_score * 100:.2f}%')
    return best_model


def convert_and_save_model_to_tflite(keras_model, X_train_cnn, model_name="converted_model.tflite"):
    # Convert the TensorFlow model to a TensorFlow Lite model with full integer quantization
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)

    # Specify optimizations for the converter to perform
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # Provide a representative dataset for calibration
    converter.representative_dataset = lambda: representative_dataset_gen(X_train_cnn)

    # Ensure the converter generates a model compatible with integer-only input and output
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8  # Set input type to int8
    converter.inference_output_type = tf.int8  # Set output type to int8

    # Convert the model
    tflite_model_quant = converter.convert()

    # Save the quantized model to a file
    with open(model_name, "wb") as f:
        f.write(tflite_model_quant)
    print(f"Model saved to {model_name}")


def main():
    remove_directories(['dataset'])

    unzip_file('dataset.zip', '/tmp')

    organize_meows('/tmp/dataset')

    min_duration, max_duration, avg_duration = get_audio_durations_and_average('/tmp/dataset')

    sampling_rates = walk_directory_for_wav_sampling_rates('/tmp/dataset')

    bit_depths_found = walk_directory_for_wav_bit_depth('/tmp/dataset')

    print(f"All files have a bit depth of {bit_depths_found} bits:")

    augment_wav_files('/tmp/dataset', sampling_rates[0])

    mfcc_data = process_directory_for_mfcc_features('/tmp/dataset')

    print(f"There are {len(mfcc_data)} MFCC features.")

    features, labels = extract_features_and_labels(mfcc_data, LABEL_MAP)

    # Normalize the features before splitting and padding
    features_normalized = normalize_features(features)

    # Split the dataset into training and a temporary set with stratification
    X_train, X_temp, y_train, y_temp = train_test_split(features_normalized, labels, test_size=0.3, stratify=labels,
                                                        random_state=42)

    # Now split the temporary set into validation and test sets, also with stratification
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

    # Determine the maximum length of the MFCC features in your dataset for padding
    # Assuming features_normalized is a list of all feature arrays
    max_length = max(len(feature) for feature in features_normalized)
    print(f"Maximum length of MFCC features: {max_length}")

    # Apply padding
    X_train = pad_features(X_train, max_length)
    X_val = pad_features(X_val, max_length)
    X_test = pad_features(X_test, max_length)

    # Convert labels to categorical (one-hot encoding)
    y_train_cat = to_categorical(y_train)
    y_val_cat = to_categorical(y_val)
    y_test_cat = to_categorical(y_test)

    # Assuming X_train has already been reshaped to fit the CNN input requirements
    # For CNNs, TensorFlow expects the data to be in the format of (samples, rows, cols, channels)
    # Since MFCC features have only one channel, we need to expand the dimensions of our data
    X_train_cnn = np.expand_dims(X_train, -1)
    X_val_cnn = np.expand_dims(X_val, -1)
    X_test_cnn = np.expand_dims(X_test, -1)

    input_shape = (X_train_cnn.shape[1], X_train_cnn.shape[2], 1)  # (MFCC features, time steps, 1)
    num_classes = y_train_cat.shape[1]

    print(f"Input shape: {input_shape}, Number of classes: {num_classes}")

    input_dimension = X_train_cnn.shape[1]
    print(f"Input dimension: {input_dimension}")

    best_model = train_and_select_best_model(X_train_cnn, y_train_cat, input_shape, num_classes)

    convert_and_save_model_to_tflite(best_model, X_train_cnn, "cat_sound_model.tflite")

    audio_constants = generate_cpp_definitions(FEATURE_SIZE, AUDIO_SAMPLING_FREQUENCY, avg_duration,
                                               FFT_WINDOW_SIZE, HOP_LENGTH)
    generate_header_file(LABEL_MAP, audio_constants)


if __name__ == '__main__':
    main()
