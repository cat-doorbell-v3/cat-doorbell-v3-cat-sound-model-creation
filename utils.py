import os
import shutil
import wave
import zipfile

import librosa
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score

import constants


def add_random_noise(data, noise_level=0.005):
    """
    Adds random noise to the data.

    Args:
        data (numpy.ndarray): The input data (spectrogram).
        noise_level (float): The amplitude of the noise to add (relative to data's range).

    Returns:
        numpy.ndarray: The data with added random noise.
    """
    noise = np.random.randn(*data.shape) * noise_level
    augmented_data = data + noise
    return np.clip(augmented_data, -np.inf, np.inf)  # Ensure values stay within a valid range


def augment_data(X, y):
    """
    Augments the dataset by applying random transformations.

    Args:
        X (numpy.ndarray): The input features (spectrograms).
        y (numpy.ndarray): The target labels.

    Returns:
        numpy.ndarray: The augmented input features.
        numpy.ndarray: The (unmodified) target labels.
    """
    X_augmented = np.array([add_random_noise(x) for x in X])
    return X_augmented, y


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


def audio_to_spectrogram(file_path, n_mels=constants.N_MELS, n_fft=constants.N_FFT, hop_length=constants.HOP_LENGTH,
                         max_pad_len=None):
    y, sr = librosa.load(file_path, sr=constants.SAMPLING_RATE, duration=constants.AUDIO_DURATION)
    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
    spectrogram = librosa.power_to_db(spectrogram, ref=np.max)

    # Pad or truncate the spectrogram to ensure a consistent shape
    pad_width = max_pad_len - spectrogram.shape[1]
    if pad_width > 0:  # Pad the spectrogram
        spectrogram = np.pad(spectrogram, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:  # Truncate the spectrogram
        spectrogram = spectrogram[:, :max_pad_len]

    return spectrogram


def find_max_spectrogram_length(dataset_path, categories, n_mels=constants.N_MELS, n_fft=constants.N_FFT,
                                hop_length=constants.HOP_LENGTH):
    max_len = 0
    for category in categories:
        category_path = os.path.join(dataset_path, category)
        for file in os.listdir(category_path):
            if file.endswith(".wav"):
                file_path = os.path.join(category_path, file)
                y, sr = librosa.load(file_path, sr=constants.SAMPLING_RATE, duration=constants.AUDIO_DURATION)
                spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=n_fft,
                                                             hop_length=hop_length)
                max_len = max(max_len, spectrogram.shape[1])
    return max_len


def load_dataset(dataset_path, categories, max_pad_len):
    X, y = [], []
    for label, category in enumerate(categories):
        category_path = os.path.join(dataset_path, category)
        for file in os.listdir(category_path):
            if file.endswith(".wav"):
                file_path = os.path.join(category_path, file)
                spectrogram = audio_to_spectrogram(file_path, max_pad_len=max_pad_len)
                X.append(spectrogram)
                y.append(label)
    return np.array(X), np.array(y)


# Convert the model to the TensorFlow Lite format with quantization

def convert_to_tflite(model, X_train, filename):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    def representative_dataset_gen():
        for i in range(100):
            # Ensure the sample input data is cast to FLOAT32
            yield [X_train[i].reshape(1, *X_train[i].shape).astype(np.float32)]

    converter.representative_dataset = representative_dataset_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model_quant = converter.convert()
    with open(filename, "wb") as f:
        f.write(tflite_model_quant)

    # Calculate the file size in kilobytes
    file_size_kb = os.path.getsize(filename) / 1024
    print(f"The size of the .tflite file is {file_size_kb:.2f} KB.")


def plot_model_fit(history_data):
    # Plotting training & validation accuracy values
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history_data['accuracy'])
    plt.plot(history_data['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')

    # Plotting training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history_data['loss'])
    plt.plot(history_data['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')

    plt.tight_layout()
    plt.show()


def get_metrics(best_model, X_val, y_val, X_train):
    # Generate predictions for the validation set
    y_val_pred = best_model.predict(X_val)
    # Convert predictions from one hot to class integers
    y_val_pred_classes = np.argmax(y_val_pred, axis=1)
    # Convert true validation labels from one hot to class integers
    y_val_true_classes = np.argmax(y_val, axis=1)

    # Calculate precision, recall, and F1 score
    precision = precision_score(y_val_true_classes, y_val_pred_classes, average='macro')
    recall = recall_score(y_val_true_classes, y_val_pred_classes, average='macro')
    f1 = f1_score(y_val_true_classes, y_val_pred_classes, average='macro')

    pos_class_probabilities = y_val_pred[:, 1]

    # Now we calculate the AUC-ROC using the true class labels and the predicted probabilities
    roc_auc = roc_auc_score(y_val_true_classes, pos_class_probabilities)

    # Now you can also include the ROC-AUC in the print statement at the end.
    print(f'Best Model - Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}, AUC-ROC: {roc_auc:.4f}')


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
