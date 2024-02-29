import os
import shutil
import zipfile

import librosa
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

import constants


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


def build_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(8, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Flatten(),
        Dense(16, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    return model


# Convert the model to the TensorFlow Lite format with quantization
def convert_to_tflite(model, X_train, filename):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # This is a simple way to implement representative dataset
    def representative_dataset_gen():
        for i in range(100):
            # Get sample input data as a numpy array in a method of your choosing.
            yield [X_train[i].reshape(1, *X_train[i].shape)]

    converter.representative_dataset = representative_dataset_gen

    # Ensure that if any ops can't be quantized, the converter throws an error
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    # Set the input and output tensors to int8 (for full integer quantization)
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model_quant = converter.convert()

    # Save the model to disk
    open(filename, "wb").write(tflite_model_quant)


def main():
    remove_directories([constants.MODEL_DATASET])

    unzip_file(constants.MODEL_DATASET_ZIP, '/tmp')

    # Find the maximum spectrogram length
    max_pad_len = find_max_spectrogram_length(constants.MODEL_DATASET_PATH, constants.MODEL_DATASET_CATEGORIES)
    print(f"Maximum pad length: {max_pad_len}")

    # Load and pad/truncate dataset
    X, y = load_dataset(constants.MODEL_DATASET_PATH, constants.MODEL_DATASET_CATEGORIES, max_pad_len)

    # Ensure each spectrogram has the same second dimension
    X = X.reshape(*X.shape, 1)  # Add channel dimension for CNN input

    # Split the dataset into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"Training samples: {X_train.shape[0]}, Validation samples: {X_val.shape[0]}")

    # Assuming binary classification (cat vs noise) and using 'binary_crossentropy' loss
    num_classes = 2  # Update if you have more classes
    input_shape = X_train.shape[1:]  # Should be (spectrogram_height, spectrogram_width, 1)

    model = build_model(input_shape, num_classes)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Model summary
    model.summary()

    # Train the model
    history = model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_val, y_val))

    convert_to_tflite(model, X_train, constants.MODEL_OUTPUT_FILE_NAME)


if __name__ == '__main__':
    main()
