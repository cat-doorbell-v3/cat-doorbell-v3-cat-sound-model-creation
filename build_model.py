import os
import pickle
import shutil
import subprocess
import zipfile

import librosa
import numpy as np
import soundfile as sf
import tensorflow as tf
from pydub import AudioSegment
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical


def get_audio_durations_and_average(directory_path):
    """
    Walks through a directory, calculates the duration of all .wav files,
    and computes the average duration.

    Args:
    directory_path (str): Path to the directory to search for .wav files.

    Returns:
    None
    """
    total_duration = 0
    file_count = 0

    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.lower().endswith('.wav'):
                file_path = os.path.join(root, file)
                y, sr = librosa.load(file_path, sr=None)  # Load with the original sampling rate
                duration = librosa.get_duration(y=y, sr=sr)
                print(f"File: {file_path}, Duration: {duration} seconds")
                total_duration += duration
                file_count += 1

    if file_count > 0:
        average_duration = total_duration / file_count
        print(f"Average Duration: {average_duration:.3f} seconds")
    else:
        print("No .wav files found in the directory.")


def get_audio_duration(file_path):
    """
    Returns the duration of the audio file in seconds.

    Args:
    file_path (str): Path to the audio file.

    Returns:
    float: Duration of the audio file in seconds.
    """
    y, sr = librosa.load(file_path, sr=None)  # Load the file with its original sampling rate
    duration = librosa.get_duration(y=y, sr=sr)
    return duration


def walk_and_get_durations(directory):
    """
    Walks through a directory, calculating the duration of all .wav files.

    Args:
    directory (str): The directory to walk through.

    Returns:
    None
    """
    total_duration = 0
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.wav'):
                file_path = os.path.join(root, file)
                duration = get_audio_duration(file_path)
                print(f"File: {file_path}, Duration: {duration} seconds")
                total_duration += duration
    print(f"Total duration of all .wav files: {total_duration} seconds")


def run_shell_command(command):
    """
    Runs a shell command.

    Args:
    command (str): The command to run.

    Returns:
    None
    """
    try:
        # Run the command
        subprocess.run(command, check=True, shell=True)
        print("Command executed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while executing the command: {e}")


def remove_specific_files_and_dirs(base_directory, dir_prefix, file_prefix):
    """
    Removes all directories and all files in the base directory that start with the specified prefixes.

    Args:
    base_directory (str): The base directory where the operation will be performed.
    dir_prefix (str): The prefix for directories to be removed.
    file_prefix (str): The prefix for files to be removed.

    Returns:
    None
    """
    # Loop through all items in the base directory
    for item in os.listdir(base_directory):
        item_path = os.path.join(base_directory, item)

        # Check if the item is a directory and starts with the specified dir_prefix
        if os.path.isdir(item_path) and item.startswith(dir_prefix):
            shutil.rmtree(item_path)
            print(f"Directory removed: {item_path}")

        # Check if the item is a file and starts with the specified file_prefix
        elif os.path.isfile(item_path) and item.startswith(file_prefix):
            os.remove(item_path)
            print(f"File removed: {item_path}")


# Example usage
# remove_specific_files_and_dirs('/tmp', 'CAT_', 'mfcc_')

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


# Example usage
# unzip_file('path/to/your/file.zip', 'path/to/extract/to')

def copy_and_convert_directory(source_dir, target_dir):
    """
    Copy the directory structure from source to target and convert all MP3 files to WAV.

    Parameters:
    - source_dir: The path to the source directory.
    - target_dir: The path to the target directory where you want to copy and convert the files.
    """
    for root, dirs, files in os.walk(source_dir):
        # Construct the path to the destination directory
        dest_dir = os.path.join(target_dir, os.path.relpath(root, source_dir))

        # Create the destination directory
        os.makedirs(dest_dir, exist_ok=True)

        for file in files:
            # Full path for source and destination files
            src_file_path = os.path.join(root, file)
            dest_file_path = os.path.join(dest_dir, file)

            # If the file is an MP3, convert it to WAV
            if src_file_path.lower().endswith('.mp3'):
                audio = AudioSegment.from_mp3(src_file_path)
                dest_file_path = dest_file_path.rsplit('.', 1)[0] + '.wav'  # Change file extension to .wav
                audio.export(dest_file_path, format="wav")
            else:
                # If not an MP3 file, just copy the file as is
                shutil.copy2(src_file_path, dest_file_path)


def add_noise(data, noise_factor=0.005):
    noise = np.random.randn(len(data))
    augmented_data = data + noise_factor * noise
    return augmented_data.astype(type(data[0]))


def shift_pitch(data, sampling_rate, pitch_factor=5):
    return librosa.effects.pitch_shift(y=data, sr=sampling_rate, n_steps=pitch_factor)


def change_speed(data, speed_factor=1.25):
    return np.interp(np.arange(0, len(data), speed_factor), np.arange(0, len(data)), data)


def augment_and_save(file_path, target_dir, sampling_rate):
    data, _ = librosa.load(file_path, sr=sampling_rate)
    noise_data = add_noise(data)
    pitch_shifted_data = shift_pitch(data, sampling_rate)
    speed_changed_data = change_speed(data)

    base_name = os.path.splitext(os.path.basename(file_path))[0]
    sf.write(os.path.join(target_dir, f"{base_name}_original.wav"), data, sampling_rate)
    sf.write(os.path.join(target_dir, f"{base_name}_noise.wav"), noise_data, sampling_rate)
    sf.write(os.path.join(target_dir, f"{base_name}_pitch.wav"), pitch_shifted_data, sampling_rate)
    sf.write(os.path.join(target_dir, f"{base_name}_speed.wav"), speed_changed_data, sampling_rate)


def copy_and_augment_directory(source_dir, target_dir, sampling_rate=22050):
    for root, dirs, files in os.walk(source_dir):
        rel_path = os.path.relpath(root, source_dir)
        current_target_dir = os.path.join(target_dir, rel_path)
        os.makedirs(current_target_dir, exist_ok=True)

        for file in files:
            if file.lower().endswith('.wav'):
                file_path = os.path.join(root, file)
                augment_and_save(file_path, current_target_dir, sampling_rate)


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


def save_features(features, output_file='/tmp/mfcc_features.pkl'):
    """
    Save the extracted features to a file.

    Parameters:
    - features: The dictionary of features to save.
    - output_file: The path to the output file.
    """
    with open(output_file, 'wb') as f:
        pickle.dump(features, f)


def create_cnn_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(16, kernel_size=(3, 3), activation='relu', input_shape=input_shape, padding='same'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    return model


def normalize_features(features):
    normalized_features = []
    for feature in features:
        if feature.size == 0:  # Skip empty features if there are any
            continue
        scaler = StandardScaler()
        normalized_feature = scaler.fit_transform(feature)  # Normalize each feature array
        normalized_features.append(normalized_feature)
    return normalized_features


# Define a generator function that provides the representative dataset for calibration
def representative_dataset_gen():
    for input_value in X_train_cnn[:100]:  # Assuming you're using a subset of your training data
        # Model has only one input so each data point has to be in a tuple
        yield [np.expand_dims(input_value, axis=0).astype(np.float32)]


remove_specific_files_and_dirs('/tmp', 'CAT_', 'mfcc_')

unzip_file('CAT_SOUND_DB_SAMPLES.zip', '/tmp/CAT_SOUND_DB_SAMPLES')

source_dir = '/tmp/CAT_SOUND_DB_SAMPLES'
target_dir = '/tmp/CAT_SOUND_DB_SAMPLES_WAV'
print(f"Converting files from {source_dir} to {target_dir}...")
copy_and_convert_directory(source_dir, target_dir)

#
directory = '/tmp/CAT_SOUND_DB_SAMPLES_WAV'
get_audio_durations_and_average(directory)

# Usage
source_dir = '/tmp/CAT_SOUND_DB_SAMPLES_WAV'
target_dir = '/tmp/CAT_SOUND_DB_SAMPLES_AUGMENTED'
print(f"Augmenting files from {source_dir} to {target_dir}...")
copy_and_augment_directory(source_dir, target_dir)

# Example usage
directory = '/tmp/CAT_SOUND_DB_SAMPLES_AUGMENTED'
features = process_directory(directory)
print(f"Extracted features for {len(features)} files.")
save_features(features, '/tmp/mfcc_features.pkl')

print(f"Extracted features for {len(features)} files.")

##
# 1. Load the data
#
# Load your serialized MFCC features
with open('/tmp/mfcc_features.pkl', 'rb') as f:
    data = pickle.load(f)

print(f"Loaded {len(data)} MFCC features.")

features = []
labels = []
label_map = {'Angry': 0, 'Defense': 1, 'Fighting': 2, 'Happy': 3, 'HuntingMind': 4,
             'Mating': 5, 'MotherCall': 6, 'Paining': 7, 'Resting': 8, 'Warning': 9}
for file_path, mfcc_features in data.items():
    features.append(mfcc_features)
    label = file_path.split('/')[4]  # Assuming the label is in the third segment of the path
    labels.append(label_map[label])

features = np.array(features, dtype=object)
labels = np.array(labels)

# Normalize the features before splitting and padding
features_normalized = normalize_features(features)

#
# 2. Split the data into training and testing sets
#

# Split the dataset into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(features_normalized, labels, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Determine the maximum length of the MFCC features in your dataset for padding
max_length = max([len(feature) for feature in features_normalized])
X_train = np.array([np.pad(feature, ((0, max_length - len(feature)), (0, 0)), mode='constant') for feature in X_train])
X_val = np.array([np.pad(feature, ((0, max_length - len(feature)), (0, 0)), mode='constant') for feature in X_val])
X_test = np.array([np.pad(feature, ((0, max_length - len(feature)), (0, 0)), mode='constant') for feature in X_test])

#
# 3. Format data for training
#
# Convert labels to categorical (one-hot encoding)
y_train_cat = to_categorical(y_train)
y_val_cat = to_categorical(y_val)
y_test_cat = to_categorical(y_test)

#
# 4. Define the CNN model
#
# Assuming X_train has already been reshaped to fit the CNN input requirements
# For CNNs, TensorFlow expects the data to be in the format of (samples, rows, cols, channels)
# Since MFCC features have only one channel, we need to expand the dimensions of our data
X_train_cnn = np.expand_dims(X_train, -1)
X_val_cnn = np.expand_dims(X_val, -1)
X_test_cnn = np.expand_dims(X_test, -1)

input_shape = (X_train_cnn.shape[1], X_train_cnn.shape[2], 1)  # (MFCC features, time steps, 1)
num_classes = y_train_cat.shape[1]

model = create_cnn_model(input_shape, num_classes)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Print model summary
model.summary()

#
# 5. Train the model
#
history = model.fit(X_train_cnn, y_train_cat, batch_size=32, epochs=30, validation_data=(X_val_cnn, y_val_cat))

#
# 6. Evaluate the model
#

test_loss, test_accuracy = model.evaluate(X_test_cnn, y_test_cat)
print(f"Test accuracy: {test_accuracy:.4f}, Test loss: {test_loss:.4f}")

# Assuming `model` is your Keras model
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# To enable full quantization, set the optimizations flag to use default optimizations
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Set the representative dataset for quantization
converter.representative_dataset = representative_dataset_gen

# Ensure that if any ops can't be quantized, the converter throws an error
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

# Set the input and output tensors to uint8 (optional)
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8

# Convert the model
tflite_quant_model = converter.convert()

# Save the fully quantized model
with open("cat_sound_model_quant.tflite", "wb") as f:
    f.write(tflite_quant_model)

# Following the conversion, you can proceed with the rest of your code to test the TFLite model

# Load the TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path="cat_sound_model_quant.tflite")
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test the model on input data (example)
input_shape = input_details[0]['shape']
input_data = np.array(np.expand_dims(X_test_cnn[0], 0), dtype=np.uint8)
interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()

# Extract the output
output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data)

predicted_category_index = np.argmax(output_data)
predicted_category_confidence = np.max(output_data)
print(f"Predicted Category Index: {predicted_category_index}, Confidence: {predicted_category_confidence:.2f}")

# Assuming `label_map` is a dictionary mapping category names to indexes
inverse_label_map = {v: k for k, v in label_map.items()}
predicted_category_name = inverse_label_map[predicted_category_index]
print(f"Predicted Category: {predicted_category_name}")

#
#  7. Save the model
# xxd -i cat_sound_model_quant.tflite > cat_sound_model.cc
#
command = "xxd -i cat_sound_model_quant.tflite > cat_sound_model.cc"
run_shell_command(command)
