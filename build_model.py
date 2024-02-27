import os
import shutil
import wave
import zipfile

import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import tensorflow as tf
from keras.callbacks import EarlyStopping
from keras.regularizers import l2
from pydub import AudioSegment
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

MODEL_NAME = "cat_sound_model.tflite"

AUDIO_SAMPLING_FREQUENCY = 44100
FEATURE_SIZE = 13
FFT_WINDOW_SIZE = 2048
HOP_LENGTH = 512

AUDIO_CONSTANTS = {
    "kMaxAudioSampleSize": 0,
    "kAudioSampleFrequency": AUDIO_SAMPLING_FREQUENCY,
    "kFeatureSize": FEATURE_SIZE,
    "kFeatureCount": 0,
    "kFeatureElementCount": 0,
    "kFeatureStrideMs": 0,
    "kFeatureDurationMs": 0
}

LABEL_MAP = {
    'Angry': 0, 'Defense': 1, 'Fighting': 2, 'Happy': 3, 'HuntingMind': 4,
    'Mating': 5, 'MotherCall': 6, 'Paining': 7, 'Resting': 8, 'Warning': 9
}

NEW_LABEL_MAP = {
    'Angry': 0, 'Defense': 1, 'Fighting': 2
}

IGNORE = [
    'Happy', 'Mating', 'MotherCall', 'Paining', 'Resting', 'Warning', 'HuntingMind'
]


def plot_training_history(history):
    # Plot training & validation accuracy values
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.tight_layout()
    plt.show()


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


def get_bit_depth(wav_file_path):
    try:
        with wave.open(wav_file_path, 'r') as wav_file:
            sample_width_bytes = wav_file.getsampwidth()
            bit_depth = sample_width_bytes * 8  # Convert bytes to bits
            return bit_depth
    except wave.Error:
        print(f"Error reading {wav_file_path}. It might not be a valid WAV file.")
        return None


def walk_directory_for_wavs(directory_path):
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
    return bit_depths


def standardize_length_and_save(audio_path, target_path, target_length=3.25, sr=44100):
    """
    Load an audio file, standardize its length, and save it to a target path.

    Args:
        audio_path (str): Path to the source audio file.
        target_path (str): Path to save the standardized audio file.
        target_length (float): Target length of the audio in seconds.
        sr (int): Sampling rate to use for loading and saving the audio.
    """
    audio, _ = librosa.load(audio_path, sr=sr)
    target_samples = int(target_length * sr)

    if len(audio) < target_samples:
        # Pad short files
        pad_length = target_samples - len(audio)
        audio = np.pad(audio, (0, pad_length), 'constant')
    elif len(audio) > target_samples:
        # Trim long files
        audio = audio[:target_samples]

    # Ensure the target directory exists
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    # Save the standardized audio file
    sf.write(target_path, audio, sr)


def standardize_directory_audio_lengths(source_dir, target_dir, target_length=3.25, sr=44100):
    """
    Walk through a directory tree, standardizing the length of all audio files and saving them to a parallel
    structure in a target directory.

    Args:
        source_dir (str): Root directory to search for audio files.
        target_dir (str): Root directory to save standardized audio files.
        target_length (float): Target length in seconds for audio files.
        sr (int): Sampling rate to use for audio files.
    """
    for subdir, _, files in os.walk(source_dir):
        for file in files:
            if file.lower().endswith('.wav'):  # Process only WAV files
                source_path = os.path.join(subdir, file)
                relative_path = os.path.relpath(source_path, source_dir)
                target_path = os.path.join(target_dir, relative_path)
                standardize_length_and_save(source_path, target_path, target_length, sr)


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
    average_duration = None

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

    return average_duration




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


def copy_and_convert_directory(source_dir, target_dir, ignore_dirs=[]):
    """
    Copy the directory structure from source to target and convert all MP3 files to WAV,
    ignoring specific subdirectories.

    Parameters:
    - source_dir: The path to the source directory.
    - target_dir: The path to the target directory where you want to copy and convert the files.
    - ignore_dirs: A list of directory names to ignore. These should be just the directory names,
                   not the full path.
    """
    for root, dirs, files in os.walk(source_dir, topdown=True):
        # Skip subdirectories that are in the ignore list
        dirs[:] = [d for d in dirs if d not in ignore_dirs]

        # Construct the path to the destination directory
        dest_dir = os.path.join(target_dir, os.path.relpath(root, source_dir))

        # Check if the destination directory is within an ignored path
        if any(ignored_dir in dest_dir for ignored_dir in ignore_dirs):
            continue

        # Create the destination directory if it does not exist
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


def copy_and_augment_directory(source_dir, target_dir, sampling_rate=AUDIO_SAMPLING_FREQUENCY):
    for root, dirs, files in os.walk(source_dir):
        rel_path = os.path.relpath(root, source_dir)
        current_target_dir = os.path.join(target_dir, rel_path)
        os.makedirs(current_target_dir, exist_ok=True)

        for file in files:
            if file.lower().endswith('.wav'):
                file_path = os.path.join(root, file)
                augment_and_save(file_path, current_target_dir, sampling_rate)


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

samples_dir = '/tmp/CAT_SOUND_DB_SAMPLES'
wav_dir = '/tmp/CAT_SOUND_DB_SAMPLES_WAV'
std_dir = '/tmp/CAT_SOUND_DB_SAMPLES_STANDARDIZED'
aug_dir = '/tmp/CAT_SOUND_DB_SAMPLES_AUGMENTED'

print(f"Converting files from {samples_dir} to {wav_dir}...")
copy_and_convert_directory(samples_dir, wav_dir, ignore_dirs=IGNORE)
#
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

print(f"Standardizing audio duration in files from {wav_dir} to {std_dir}...")
standardize_directory_audio_lengths(wav_dir, std_dir, target_length=average_duration, sr=sampling_rates[0])

print(f"Augmenting files from {std_dir} to {aug_dir}...")
copy_and_augment_directory(std_dir, aug_dir)

data = process_directory(aug_dir)

print(f"There are {len(data)} MFCC features.")

features = []
labels = []
for file_path, mfcc_features in data.items():
    features.append(mfcc_features)
    label = file_path.split('/')[4]  # Assuming the label is in the forth segment of the path
    labels.append(LABEL_MAP[label])

features = np.array(features, dtype=object)
# Assuming labels is a NumPy array after this conversion
labels = np.array(labels)

# Normalize the features before splitting and padding
features_normalized = normalize_features(features)

# Split the dataset into training and a temporary set with stratification
X_train, X_temp, y_train, y_temp = train_test_split(features_normalized, labels, test_size=0.3, stratify=labels,
                                                    random_state=42)

# Now split the temporary set into validation and test sets, also with stratification
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

# Determine the maximum length of the MFCC features in your dataset for padding
max_length = max(len(feature) for feature in features_normalized)

# Apply padding
X_train = np.array([np.pad(feature, ((0, max_length - len(feature)), (0, 0)), 'constant') for feature in X_train])
X_val = np.array([np.pad(feature, ((0, max_length - len(feature)), (0, 0)), 'constant') for feature in X_val])
X_test = np.array([np.pad(feature, ((0, max_length - len(feature)), (0, 0)), 'constant') for feature in X_test])

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

print(f"Input shape: {input_shape}, Number of classes: {num_classes}")

input_dimension = X_train_cnn.shape[1]

model = Sequential()
model.add(Flatten(input_shape=(119, 13, 1)))  # Adjust the input_shape to match your data
model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print("model compiled. Here is the summary:")
model.summary()
print("end model summary")

#
# 5. Train the model
#

# Define early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=5)

# Fit the model with early stopping
history = model.fit(X_train_cnn, y_train_cat, batch_size=32, epochs=30, validation_data=(X_val_cnn, y_val_cat),
                    callbacks=[early_stopping])
plot_training_history(history)
#
# 6. Evaluate the model
#
test_loss, test_accuracy = model.evaluate(X_test_cnn, y_test_cat)
print(f"Test accuracy: {test_accuracy:.4f}, Test loss: {test_loss:.4f}")

# Assuming `model` is your Keras model
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Enable full integer quantization
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Set the representative dataset for calibration
converter.representative_dataset = representative_dataset_gen

for data in representative_dataset_gen():
    print(data[0].shape)  # Should match the model's input shape
    break  # Just to check the first batch

# Ensure that if any ops can't be quantized, the converter throws an error
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

# Specify the input and output types to be int8
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

# Convert the model to a TensorFlow Lite model
tflite_quant_model = converter.convert()

# Save the quantized model
with open(MODEL_NAME, "wb") as f:
    f.write(tflite_quant_model)

generate_header_file(LABEL_MAP, AUDIO_CONSTANTS)

#
# # Load the TFLite model
# interpreter = tf.lite.Interpreter(model_path=MODEL_NAME)
# interpreter.allocate_tensors()
#
# # Get input tensor details
# input_details = interpreter.get_input_details()
# print(f"Input details for C:\n{input_details}")
