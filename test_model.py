import time  # Import the time module for the sleep function

import librosa
import numpy as np
import sounddevice as sd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

AUDIO_SAMPLING_FREQUENCY = 8000
FEATURE_SIZE = 13
FFT_WINDOW_SIZE = 512
HOP_LENGTH = 256
MODEL_NAME = "cat_sound_model.tflite"


def run_inference(audio_features, interpreter):
    input_details = interpreter.get_input_details()  # Get the input tensor details
    output_details = interpreter.get_output_details()  # Get the output tensor details

    # Check the shape of your input tensor and reshape if necessary
    audio_features_reshaped = np.expand_dims(audio_features, axis=-1)  # Match the model's input shape

    # Set the value of the input tensor
    interpreter.set_tensor(input_details[0]['index'], audio_features_reshaped)

    # Run the inference
    interpreter.invoke()

    # Retrieve the output of the model
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Process the model's output if necessary (e.g., applying softmax for probabilities)
    return output_data


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


def record_audio(duration, samplerate=8000):
    """
    Records audio from the microphone for a given duration and samplerate.

    Parameters:
    - duration: The duration to record in seconds.
    - samplerate: The samplerate for the recording.

    Returns:
    - A NumPy array containing the recorded audio data.
    """
    print(f"Recording for {duration} seconds...")
    recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='float32')
    sd.wait()  # Wait until the recording is finished
    # print("Recording finished")
    return recording.flatten()


def preprocess_audio_data(audio_data, sample_rate, interpreter):
    # Adjusted to handle raw audio data
    mfcc_features = extract_mfcc_features(audio_data, sample_rate)
    mfcc_features_normalized = normalize_features([mfcc_features])
    input_details = interpreter.get_input_details()
    max_length = input_details[0]['shape'][1]
    mfcc_features_padded = pad_features(mfcc_features_normalized, max_length)
    return mfcc_features_padded


def extract_mfcc_features(audio_data, sample_rate, n_mfcc=FEATURE_SIZE, n_fft=FFT_WINDOW_SIZE, hop_length=HOP_LENGTH):
    # Adjusted to work directly with audio data and sample rate
    mfcc_features = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=n_mfcc, n_fft=n_fft,
                                         hop_length=hop_length)
    return mfcc_features.T


def softmax(x):
    """Compute softmax values using TensorFlow for enhanced numerical stability."""
    return tf.nn.softmax(x).numpy()


def is_cat_sound(prediction):
    # Convert prediction from int8 to float32 before softmax
    prediction_float = prediction.astype(np.float32)

    # Apply softmax to convert logits to probabilities
    probabilities = softmax(prediction_float)
    cat_sound_prob = probabilities[0][1]  # Adjust index based on your model's output structure

    # Define a threshold for deciding
    threshold = 0.8  # This is arbitrary; adjust based on your model's performance and requirements

    if cat_sound_prob > threshold:
        return "yes 👍"
    else:
        return "no"


def main():
    samplerate = 8000  # Use the same sample rate as your training data

    # Load the TFLite model and allocate tensors
    interpreter = tf.lite.Interpreter(model_path=MODEL_NAME)
    interpreter.allocate_tensors()

    print("Press Ctrl+C to stop the recording loop...")

    try:
        while True:
            # Record a snippet of audio from the microphone
            recorded_audio = record_audio(duration=5, samplerate=samplerate)

            # Process the audio and extract features
            audio_features = preprocess_audio_data(recorded_audio, samplerate, interpreter)

            # Scale the normalized features to INT8 range
            audio_features_scaled = np.round(audio_features * 127).astype(np.int8)

            # Use the run_inference function to make predictions with the correctly typed and scaled features
            prediction = run_inference(audio_features_scaled, interpreter)

            # Determine if it is a cat sound
            print("Is it a cat sound?", is_cat_sound(prediction))

            # print("Waiting for 5 seconds before the next recording...")
            time.sleep(5)  # Wait for 5 seconds before next iteration

    except KeyboardInterrupt:
        print("Recording loop stopped by user.")


if __name__ == '__main__':
    main()
