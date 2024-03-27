"""
Yamnet was downloaded from here:
https://www.kaggle.com/models/google/yamnet/frameworks/tensorFlow2
"""
import os

import librosa
import tensorflow as tf

# Path to the directory containing .wav files
wav_files_directory = '/tmp/google-audioset-samples/meow'
yamnet_model_directory = '/Users/tennis/sound-library/yamnet/archive'

# Load the YAMNet model
yamnet_model = tf.saved_model.load(yamnet_model_directory)
infer = yamnet_model.signatures["serving_default"]

CAT_ID_INDEX = 76
CAT_SCORE_THRESHOLD = 0.8  # 80%

# Load the YAMNet model
yamnet_model = tf.saved_model.load(yamnet_model_directory)
infer = yamnet_model.signatures["serving_default"]

# Get the input tensor name and shape dynamically
input_tensor_name = list(infer.structured_input_signature[1].keys())[0]
input_tensor_shape = infer.structured_input_signature[1][input_tensor_name].shape


# Function to load and preprocess audio files
def load_audio_file(file_path, target_sr=16000):
    # Load an audio file as a floating point time series
    audio, sr = librosa.load(file_path, sr=target_sr)
    return audio


# Go through each file in the directory
for filename in os.listdir(wav_files_directory):
    if filename.endswith('.wav'):
        file_path = os.path.join(wav_files_directory, filename)

        # Load and preprocess the audio file
        audio = load_audio_file(file_path)

        # Prepare the audio tensor according to the model's expected input
        audio_tensor = tf.constant(audio, dtype=tf.float32)

        # Run inference
        output = infer(waveform=audio_tensor)

        # Extract scores from the output
        scores = output['output_0'].numpy()[0]

        if scores[CAT_ID_INDEX] < CAT_SCORE_THRESHOLD:
            os.remove(file_path)
