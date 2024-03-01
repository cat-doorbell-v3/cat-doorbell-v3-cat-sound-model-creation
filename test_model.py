import time

import librosa
import numpy as np
import sounddevice as sd
import tensorflow as tf

import constants


def record_audio(duration, samplerate=8000):
    print("Recording...")
    recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='float32')
    sd.wait()  # Wait for the recording to finish
    print("Recording stopped.")
    return np.squeeze(recording)  # Remove channel dimension


def preprocess_audio(audio, sr):
    # Generate a mel-spectrogram
    spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=constants.N_MELS, n_fft=constants.N_FFT,
                                                 hop_length=constants.HOP_LENGTH)
    spectrogram = librosa.power_to_db(spectrogram, ref=np.max)

    # Pad or truncate the spectrogram to ensure a consistent shape
    if spectrogram.shape[1] < constants.MAX_PAD_LENGTH:
        spectrogram = np.pad(spectrogram, pad_width=((0, 0), (0, constants.MAX_PAD_LENGTH - spectrogram.shape[1])),
                             mode='constant')
    else:
        spectrogram = spectrogram[:, :constants.MAX_PAD_LENGTH]

    # Add channel dimension for CNN input
    spectrogram = np.expand_dims(spectrogram, axis=0)  # Add batch dimension
    spectrogram = np.expand_dims(spectrogram, axis=-1)  # Add channel dimension

    return spectrogram


def load_interpreter(model_path):
    # Load the TFLite model
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter


def run_inference(audio_features_quantized, interpreter):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("Input shape to the model:", audio_features_quantized.shape)  # Debugging line
    interpreter.set_tensor(input_details[0]['index'], audio_features_quantized)

    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data


def quantize_input(features, input_details):
    # Obtain the quantization parameters
    scale, zero_point = input_details[0]['quantization']

    # Quantize the input features
    features_quantized = features / scale + zero_point
    features_quantized = np.round(features_quantized).astype(np.int8)  # Convert to int8

    return features_quantized


def print_model_input_details(interpreter):
    input_details = interpreter.get_input_details()
    print("Model Input Details:")
    for detail in input_details:
        print(detail)


def old_interpret_prediction(prediction):
    # Assuming the "cat sound" class is at index 1
    cat_sound_prob = prediction[0][1]  # Adjust based on your model's output
    threshold = 0.5  # Threshold can be adjusted based on your requirements

    if cat_sound_prob > threshold:
        return "👍"  # Thumbs up emoji
    else:
        return "no"


def dequantize_output(output_data, output_details):
    # Extract scale and zero-point from output details
    scale = output_details[0]['quantization_parameters']['scales']
    zero_point = output_details[0]['quantization_parameters']['zero_points']

    # Dequantize the output data
    dequantized_output = scale * (output_data.astype(np.float32) - zero_point)
    return dequantized_output


def interpret_prediction(output_data, output_details):
    # Dequantize model output
    dequantized_output = dequantize_output(output_data, output_details)

    # Apply softmax to convert logits to probabilities (if necessary)
    probabilities = tf.nn.softmax(dequantized_output).numpy()

    # Assuming binary classification with 'cat sound' as the second class
    cat_sound_prob = probabilities[0][1]
    print("Dequantized probability of cat sound:", cat_sound_prob)

    return "👍" if cat_sound_prob > 0.25 else "no"




def main():
    interpreter = load_interpreter(constants.MODEL_OUTPUT_FILE_NAME)
    print_model_input_details(interpreter)

    try:
        while True:
            audio = record_audio(constants.AUDIO_DURATION, constants.SAMPLING_RATE)
            audio_features = preprocess_audio(audio, constants.SAMPLING_RATE)
            print("Preprocessed audio shape:", audio_features.shape)  # Verify shape

            audio_features_quantized = quantize_input(audio_features, interpreter.get_input_details())
            print("Input shape to the model:", audio_features_quantized.shape)  # Verify shape after quantization

            # Inside your main function or wherever you handle model inference
            output_data = run_inference(audio_features_quantized, interpreter)
            output_details = interpreter.get_output_details()
            cat_sound_detected = interpret_prediction(output_data, output_details)
            print(f"Cat sound detected? {cat_sound_detected}")

            time.sleep(2)  # Pause for 2 seconds before next recording
    except KeyboardInterrupt:
        print("Program terminated by user.")


if __name__ == "__main__":
    main()
