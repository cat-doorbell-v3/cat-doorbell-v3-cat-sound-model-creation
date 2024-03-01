import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

import constants
import utils


# ...

def load_tflite_model(tflite_model_path):
    # Load the TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    return interpreter, input_details, output_details


def predict_with_tflite_model(interpreter, input_details, output_details, X_test):
    # Ensure that the input data is float32
    X_test = X_test.astype('float32')

    # Initialize an array for storing predictions
    predictions = []

    # Iterate over each test sample and make predictions
    for i in range(len(X_test)):
        # Set the tensor to point to the input data to be inferred
        interpreter.set_tensor(input_details[0]['index'], [X_test[i]])
        # Run the inference
        interpreter.invoke()
        # Extract the output
        output_data = interpreter.get_tensor(output_details[0]['index'])
        predictions.append(output_data[0])

    return np.array(predictions)


def predict_with_tflite_model(interpreter, input_details, output_details, X_test):
    # Depending on the model, the input data might need to be quantized to int8 or uint8
    # Check `input_details[0]['dtype']` to determine the correct data type
    # Ensure that the input data type matches the model's expected input
    X_test = X_test.astype(input_details[0]['dtype'])

    predictions = []

    for i in range(len(X_test)):
        # If the model expects quantized input, quantize the test data accordingly before setting the tensor
        # This depends on the `input_details[0]['quantization']` parameters

        interpreter.set_tensor(input_details[0]['index'], [X_test[i]])
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        predictions.append(output_data[0])

    return np.array(predictions)


# ... (rest of your code)

def main():
    utils.remove_directories([constants.TEST_DATASET])

    utils.unzip_file(constants.TEST_DATASET_ZIP, '/tmp')

    # Find the maximum spectrogram length
    max_pad_len = utils.find_max_spectrogram_length(constants.TEST_DATASET_PATH, constants.TEST_DATASET_CATEGORIES)
    print(f"Maximum pad length: {max_pad_len}")

    # Load and pad/truncate dataset
    X, y = utils.load_dataset(constants.TEST_DATASET_PATH, constants.TEST_DATASET_CATEGORIES, max_pad_len)

    # Ensure X is reshaped correctly as per the model's input requirements
    X = X.reshape(*X.shape, 1)  # Add the channel dimension if necessary

    # Load the TFLite model
    interpreter, input_details, output_details = load_tflite_model(constants.MODEL_OUTPUT_FILE_NAME)

    # Make predictions on the test data
    predictions = predict_with_tflite_model(interpreter, input_details, output_details, X)

    # If the model outputs probabilities, use argmax to get the predicted class label
    predicted_labels = np.argmax(predictions, axis=1)

    # Calculate the accuracy
    test_accuracy = accuracy_score(y, predicted_labels)
    print(f"Test accuracy: {test_accuracy}")
    cm = confusion_matrix(y, predicted_labels)
    sns.heatmap(cm, annot=True, fmt='d')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()


if __name__ == "__main__":
    main()
