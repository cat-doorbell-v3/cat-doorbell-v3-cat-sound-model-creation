import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

import constants
import utils


def main():
    # Find the maximum spectrogram length
    max_pad_len = utils.find_max_spectrogram_length(constants.SAMPLES_DATASET_TEST_PATH, constants.DATASET_CATEGORIES)

    # Load and pad/truncate dataset
    X, y = utils.load_dataset(constants.SAMPLES_DATASET_TEST_PATH, constants.DATASET_CATEGORIES, max_pad_len)

    # Ensure X is reshaped correctly as per the model's input requirements
    X = X.reshape(*X.shape, 1)  # Add the channel dimension if necessary

    # Load the TFLite model
    interpreter, input_details, output_details = utils.load_tflite_model(constants.MODEL_FILE_NAME)

    # Make predictions on the test data
    predictions = utils.predict_with_tflite_model(interpreter, input_details, output_details, X)

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


if __name__ == '__main__':
    main()
