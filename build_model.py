from keras.regularizers import l2
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

import constants
import utils


def main():
    utils.remove_directories([constants.MODEL_DATASET])

    utils.unzip_file(constants.MODEL_DATASET_ZIP, '/tmp')

    # Find the maximum spectrogram length
    max_pad_len = utils.find_max_spectrogram_length(constants.MODEL_DATASET_PATH, constants.MODEL_DATASET_CATEGORIES)
    print(f"Maximum pad length: {max_pad_len}")

    # Load and pad/truncate dataset
    X, y = utils.load_dataset(constants.MODEL_DATASET_PATH, constants.MODEL_DATASET_CATEGORIES, max_pad_len)

    # Ensure each spectrogram has the same second dimension
    X = X.reshape(*X.shape, 1)  # Add channel dimension for CNN input

    X_augmented, y_augmented = utils.augment_data(X, y)
    # Split the dataset into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_augmented, y_augmented, test_size=0.2, random_state=42)

    num_classes = len(constants.MODEL_DATASET_CATEGORIES)

    # Convert labels to one-hot encoding
    y_train = to_categorical(y_train, num_classes)
    y_val = to_categorical(y_val, num_classes)

    input_shape = X_train.shape[1:]  # Should be (spectrogram_height, spectrogram_width, 1)

    model = Sequential([
        Conv2D(8, kernel_size=(3, 3), activation='relu', input_shape=input_shape, kernel_regularizer=l2(0.001)),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.3),  # Increased dropout rate
        Flatten(),
        Dense(10, activation='relu', kernel_regularizer=l2(0.001)),  # Reduced complexity and added regularizer
        Dropout(0.6),  # Increased dropout rate
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Model summary
    model.summary()

    # Define early stopping callback
    early_stopping = EarlyStopping(
        monitor='val_loss',  # Monitor the validation set loss
        patience=3,  # Number of epochs with no improvement after which training will be stopped
        restore_best_weights=True  # Restores model weights from the epoch with the best value of the monitored quantity
    )

    # Include early stopping in the fit function
    history = model.fit(
        X_train, y_train,
        epochs=30,
        batch_size=32,
        validation_data=(X_val, y_val),
        shuffle=True,
        callbacks=[early_stopping]
    )

    # Train the model
    utils.plot_model_fit(history.history)

    utils.convert_to_tflite(model, X_train, constants.MODEL_OUTPUT_FILE_NAME)


if __name__ == '__main__':
    main()
