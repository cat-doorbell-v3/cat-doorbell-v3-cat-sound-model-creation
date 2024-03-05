import numpy as np
from keras.regularizers import l2
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import save_model
from tensorflow.keras.utils import to_categorical

import constants
import utils


def main():
    utils.remove_directories([constants.MODEL_DATASET])

    utils.unzip_file(constants.MODEL_DATASET_ZIP, '/tmp')

    # Find the maximum spectrogram length
    max_pad_len = utils.find_max_spectrogram_length(constants.MODEL_DATASET_PATH, constants.DATASET_CATEGORIES)
    print(f"Maximum pad length: {max_pad_len}")

    # Load and pad/truncate dataset
    X, y = utils.load_dataset(constants.MODEL_DATASET_PATH, constants.DATASET_CATEGORIES, max_pad_len)

    # Ensure each spectrogram has the same second dimension
    X = X.reshape(*X.shape, 1)  # Add channel dimension for CNN input

    X_augmented, y_augmented = utils.augment_data(X, y)
    # Split the dataset into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_augmented, y_augmented, test_size=0.2, random_state=42)

    num_classes = len(constants.DATASET_CATEGORIES)

    input_shape = X_train.shape[1:]  # Should be (spectrogram_height, spectrogram_width, 1)

    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    best_model = None
    best_model_history = None
    best_accuracy = 0
    fold_no = 1
    fold_metrics = []
    for train, val in kfold.split(X_augmented, y_augmented):
        # Split the dataset into the current train and validation sets
        X_train, X_val = X_augmented[train], X_augmented[val]
        y_train, y_val = y_augmented[train], y_augmented[val]

        # Convert labels to one-hot encoding
        y_train = to_categorical(y_train, num_classes)
        y_val = to_categorical(y_val, num_classes)

        model = Sequential([
            Conv2D(8, kernel_size=(3, 3), activation='relu', input_shape=input_shape, kernel_regularizer=l2(0.001)),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.2),
            Flatten(),
            Dense(10, activation='relu', kernel_regularizer=l2(0.001)),
            Dropout(0.5),
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
            patience=1,  # Number of epochs with no improvement after which training will be stopped
            restore_best_weights=True
            # Restores model weights from the epoch with the best value of the monitored quantity
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

        # Collect the metrics from the history object
        fold_metrics.append({
            'fold': fold_no,
            'accuracy': history.history['accuracy'],
            'val_accuracy': history.history['val_accuracy'],
            'loss': history.history['loss'],
            'val_loss': history.history['val_loss']
        })

        # Check if this is the best model so far
        mean_val_accuracy = np.mean(history.history['val_accuracy'])
        if mean_val_accuracy > best_accuracy:
            best_accuracy = mean_val_accuracy
            best_model = model
            best_model_history = history.history  # Store the current best model's history
            # Saving the best model's weights
            model_save_path = '/tmp/best_model.h5'
            save_model(model, model_save_path)

        fold_no += 1

    # After all folds are completed, we have the best model based on validation accuracy
    print(f"Best model achieved an average validation accuracy of: {best_accuracy}")

    utils.plot_model_fit(best_model_history)

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

    utils.convert_to_tflite(best_model, X_train, constants.MODEL_FILE_NAME)

    print(f'Best Model - Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')




if __name__ == '__main__':
    main()
