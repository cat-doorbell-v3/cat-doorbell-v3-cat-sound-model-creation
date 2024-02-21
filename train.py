##
# 1. Load the data
#
import pickle

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical


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


# Load your serialized MFCC features
with open('mfcc_features.pkl', 'rb') as f:
    data = pickle.load(f)

features = []
labels = []
label_map = {'Angry': 0, 'Defense': 1, 'Fighting': 2, 'Happy': 3, 'HuntingMind': 4,
             'Mating': 5, 'MotherCall': 6, 'Paining': 7, 'Resting': 8, 'Warning': 9}
for file_path, mfcc_features in data.items():
    features.append(mfcc_features)
    label = file_path.split('/')[2]  # Assuming the label is in the third segment of the path
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


# Define a generator function that provides the representative dataset for calibration
def representative_dataset_gen():
    for input_value in X_train_cnn[:100]:  # Assuming you're using a subset of your training data
        # Model has only one input so each data point has to be in a tuple
        yield [np.expand_dims(input_value, axis=0).astype(np.float32)]


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
