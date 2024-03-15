import os

import emlearn
import librosa
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def extract_features(file_path):
    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccs_processed = np.mean(mfccs.T, axis=0)
    except Exception as e:
        print(f"Error encountered while parsing {file_path}: {e}")
        return None
    return mfccs_processed


def load_dataset(base_path):
    features = []
    labels = []

    for label, sub_dir in enumerate(['meow', 'other']):
        folder_path = os.path.join(base_path, sub_dir)
        for file_name in os.listdir(folder_path):
            if file_name.lower().endswith('.wav'):
                file_path = os.path.join(folder_path, file_name)
                mfccs = extract_features(file_path)
                if mfccs is not None:
                    features.append(mfccs)
                    labels.append(label)

    return np.array(features), np.array(labels)


# Load dataset
base_path = '/tmp/cat-doorbell-model-test'
X, y = load_dataset(base_path)

# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Train a model
model = RandomForestClassifier(n_estimators=10, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Convert the model to C code
c_model = emlearn.convert(model, method='inline')

code = c_model.save(file='/tmp/meow.h', name='meow')

# print(code)
