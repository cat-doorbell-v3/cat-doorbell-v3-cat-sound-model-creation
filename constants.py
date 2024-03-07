#
MODEL_FILE_NAME = "/tmp/cat_doorbell_model.tflite"
#
MODEL_DATASET = "cat-doorbell-model-input"
MODEL_DATASET_ZIP = f"{MODEL_DATASET}.zip"
MODEL_DATASET_PATH = f"/tmp/{MODEL_DATASET}"
#
TEST_DATASET = "cat-doorbell-model-test"
TEST_DATASET_ZIP = f"{TEST_DATASET}.zip"
TEST_DATASET_PATH = f"/tmp/{TEST_DATASET}"
#
# This is common to both build and test
DATASET_CATEGORIES = [
    "cat",
    "not_cat"
]
DATASET_LABEL_MAP = {
    "cat": 0,
    "not_cat": 1
}
#
#
SAMPLING_RATE = 16000
AUDIO_DURATION = 1  # second
#
"""
N_MELS: This specifies the number of Mel bands to generate. The Mel scale is a perceptual scale of pitches judged by 
listeners to be equal in distance from one another. Choosing 40 Mel bands is a common practice in speech and audio 
processing as it provides a good balance between resolution and computational efficiency for most tasks, including 
voice recognition and sound classification.
"""
N_MELS = 40

"""
N_FFT: This is the length of the FFT (Fast Fourier Transform) window. It specifies the number of bins used for 
dividing the window into equal strips, or bins. A higher number increases the frequency resolution of the 
transformation at the cost of temporal resolution and increased computational complexity. 512 is a standard choice 
for many audio processing tasks, offering a good balance between frequency resolution and time.
"""
N_FFT = 512

"""
HOP_LENGTH: This is the number of samples between successive frames. A smaller hop length results in higher overlap 
between the frames and finer time resolution in the spectrogram. 320 is often chosen to ensure that there is enough 
overlap for smooth transitions between frames while keeping a reasonable computational load.
"""
HOP_LENGTH = 320

"""
MAX_PAD_LENGTH: In the context of generating spectrograms from audio clips, this value is likely used to define 
the maximum length of the time axis of the spectrogram matrix. It ensures uniform input size for neural network 
models. The specific value of 50 depends on the expected maximum duration of the audio clips when converted to 
spectrograms and the temporal resolution needed for the model to perform effectively.
"""
MAX_PAD_LENGTH = 50

"""
FEATURE_SIZE: Why 13? The choice of 13 as the feature size is a conventional practice that 
dates back to early speech recognition research. It has been found empirically 
that the first 12 MFCCs (along with the energy, making it 13) capture most of 
the relevant information about the spectral envelope of the audio signal for 
many tasks, including speech recognition and music information retrieval. 
Additional coefficients can provide diminishing returns or even introduce 
noise for certain applications.
"""
FEATURE_SIZE = 13

#
# Machine learning parameters
#
MODEL_CONV2D_FILTERS = 4

MODEL_CONV2D_KERNEL_SIZE = (3, 3)

MODEL_CONV2D_ACTIVATION = 'relu'

MODEL_CONV2D_KERNEL_REGULARIZER = 0.001

MODEL_MAX_POOL_SIZE = (2, 2)

MODEL_DROPOUT_RATE_ONE = 0.2

MODEL_DENSE_UNITS = 10

MODEL_DENSE_ONE_ACTIVATION = 'relu'

MODEL_DENSE_KERNEL_REGULARIZER = 0.001

MODEL_DROPOUT_RATE_TWO = 0.5

MODEL_DENSE_TWO_ACTIVATION = 'softmax'

#
# Training parameters
#
KFOLD_SPLITS = 5

TRAIN_TEST_SPLIT_SIZE = 0.2

PATIENCE = 3

BATCH_SIZE = 32

EPOCHS = 10

#
# Google AudioSet extract sample parameters
#
TFRECORD_FILES_PATTERN = '/tmp/audioset/*/*.tfrecord'
CAT_OUTPUT_DIR = '/tmp/cat-doorbell-model-input/cat'
NOT_CAT_OUTPUT_DIR = '/tmp/cat-doorbell-model-input/not_cat'

"""
Indices per this file: http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv
"""
CAT_CAT_SOUND_INDEX = 81
CAT_PURR_SOUND_INDEX = 82
CAT_MEOW_SOUND_INDEX = 83
CAT_HISS_SOUND_INDEX = 84
CAT_CATERWAUL_SOUND_INDEX = 85

CAT_SOUND_INDICES = {
    CAT_CAT_SOUND_INDEX,
    CAT_PURR_SOUND_INDEX,
    CAT_MEOW_SOUND_INDEX,
    CAT_HISS_SOUND_INDEX,
    CAT_CATERWAUL_SOUND_INDEX
}
