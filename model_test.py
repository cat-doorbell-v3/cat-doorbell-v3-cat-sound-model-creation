import numpy as np
import tensorflow as tf

# Load TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path="cat_sound_model.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test the model on random input data or actual test data
input_shape = input_details[0]['shape']
# Generate random FLOAT32 data in the range -1 to 1
input_data = np.random.uniform(-1.0, 1.0, size=input_shape).astype(np.float32)
interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()

# Extract the output and use it
output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data)
