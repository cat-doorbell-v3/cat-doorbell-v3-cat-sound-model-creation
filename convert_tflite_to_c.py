import os


def convert_tflite_to_c(source_file, header_file, source_file_path, array_name):
    """
    Converts a TFLite model file into a C source and header file.

    Parameters:
    - source_file: Path to the input .tflite file.
    - header_file: Path to the output .h file for declaration.
    - source_file_path: Path to the output .c file for data definition.
    - array_name: Name of the array.
    """
    with open(source_file, 'rb') as file:
        data = file.read()

    # Convert to hex
    hex_data = ["0x{:02x}".format(byte) for byte in data]

    # Generate C source file with data array
    # Generate C source file with data array
    with open(source_file_path, 'w') as file:
        file.write(f"#include \"{os.path.basename(header_file)}\"\n\n")
        file.write(f"const unsigned char {array_name}[] = " + "{\n")
        line = ",\n".join([", ".join(hex_data[i:i + 12]) for i in range(0, len(hex_data), 12)])
        file.write(line)
        file.write("\n};\n")
        # Use sizeof to determine the array length
        file.write(f"const unsigned int {array_name}_len = sizeof({array_name});")

    # Generate header file with array declaration
    with open(header_file, 'w') as file:
        file.write(f"#ifndef {array_name.upper()}_H\n")
        file.write(f"#define {array_name.upper()}_H\n\n")
        file.write(f"extern const unsigned char {array_name}[];\n")
        file.write(f"extern const unsigned int {array_name}_len;\n\n")
        file.write("#endif")


if __name__ == "__main__":
    # Path to your TFLite model file
    source_tflite_file = './cat_sound_model_quant.tflite'
    # Header file for declaration
    header_file = 'cat_sound_model.h'
    # Source file for data definition
    source_file_path = 'cat_sound_model.c'
    # Array name
    array_name = 'cat_sound_model'

    convert_tflite_to_c(source_tflite_file, header_file, source_file_path, array_name)
