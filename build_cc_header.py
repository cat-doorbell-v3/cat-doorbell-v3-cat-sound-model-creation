import constants
import utils


def main():
    utils.remove_directories([constants.MODEL_DATASET])

    utils.unzip_file(constants.MODEL_DATASET_ZIP, '/tmp')

    bit_depths = utils.get_all_bit_depth(constants.MODEL_DATASET_PATH)
    print(f"Found {bit_depths} bit depths")

    sample_rates = utils.get_directory_wav_sampling_rates(constants.MODEL_DATASET_PATH)
    print(f"Found {len(sample_rates)} sample rate")
    for s in sample_rates:
        print(f"Sample Rate: {s}")

    _, _, avg_duration = utils.get_audio_durations(constants.MODEL_DATASET_PATH)

    audio_cc_constants = utils.generate_cpp_definitions(constants.FEATURE_SIZE,
                                                        constants.SAMPLING_RATE,
                                                        avg_duration,
                                                        constants.N_FFT,
                                                        constants.HOP_LENGTH)

    utils.generate_header_file(constants.DATASET_LABEL_MAP, audio_cc_constants)


if __name__ == '__main__':
    main()
