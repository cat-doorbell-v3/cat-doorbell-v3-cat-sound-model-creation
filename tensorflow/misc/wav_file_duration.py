import glob
import os
import wave

# Path to the directory containing .wav files
directory_path = "/Users/tennis/sound-library/vetted-meows"

# Find all .wav files in the directory
wav_files = glob.glob(os.path.join(directory_path, '*.wav'))

# Initialize counters
count_less_than_one_second = 0
count_equal_one_second = 0
count_greater_than_one_second = 0

for wav_file in wav_files:
    # Open the wave file
    with wave.open(wav_file, 'rb') as wf:
        # Calculate the duration of the audio file in seconds
        frames = wf.getnframes()
        rate = wf.getframerate()
        duration_seconds = frames / float(rate)
        duration_milliseconds = duration_seconds * 1000

        # Determine the visual scale and update counters
        if duration_seconds < 1:
            scale = "<"
            count_less_than_one_second += 1
        elif duration_seconds == 1:
            scale = "="
            count_equal_one_second += 1
        else:
            scale = ">"
            count_greater_than_one_second += 1

        # Print out the name, visual scale, and duration in milliseconds
        print(f"File: {os.path.basename(wav_file)}, Duration: {scale}, {duration_milliseconds:.2f} ms")

# Print the tally
print("\nTally:")
print(f"Files less than 1 second: {count_less_than_one_second}")
print(f"Files equal to 1 second: {count_equal_one_second}")
print(f"Files greater than 1 second: {count_greater_than_one_second}")

# Note: Update the 'directory_path' variable with the actual path to the directory containing your .wav files.
