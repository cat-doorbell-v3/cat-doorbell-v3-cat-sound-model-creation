import os

from pydub import AudioSegment


def process_audio_files(directory_path):
    for filename in os.listdir(directory_path):
        if filename.endswith('.wav'):
            file_path = os.path.join(directory_path, filename)
            audio = AudioSegment.from_wav(file_path)

            if len(audio) > 1000:
                # Abbreviate to 1000ms and save
                abbreviated_audio = audio[:1000]
                abbreviated_audio.export(file_path, format="wav")
                print(f"Abbreviated {filename} to 1000ms.")

            elif len(audio) < 1000:
                # Remove file
                os.remove(file_path)
                print(f"Removed {filename} as it is shorter than 1 second.")


# Replace 'your_directory_path_here' with the path to your directory containing the .wav files
directory_path = '/Users/tennis/sound-library/more-vetted-meows'
process_audio_files(directory_path)
