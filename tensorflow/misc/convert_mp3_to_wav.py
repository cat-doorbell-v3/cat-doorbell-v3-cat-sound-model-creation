import os

from pydub import AudioSegment


def convert_mp3_to_wav(source_dir, target_dir):
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.endswith(".mp3"):
                mp3_path = os.path.join(root, file)
                wav_path = os.path.join(target_dir, os.path.splitext(file)[0] + '.wav')

                # Load MP3
                audio = AudioSegment.from_mp3(mp3_path)

                # Convert to mono, 16KHz sample rate, and 16-bit depth
                audio = audio.set_frame_rate(16000).set_sample_width(2).set_channels(1)

                # Saving the result
                audio.export(wav_path, format="wav")
                print(f"Converted {mp3_path} to {wav_path}")


# Example usage
source_directory = '/Users/tennis/sound-library/CAT_SOUND_DB_SAMPLES'
target_directory = '/Users/tennis/sound-library/CAT_SOUND_DB_SAMPLES_WAV'

convert_mp3_to_wav(source_directory, target_directory)
