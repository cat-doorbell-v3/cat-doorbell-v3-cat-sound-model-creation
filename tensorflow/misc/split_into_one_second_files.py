import os

from pydub import AudioSegment


def split_wav_files(directory_path):
    # Ensure the provided path exists and is a directory
    if not os.path.isdir(directory_path):
        raise ValueError("The provided path is not a directory or does not exist.")

    # Iterate through all files in the directory
    for filename in os.listdir(directory_path):
        if filename.endswith('.wav'):
            full_path = os.path.join(directory_path, filename)

            # Load the .wav file
            audio = AudioSegment.from_wav(full_path)

            # Calculate the number of full 1-second chunks
            num_chunks = len(audio) // 1000

            # Split and save 1-second chunks
            for i in range(num_chunks):
                chunk = audio[i * 1000:(i + 1) * 1000]
                chunk_name = f"{filename[:-4]}-{i}.wav"
                chunk_path = os.path.join(directory_path, chunk_name)
                chunk.export(chunk_path, format="wav")

            # Delete the original file
            os.remove(full_path)


# Example usage:
# Replace '/path/to/your/directory' with the actual path to the directory containing your .wav files
directory_path = '/tmp/youtube_downloads'
split_wav_files(directory_path)
