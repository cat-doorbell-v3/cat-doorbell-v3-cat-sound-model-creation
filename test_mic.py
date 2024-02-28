import sounddevice as sd
from scipy.io.wavfile import write, read


def record_audio(duration, samplerate=44100, filename='output.wav'):
    """
    Records audio from the default microphone for the given duration and samplerate.
    The recording is saved to a WAV file.

    Args:
    - duration (float): Recording duration in seconds.
    - samplerate (int): Sampling rate in Hz.
    - filename (str): Filename to save the recorded audio.
    """
    print(f"Recording for {duration} seconds...")
    recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='float32')
    sd.wait()  # Wait until the recording is finished
    write(filename, samplerate, recording)  # Save as WAV file
    print(f"Recording saved to {filename}")


def play_audio(filename='output.wav'):
    """
    Plays an audio file.

    Args:
    - filename (str): Filename of the audio file to play.
    """
    samplerate, data = read(filename)
    print(f"Playing {filename}...")
    sd.play(data, samplerate)
    sd.wait()  # Wait until the audio is finished playing


if __name__ == '__main__':
    duration = 5  # seconds
    samplerate = 8000  # Hz
    filename = 'test_recording.wav'

    record_audio(duration, samplerate, filename)
    play_audio(filename)
