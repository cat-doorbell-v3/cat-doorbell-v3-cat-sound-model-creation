import os

import pytube.exceptions
from moviepy.editor import *
from pytube import Search
from pytube import YouTube


def search_and_download(query, max_downloads=1, output_path='downloads'):
    """
    Search YouTube for videos matching the query and download the audio of up to max_downloads results that are less
    than 1 hour long, not age-restricted, and saves the file as the YouTube URL short name.

    Parameters:
    - query: Search query (string)
    - max_downloads: Maximum number of audio files to download (integer)
    - output_path: Directory to save the downloaded audio (string)
    """
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    downloads = 0  # Count the number of downloads
    s = Search(query)
    for video in s.results:
        if downloads >= max_downloads:
            break  # Stop if we've reached the maximum number of downloads

        try:
            # Use YouTube object to get detailed info like duration and age restriction
            yt = YouTube(video.watch_url)

            # Check for "cat" and "meow" in the video title, duration less than 1 hour, and not age-restricted
            if all(keyword in yt.title.lower() for keyword in
                   ['cat', 'meow']) and yt.length < 3600 and not yt.age_restricted:
                print(f"Downloading audio from: {yt.title}")
                # Download the video
                video_stream = yt.streams.filter(only_audio=True).first()
                download_path = video_stream.download(output_path=output_path)
                # Extract audio and save as .wav using the video's "short name" (video ID)
                video_id = yt.video_id
                wav_filename = os.path.join(output_path, f"{video_id}.wav")
                audio_clip = AudioFileClip(download_path)
                audio_clip.write_audiofile(wav_filename)
                # Remove the original download (if not a .wav file)
                if download_path != wav_filename:
                    os.remove(download_path)
                print(f"Audio saved as: {wav_filename}")
                downloads += 1
        except pytube.exceptions.AgeRestrictedError:
            print(f"Skipping age-restricted video: {video.watch_url}")

    if downloads == 0:
        print("No suitable videos found.")


search_and_download("cat meow", max_downloads=30, output_path='/tmp/youtube_downloads')
