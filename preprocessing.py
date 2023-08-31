import os
import googleapiclient.discovery
import json
import requests
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import spectrogram
from bs4 import BeautifulSoup
from PIL import Image
import traceback
import librosa
from sentence_transformers import SentenceTransformer
sentence_transformer_model = SentenceTransformer('all-MiniLM-L6-v2')
from sklearn.metrics import pairwise_distances

def search_youtube_video(search_string, api_key):
    # Create a YouTube API client
    youtube = googleapiclient.discovery.build("youtube", "v3", developerKey=api_key)

    try:
        # Search for videos using the provided search string
        search_response = youtube.search().list(
            q=search_string,
            type="video",
            part="id",
            maxResults=1
        ).execute()

        # Extract the video ID of the first result
        if "items" in search_response and len(search_response["items"]) > 0:
            video_id = search_response["items"][0]["id"]["videoId"]
            video_url = f"https://www.youtube.com/watch?v={video_id}"
            return video_url
        else:
            return None

    except googleapiclient.errors.HttpError as e:
        print("An error occurred:", e)
        return None

google_api_key = ""
lyrics_api_user_id = 0
lyrics_api_token = ''

def download_song(name_and_artist:str, song_dir:str):
    if os.path.exists(f'{song_dir}/audio.wav'):
        print('Skipping song download for', name_and_artist.strip(), 'exists')
        return
    song_url = search_youtube_video(name_and_artist.strip() + " audio", google_api_key)


    os.system(f'yt-dlp "{song_url}" -o "{song_dir}/audio.%(ext)s" -f ba --extract-audio --audio-format wav --postprocessor-args "-ac 1"')

def download_lyrics(name_and_artist:str, lyrics_api_user_id:str, lyrics_api_token:str, song_dir:str):
    lyrics_path = f'{song_dir}/lyrics.txt'
    if os.path.exists(lyrics_path):
        print('Skipping lyrics for', name_and_artist.strip(), 'exists')
        return
    name, artist = name_and_artist.replace('"', '').split(' - ')
    
    # Removing featured artists (ft.) for this search
    artist = artist.split('ft.')[0]

    path = f'https://www.stands4.com/services/v2/lyrics.php?uid={lyrics_api_user_id}&tokenid={lyrics_api_token}&term={name}&artist={artist}&format=json'
    print(path)
    link_to_lyrics = requests.get(path).json()['result'][0]['song-link']
    soup = BeautifulSoup(requests.get(link_to_lyrics).text)
    lyrics = soup.find(id='lyric-body-text').text
    open(lyrics_path, 'w').write(lyrics)

def generate_stft(song_dir:str):
    # Load the WAV file
    # sample_rate, audio_data = wavfile.read(f'{song_dir}/audio.wav')
    N_FFT = 4096
    HOP_LENGTH = int(N_FFT / 2)

    audio_data, sample_rate = librosa.load(f'{song_dir}/audio.wav', offset=60, duration=60)
    mel_spect = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate, n_fft=N_FFT, hop_length=HOP_LENGTH)
    mel_spect = librosa.power_to_db(mel_spect, ref=np.max)
    librosa.display.specshow(mel_spect, y_axis='mel', fmax=8000, x_axis='time', hop_length=HOP_LENGTH)
    plt.axis('off')
    plt.savefig(f"{song_dir}/spectrogram.png", bbox_inches='tight', pad_inches=0)

    # STFT parameters
    # window_size = 2048
    # hop_size = 512

    # Compute the STFT
    # frequencies, times, spectrogram_data = spectrogram(audio_data, fs=sample_rate, 
                                                        # nperseg=4096)


    # Replacing zeros with very small values
    # spectrogram_data[spectrogram_data == 0] = 1e-20

    # Normalize the spectrogram data to the range [0, 255]
    # spectrogram_data_normalized = (10 * np.log10(spectrogram_data) - np.min(10 * np.log10(spectrogram_data)))
    # spectrogram_data_normalized = (spectrogram_data_normalized / np.max(spectrogram_data_normalized)) * 255
    # spectrogram_data_normalized = spectrogram_data_normalized.astype(np.uint8)
    # spectrogram_image = Image.fromarray(spectrogram_data_normalized, 'L')
    # spectrogram_image.save(f"{song_dir}/spectrogram.png")

    # Prepare the STFT histogram image
    # plt.figure(figsize=(10, 6))
    # plt.imshow(10 * np.log10(spectrogram_data), 
    #            origin='lower', aspect='auto', cmap='inferno')
    # plt.axis('off')
    # plt.savefig(f"{song_dir}/spectrogram.png", bbox_inches='tight', pad_inches=0)

    # Load the image and resize it
    image = Image.open(f"{song_dir}/spectrogram.png")

    # Resize the image to have a height of 224 pixels while maintaining aspect ratio
    new_height = 224
    aspect_ratio = image.width / image.height
    new_width = int(aspect_ratio * new_height)
    resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # Calculate the amount to crop from each side
    crop_amount = (new_width - new_height) // 2

    # Crop the image to have a width of 224 pixels
    cropped_image = resized_image.crop((crop_amount, 0, new_width - crop_amount, new_height))

    # Save the cropped image
    cropped_image.save(f"{song_dir}/spectrogram_224px.png")

def extract_chorus(song_dir:str):
    song_text = open(os.path.join(song_dir, 'lyrics.txt')).read()
    song_parts = song_text.split('\n\n')
    part_embeddings = sentence_transformer_model.encode(song_parts)
    chorus_candidate = pairwise_distances(part_embeddings).mean(axis=1).argmin()
    open(os.path.join(song_dir, 'chorus.txt'), 'w').write(''.join(song_parts[chorus_candidate]))

def preprocess_all(song_filename:str):
    songs = open(song_filename).readlines()
    songs = list(set(songs))
    index2song = {i:songs[i] for i in range(len(songs))}
    preprocessed_songs = []
    for song_index, song in index2song.items():
        try:
            song_dir = f'data/songs/all/{song_index}'
            os.makedirs(song_dir, exist_ok=True)
            
            download_song(song, song_dir)
            download_lyrics(song, lyrics_api_user_id, 
                            lyrics_api_token, song_dir)
            extract_chorus(song_dir)
            generate_stft(song_dir)
            preprocessed_songs.append(song_index)
        except:
            print(song, 'failed')
            traceback.print_exc()
    #    if os.path.exists(filename):
    index2song = {i:songs[i] for i in preprocessed_songs}
    json.dump(index2song, open('data/index2song.json', 'w'))
    return preprocessed_songs

if __name__ == "__main__":
    preprocess_all('songs.txt')
