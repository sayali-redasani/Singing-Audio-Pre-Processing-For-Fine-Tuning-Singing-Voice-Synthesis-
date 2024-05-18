from pytube import YouTube
import os
from pydub import AudioSegment
from spleeter.separator import Separator
import csv
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

def download_youtube_audio(link, output_path="temp_audio"):
    """
    Downloads the highest resolution video from YouTube, extracts audio, and saves it as a WAV file.
    
    Args:
    link (str): The YouTube video URL.
    output_path (str): The directory to save the downloaded video and extracted audio.
    
    Returns:
    str: The file path to the extracted WAV audio.
    """
    # Download the highest resolution video from YouTube
    yt = YouTube(link)
    video = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
    out_file = video.download(output_path=output_path)
    base, _ = os.path.splitext(out_file)

    # Convert the video to WAV audio using pydub
    audio = AudioSegment.from_file(out_file)
    wav_file = f"{base}.wav"
    audio.export(wav_file, format='wav')
    
    return wav_file

def split_audio(audio_folder, output_folder):
    """
    Splits audio files in a given folder into separate stems (vocals and accompaniment) using Spleeter.
    
    Args:
    audio_folder (str): The directory containing the input audio files.
    output_folder (str): The directory to save the separated stems.
    """
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Initialize Spleeter with 2 stems separation
    separator = Separator('spleeter:2stems')

    # Iterate over files in the input audio folder
    for filename in os.listdir(audio_folder):
        if filename.endswith('.mp3') or filename.endswith('.wav'):
            # Construct the full path of the audio file
            audio_file = os.path.join(audio_folder, filename)
            
            # Perform separation and save the output to the output folder
            separator.separate_to_file(audio_file, output_folder)
    
def transcription(folder_path):
    """
    Processes all vocal tracks in the specified folder for speech-to-text transcription.
    
    Args:
    folder_path (str): The directory containing the subfolders of songs.
    """
    # Check for cuda availability
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    # Load the speech-to-text model
    model_id = "openai/whisper-large-v3"
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, torch_dtype=torch_dtype, use_safetensors=True)
    model.to(device)
    processor = AutoProcessor.from_pretrained(model_id)

    # Define the pipeline for automatic speech recognition
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        chunk_length_s=30,
        batch_size=16,
        return_timestamps=True,
        torch_dtype=torch_dtype,
        device=device,
    )

    for root, _, files in os.walk(folder_path):
        for filename in files:
            if filename.endswith("vocals.wav"):
                song_path = os.path.join(root, filename)
                # Generate lyrics and timestamps for the song
                result = pipe(song_path, generate_kwargs={"language": "hindi"})
                csv_file_path = os.path.splitext(song_path)[0] + ".csv"
                
                # Write the results to a CSV file
                with open(csv_file_path, "w", encoding="utf-8", newline='') as csvfile:
                    csv_writer = csv.writer(csvfile)
                    csv_writer.writerow(['text', 'timestamp'])
                    for chunk in result['chunks']:
                        csv_writer.writerow([chunk['text'], chunk['timestamp']])
                print(f"Lyrics and timestamps saved to: {csv_file_path}")

def split_songs_by_seconds(input_file, output_folder, start_time, end_time, segment_count):
    """
    Splits a segment of the song from start_time to end_time and exports it as a WAV file.
    
    Args:
    input_file (str): The path to the input audio file.
    output_folder (str): The directory to save the output segments.
    start_time (float): The start time in seconds for the segment.
    end_time (float): The end time in seconds for the segment.
    segment_count (int): The segment number to name the output file.
    """
    # Convert start and end times to milliseconds
    start_time_ms = start_time * 1000
    end_time_ms = end_time * 1000

    # Load the audio file and extract the segment
    segment = AudioSegment.from_file(input_file)
    song_segment = segment[start_time_ms:end_time_ms]

    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Export the segment to a new file
    output_file = os.path.join(output_folder, f"segment_{segment_count}.wav")
    song_segment.export(output_file, format="wav")

def process_csv_and_split_songs(csv_file_path, input_file, output_folder):
    """
    Reads a CSV file with start and end times and splits the input audio file accordingly.
    
    Args:
    csv_file_path (str): The path to the CSV file containing start and end times.
    input_file (str): The path to the input audio file.
    output_folder (str): The directory to save the output segments.
    """
    with open(csv_file_path, 'r', encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file)
        header = next(csv_reader)  # Skip the header row

        for count, row in enumerate(csv_reader):
            start_time = float(row[2])
            end_time = float(row[3])
            split_songs_by_seconds(input_file, output_folder, start_time, end_time, count)

def generate_mel_spectrograms(input_folder, output_folder):
    """
    Generates and saves Mel spectrograms for all .wav files in the specified input folder.
    
    Args:
    input_folder (str): The directory containing the input .wav audio files.
    output_folder (str): The directory to save the generated Mel spectrogram images.
    """
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file_name in os.listdir(input_folder):
        if file_name.endswith(".wav"):
            # Load the audio file
            audio_path = os.path.join(input_folder, file_name)
            y, sr = librosa.load(audio_path)

            # Generate the Mel spectrogram
            S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)

            # Convert to decibel scale
            S_db = librosa.power_to_db(S, ref=np.max)

            # Display and save the Mel spectrogram
            plt.figure(figsize=(10, 4))
            librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='mel')
            plt.colorbar(format='%+2.0f dB')
            plt.title('Mel spectrogram')

            # Construct the output file path and save the plot
            output_path = os.path.join(output_folder, os.path.splitext(file_name)[0] + "_mel_spectrogram.png")
            plt.savefig(output_path)
            plt.close() 
    
def main(youtube_link, temp_audio_folder, spleeter_output_folder, final_output_folder, mel_spectrogram_folder):
    wav_file = download_youtube_audio(youtube_link, temp_audio_folder)
    split_audio(temp_audio_folder, spleeter_output_folder)
    transcription(spleeter_output_folder)
    
    # Assuming only one CSV file will be generated per video download
    csv_files = [f for f in os.listdir(spleeter_output_folder) if f.endswith(".csv")]
    for csv_file in csv_files:
        csv_file_path = os.path.join(spleeter_output_folder, csv_file)
        process_csv_and_split_songs(csv_file_path, wav_file, final_output_folder)
    
    generate_mel_spectrograms(final_output_folder, mel_spectrogram_folder)

if __name__ == "__main__":
    youtube_link = "https://www.youtube.com/watch?v=DRBxROqWOPk"
    temp_audio_folder = "temp_audio"
    spleeter_output_folder = "spleeter_output"
    final_output_folder = "final_output"
    mel_spectrogram_folder = "mel_spectrograms"
    
    main(youtube_link, temp_audio_folder, spleeter_output_folder, final_output_folder, mel_spectrogram_folder)