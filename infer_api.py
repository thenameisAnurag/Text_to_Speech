from flask import Flask, request, jsonify, send_file
import torch
import torchaudio
from pydub import AudioSegment
import os
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
import re

app = Flask(__name__)

# Function to clean the text (remove HTML tags and newlines)
def clean_text(text):
    text = re.sub(r'<[^>]+>', '', text)  # Remove HTML tags
    text = text.replace('\n', ' ')  # Remove newlines
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    return text.strip()

# Function to split the text into chunks based on character length
def split_text_by_length(text, max_length=400):
    # Clean the text first (remove HTML and newlines)
    cleaned_text = clean_text(text)
    
    chunks = []
    while len(cleaned_text) > max_length:
        # Split the text into chunks based on the max_length
        chunk = cleaned_text[:max_length]
        chunks.append(chunk)
        cleaned_text = cleaned_text[max_length:]  # Remove the processed part
    
    # Append the remaining part if it's smaller than max_length
    if cleaned_text:
        chunks.append(cleaned_text)
    
    return chunks

# Function to generate and store audio chunks in an array
def generate_audio_chunks_with_token_limit(text, model, config, speaker_wav):
    # Split the text into chunks based on character length
    chunks = split_text_by_length(text, max_length=400)
    
    audio_chunks = []  # This will store audio file paths
    
    for i, chunk in enumerate(chunks):
        # Synthesize speech for each chunk
        outputs = model.synthesize(
            chunk,
            config=config,  # Pass the config here
            speaker_wav=speaker_wav,  # Pass the speaker wav file here
            gpt_cond_len=3,
            temperature=0.7,
            language="en"
        )
        
        audio_tensor = torch.tensor(outputs['wav'])  # Get the audio waveform (tensor)
        audio_tensor = audio_tensor.unsqueeze(0)  # Reshape to 2D tensor (required by torchaudio.save)

        # Save the chunk as a temporary WAV file
        temp_audio_file = f"temp_audio_chunk_{i}.wav"
        torchaudio.save(temp_audio_file, audio_tensor, 24000)  # Save with 24000 Hz sample rate
        audio_chunks.append(temp_audio_file)  # Add the file path to the list
    
    return audio_chunks

# Function to merge all audio chunks into a single WAV file
def merge_audio_chunks(audio_chunks, output_file):
    combined_audio = AudioSegment.empty()
    for chunk_file in audio_chunks:
        audio = AudioSegment.from_wav(chunk_file)  # Load each WAV chunk
        combined_audio += audio  # Append the audio to the combined audio
    
    # Export the combined audio to a single WAV file
    combined_audio.export(output_file, format="wav")
    print(f"Audio successfully merged and saved to {output_file}")

    # Clean up temporary files (delete them)
    for chunk_file in audio_chunks:
        os.remove(chunk_file)

# Function to convert the final WAV file to MP3 format
def convert_wav_to_mp3(wav_file, mp3_file):
    audio = AudioSegment.from_wav(wav_file)
    audio.export(mp3_file, format="mp3")
    print(f"Audio successfully converted to {mp3_file}")

# Load the TTS model based on the language selection
def load_model_for_language(language_code):
    # Initialize the model and config
    config = XttsConfig()
    config.load_json("XTTS config.json path")
    
    # Check for different language configurations and paths
    if language_code == 'en':  # English
        speaker_wav = "path_of_sample_file/en_sample.wav"
    elif language_code == 'fr':  # French
        speaker_wav = "path_of_sample_file/fr_sample.wav"
    elif language_code == 'de':  # German
        speaker_wav = "path_of_sample_file/de_sample.wav"
    elif language_code == 'ja':  # Japanese
        speaker_wav = "path_of_sample_file/ja-sample.wav"
    else:
        return None, None  # Unknown language code

    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_dir="XTTS v2 path/", eval=True)
    model.cuda()  # Move the model to GPU

    return model, speaker_wav, config

@app.route('/generate_speech', methods=['POST'])
def generate_speech():
    # Get the data from the request
    data = request.get_json()

    text = data.get("text", "")
    ascent = data.get("ascent", "en").lower()  # Default to English if no ascent is provided
    
    # Load the appropriate model based on the ascent (language)
    model, speaker_wav, config = load_model_for_language(ascent)
    
    if not model:
        return jsonify({"error": "Unsupported language code"}), 400

    # Generate audio for the provided text
    audio_chunks = generate_audio_chunks_with_token_limit(text, model, config, speaker_wav)  # Pass config and speaker_wav

    # Define output paths for the WAV and MP3 files
    output_wav = "path_of_sample_file_sample.wav/output_audio.wav"
    output_mp3 = "path_of_sample_file_sample.mp3/output_audio.mp3"
    
    # Merge the audio chunks into a single WAV file
    merge_audio_chunks(audio_chunks, output_wav)

    # Convert the merged WAV file to MP3
    convert_wav_to_mp3(output_wav, output_mp3)
    torch.cuda.empty_cache()

    # Return the MP3 file as a response
    return send_file(output_mp3, mimetype="audio/mpeg", download_name="generated_audio.mp3") 

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
   
