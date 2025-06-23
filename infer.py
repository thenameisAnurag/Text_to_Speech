import torch
import torchaudio
from pydub import AudioSegment
import os
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
import re

# Function to clean the text (remove HTML tags and newlines)
def clean_text(text):
    text = re.sub(r'<[^>]+>', '', text)  # Remove HTML tags
    text = text.replace('\n', ' ')  # Remove newlines
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    return text.strip()

# Function to split the text into chunks based on sentence boundaries and token limit (heuristic)
def split_text_at_sentence_boundaries(text, max_length=400):
    # Clean the text first (remove HTML and newlines)
    cleaned_text = clean_text(text)
    
    # Split the cleaned text into sentences using punctuation marks as delimiters
    sentences = re.split(r'(?<=\.)\s+', cleaned_text)  # Split at periods and maintain the sentence
    
    chunks = []
    current_chunk = ""
    
    # Iterate through the sentences and build chunks
    for sentence in sentences:
        # Check if the sentence fits within the max_length
        if len(current_chunk) + len(sentence) + 1 <= max_length:
            current_chunk += " " + sentence
        else:
            # If adding the sentence exceeds max_length, store the current chunk
            chunks.append(current_chunk.strip())
            current_chunk = sentence
    
    # Add the last chunk if there's any remaining text
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

# Function to generate and store audio chunks in an array
def generate_audio_chunks_with_token_limit(text, model, config, speaker_wav):
    # Split the text into chunks based on sentence boundaries and token length
    chunks = split_text_at_sentence_boundaries(text, max_length=400)
    
    audio_chunks = []  # This will store audio file paths
    
    for i, chunk in enumerate(chunks):
        # Synthesize speech for each chunk
        outputs = model.synthesize(
            chunk,
            config,
            speaker_wav=speaker_wav,
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

# Example usage
if __name__ == "__main__":
    # The input text to be converted to speech
    text = """ your input text """
    # Paths for output files
    output_wav = "Wave file path"
    output_mp3 = "Mp3 path"

    # Initialize the model and config
    config = XttsConfig()
    config.load_json("Json Path")
    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_dir="chcekpointpath", eval=True)
    model.cuda()  # Move the model to GPU

    # Generate audio for the provided text
    audio_chunks = generate_audio_chunks_with_token_limit(text, model, config, "sample audio path")

    # Merge the audio chunks into a single WAV file
    merge_audio_chunks(audio_chunks, output_wav)

    # Convert the merged WAV file to MP3
    convert_wav_to_mp3(output_wav, output_mp3)
    torch.cuda.empty_cache()
    print('Saved and the Cache is cleared ')
