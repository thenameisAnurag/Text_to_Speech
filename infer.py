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
    text = """While retrieval augmented generation (RAG) has been overshadowed by large language model (LLM) agents lately, RAG is still a great option when advanced search functionality is needed. One of my latest project has been to set up a customized production RAG system for class documents.

The goal of this generative AI based project is to help college students find class information more easily to improve student experience and outcomes. This is a good use case for RAG, since enhanced search capabilities are needed.

As we are nearing the end of the school semester, the project has been used and deployed for around 3 months. While the project is still in its early state, we are open sourcing the project for anyone to use and sharing our initial experiences and thoughts with the community.

You can find the app code with the helm chart on GitHub

In the time that app has been deployed, it has answered over 250+ questions, and helped students in over 60 sessions (We do not track any sort of PII, so we use a randomized session id to protect student information). More information and data related to the study will be available when follow up research work has been completed. For this post I’ll focus on the development, deployment, and architecture of the generative AI app.

App Architecture
From a high level the architecture is similar to a typical python application. Later on we’ll drill down into the RAG workflow. As I worked on this app in my free time, I didn’t have time to add all of the bells and whistles that I was hoping for.

We chose to deploy the app on Google cloud and Kubernetes due to robustness being a priority. I knew I might not have a lot of time to troubleshoot any issues that could come up during production. I had originally mocked up the POC of the app in Streamlit, but chose to create the first version of the app with HTMX and FastAPI for better flexibility and performance.

The architecture diagram below goes into more detail about the high level components we used.

Components:

GCP — The production app is deployed on GCP. We had grant credits on GCP for the research, but I also like the kubernetes experience on GCP the best out of the three main cloud providers
Kubernetes — Kubernetes is fantastic to develop with. It is easy to switch from a local environment to a cloud environment for testing, and it supports every component you need to deploy an app to production.
Load Balancer — We used a GCP load balancer to manager the traffic from the public internet
Nginx Ingress — Nginx has great support with kubernetes and is really well documented.
Python Backend — In the backend we used the framework, llama-index, for RAG, FastAPI for serving, and HTMX for the UI.
Postgres DB for state — We have a postgres database to manage state, and save relevant data to the research. We also a nightly job that backs up the database.
Optional Local LLM hosting — Optional Ollama support was added to allow for running the stack locally without the need for an OpenAI API key.
A couple high level considerations factored into our choice of Kubernetes and GCP for the cloud platform and OpenAI for the LLM.

Robustness

I may not have time to fix any issues right away, so it is important that the deployment was robust to failures. Deploying on GCP and Kuberentes helped ensure that the app was highly available, and could restart automatically if there were any failures.

I’m pleased to say there was only one issue I had to fix related session state, and I was able to just push the changes to my CI/CD pipeline which were propagated to the gcp k8s cluster automatically. Other than that one issue, I didn’t have to intervene in the course of the 3 months.

Cost

The most expensive part of the app is the cloud hosting. Serverless options like AWS Lightsail are cheaper, but for me, Kubernetes makes it easiest to add additional services and test them all together locally before pushing changes to the cloud cluster.

LLM costs were actually very cheap. For the production LLM, we use OpenAI’s chatgpt 3.5, due to its good price to performance ratio. Based on our testing, hosting our own llm would be much more expensive and slower for the small volumes requests we receive. However, I did add an option to use Ollama for testing LLMs locally, so no API key is required for testing it out.

RAG Workflows
When we first started the project there were less solutions that allowed you to easily query your own documents. Now we’ve seen a few services pop up for RAG over uploaded documents. These services are good for generic use cases, but obviously having our own solution allows us to have full control over all aspects of the app.

I ended up creating a custom markdown document splitter (extending Llama Index) to fit our needs. We decided a granularity of “h2” tags would be a good level to split on, where everything under h2 tags would be grouped up to the h2 chunk. You would need to have a good understanding of your documents to know if this would be applicable to your documents. To avoid the hassle of PDF parsing all documents were uploaded in Markdown format.

We also decided to only return the document references and headings of the returned document chunks, instead of the whole reference text, to try and encourage students to cross reference the source material.

Other than the above we aren’t doing anything too fancy when it comes to RAG. Likely if you have used RAG before you’ll be familiar with the following workflow that we used.

On startup, the embeddings were initialized following the markdown chunking process described above. A diagram and explanation of the process is described below.

"""
    # Paths for output files
    output_wav = "/home/anuragmishra/Anurag/TTS/XTTS-v2/final_output.wav"
    output_mp3 = "/home/anuragmishra/Anurag/TTS/XTTS-v2/final_output.mp3"

    # Initialize the model and config
    config = XttsConfig()
    config.load_json("/home/anuragmishra/Anurag/TTS/XTTS-v2/config.json")
    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_dir="/home/anuragmishra/Anurag/TTS/XTTS-v2/", eval=True)
    model.cuda()  # Move the model to GPU

    # Generate audio for the provided text
    audio_chunks = generate_audio_chunks_with_token_limit(text, model, config, "/home/anuragmishra/Anurag/TTS/XTTS-v2/samples/es_sample.wav")

    # Merge the audio chunks into a single WAV file
    merge_audio_chunks(audio_chunks, output_wav)

    # Convert the merged WAV file to MP3
    convert_wav_to_mp3(output_wav, output_mp3)
    torch.cuda.empty_cache()
    print('Saved and the Cache is cleared ')
