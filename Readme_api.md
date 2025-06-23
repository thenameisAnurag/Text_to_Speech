# TTS API for Text-to-Speech Conversion

This is a simple Flask API for converting text to speech using a neural network-based TTS model.

## Setup

1. **Install Dependencies**:
    - Clone the repository and navigate to the project directory.
    - Create a Python virtual environment:
      ```bash
      python3 -m venv venv
      ```
    - Activate the virtual environment:
      ```bash
      source venv/bin/activate
      ```
    - Install the required dependencies:
      ```bash
      pip install -r requirements.txt
      ```

2. **Download the TTS Model**:
    - Ensure that you have the TTS model and configuration files in the correct path:
      `/home/anuragmishra/Anurag/TTS/XTTS-v2/`

3. **Run the Flask Application**:
    - Start the Flask API server:
      ```bash
      python app.py
      ```
    - The application will run on `http://0.0.0.0:5000/`.

## API Documentation

### **POST /generate_speech**

This endpoint accepts a POST request with a JSON payload containing the following fields:

#### Request Body:
```json
{
  "text": "Your long paragraph of text goes here.",
  "ascent": "en"
}
