

# 1. ** Text-to-Speech Conversion**

This section covers the implementation of a simple **Flask API** for converting text to speech using the neural network-based **TTS** model.

### **Model Used:**

This repository uses the **XTTS model** developed by Coqui. You can refer to the [Coqui XTTS Documentation](https://docs.coqui.ai/en/latest/models/xtts.html#) for a more detailed understanding.

### **Model Features:**

* **Language Support:** English, Spanish, French, German, Italian, Portuguese, Polish, Russian, Arabic, Japanese, Chinese, etc.
* **Cross-Language Voice Cloning:** Generate speech in various languages based on a single voice.
* **Emotion and Style Transfer:** Modify the emotion and style of the generated voice.

### Working of the model 
![image](https://github.com/user-attachments/assets/ac0df929-e14c-4da1-a033-745fb2a24619)

## **Setup**

### 1. **Install Dependencies:**

* Clone the repository and navigate to the project directory.

  ```bash
  git clone https://github.com/thenameisAnurag/Text_to_Speech.git
  cd Text_to_Speech
  ```

* Create a Python virtual environment:

  ```bash
  python3 -m venv venv
  ```

* Activate the virtual environment:

  ```bash
  source venv/bin/activate
  ```

* Install the required dependencies:

  ```bash
  pip install -r requirements.txt
  ```

### 2. **Download the TTS Model:**

Ensure that you have the **TTS model** and **configuration files** stored in the correct path:

```bash
/home/anuragmishra/Anurag/TTS/XTTS-v2/
```

### 3. **Run the Flask Application:**

Start the Flask API server:

```bash
python app.py
```

The application will run on **[http://0.0.0.0:5000/](http://0.0.0.0:5000/)**.

---

## **API Documentation**

### **POST /generate\_speech**

This endpoint accepts a **POST** request with a **JSON payload** containing the following fields:

#### Request Body Example:

```json
{
  "text": "Your long paragraph of text goes here.",
  "ascent": "en"
}
```

following the markdown chunking process described above. A diagram and explanation of the process is described below.

#### Response:

* The response will include a generated **audio file** in **MP3** and **Wav** format, containing the spoken version of the provided text.


---


# 2. **Inference of the Model Only**

The inference section highlights how to utilize the **XTTS model** for **text-to-speech conversion** directly, without any additional server setup.



### **Code Example:**

```python
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

# Load model and config
config = XttsConfig()
config.load_json("/path/to/xtts/config.json")
model = Xtts.init_from_config(config)
model.load_checkpoint(config, checkpoint_dir="/path/to/xtts/", eval=True)
model.cuda()

# Synthesize speech
outputs = model.synthesize(
    "This is an example text for generating speech using XTTS.",
    config,
    speaker_wav="/path/to/speaker_audio.wav",  # Reference speaker audio
    gpt_cond_len=3,
    language="en"
)

# Save the output audio
with open("generated_speech.wav", "wb") as f:
    f.write(outputs["wav"])
```

---



---


