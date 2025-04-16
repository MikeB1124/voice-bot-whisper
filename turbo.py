import pyaudio
import wave
import threading
import keyboard
import requests
import io
import tempfile
from turbo import WhisperTurbo

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000

recording = False
frames = []

def record_audio():
    global recording, frames
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("Ready. Hold SPACE to record.")

    while True:
        if recording:
            data = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(data)

def on_space_down(e):
    global recording, frames
    if not recording:
        print("Recording...")
        frames = []
        recording = True

def on_space_up(e):
    global recording, frames
    if recording:
        print("Stopped recording. Sending to server...")
        recording = False
        send_audio(frames)

def send_audio(frames):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
        buffer = io.BytesIO()
        wf = wave.open(buffer, 'wb')
        # wf = wave.open(tmpfile, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(pyaudio.PyAudio().get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()
        buffer.seek(0)
    #     tmpfile_path = tmpfile.name
    # turboWhisper = WhisperTurbo("openai/whisper-large-v3-turbo")
    # transcription = turboWhisper.transcribe(tmpfile_path)
    # print("Transcription:", transcription)

    # turboWhisper = WhisperTurbo("openai/whisper-large-v3-turbo")

    response = requests.post("http://64.247.196.75:5000/transcribe", files={'audio': ('audio.wav', buffer, 'audio/wav')})
    print("Transcription:", response.text)

keyboard.on_press_key("space", on_space_down)
keyboard.on_release_key("space", on_space_up)

record_thread = threading.Thread(target=record_audio)
record_thread.start()



import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

class WhisperTurbo():
    def __init__(self, model_id: str):
        self.device = "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, torch_dtype=self.torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
        )
        self.model.to(self.device)

        self.processor = AutoProcessor.from_pretrained(model_id)

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            torch_dtype=self.torch_dtype,
            device=self.device,
        )

    def transcribe(self, audio_file: str):
        result = self.pipe(audio_file)
        return result["text"]