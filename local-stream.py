import pyaudio
import wave
import threading
import keyboard
from websockets.sync.client import connect

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000

recording = False
frames = []
ws_url = "ws://localhost:8000"
ws = None
ws_lock = threading.Lock()

def record_audio():
    global recording, frames, ws
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
            with ws_lock:
                if ws:
                    try:
                        ws.send(data)
                    except Exception as e:
                        print(f"Error sending data: {e}")

def on_space_down(e):
    global recording, frames, ws
    if not recording:
        print("Recording...")
        frames = []
        recording = True
        try:
            # Establish connection BEFORE recording starts
            ws_connection = connect(ws_url)
            with ws_lock:
                ws = ws_connection
            print("Connected to server.")
        except Exception as e:
            print(f"Failed to connect to server: {e}")
            recording = False

def on_space_up(e):
    global recording, frames, ws
    if recording:
        print("Stopped recording. Sending to server...")
        recording = False
        # save_audio(frames)
        with ws_lock:
            if ws:
                ws.close()
                ws = None

def save_audio(frames):
    filename = "recording.wav"
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(pyaudio.PyAudio().get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
    print(f"Audio saved to {filename}")

# Setup
keyboard.on_press_key("space", on_space_down)
keyboard.on_release_key("space", on_space_up)

record_thread = threading.Thread(target=record_audio)
record_thread.start()
