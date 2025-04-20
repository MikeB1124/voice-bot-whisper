import asyncio
import datetime
import numpy as np
import torch
from collections import deque
from websockets.asyncio.server import serve
import wave
from turbo import WhisperTurbo
from llama import Llama3
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


llamaClient = None
turboWhisper = None
# Constants
SAMPLE_RATE = 16000
WINDOW_SIZE = 512  # 32ms for 16kHz
HOP_SIZE = 256     # 50% overlap
SILENCE_THRESHOLD_SECONDS = 1.0
SPEECH_PROB_THRESHOLD = 0.5
PROB_SMOOTHING_WINDOW = 5

# Load Silero VAD model
model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False)

# Audio helper
def save_audio(frames):
    global turboWhisper
    filename = f"output_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit audio
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(b''.join(frames))
    print(f"💾 Audio saved to {filename}")
    transcript = turboWhisper.transcribe(filename)
    print(f"USER TRANSCRIPT: {transcript}")
    llama_response = llamaClient.invoke_bedrock(transcript)
    print(f"AGENT RESPONSE: {llama_response}")



async def echo(websocket):
    buffer = b""
    frames = []
    silence_duration = 0.0
    prob_buffer = deque(maxlen=PROB_SMOOTHING_WINDOW)
    speech_detected_in_session = False

    async for message in websocket:
        buffer += message
        frames.append(message)

        # Process audio in overlapping windows
        while len(buffer) >= (WINDOW_SIZE * 2):
            frame_bytes = buffer[:WINDOW_SIZE * 2]
            buffer = buffer[HOP_SIZE * 2:]  # hop forward

            # Convert to float32 PCM
            pcm16 = np.frombuffer(frame_bytes, dtype=np.int16).astype(np.float32)
            pcm_float = np.clip(pcm16 / 32768.0, -1.0, 1.0)
            tensor_chunk = torch.from_numpy(pcm_float)

            # Get speech probability
            speech_prob = model(tensor_chunk, SAMPLE_RATE).item()
            prob_buffer.append(speech_prob)
            avg_prob = sum(prob_buffer) / len(prob_buffer)

            #print(f"Speech prob: {speech_prob:.2f} | Avg: {avg_prob:.2f}")

            if avg_prob > SPEECH_PROB_THRESHOLD:
                print("✅ Speech detected")
                speech_detected_in_session = True
                silence_duration = 0.0
            else:
                silence_duration += HOP_SIZE / SAMPLE_RATE
                if silence_duration >= SILENCE_THRESHOLD_SECONDS:
                    if speech_detected_in_session:
                        print("🛑 Silence threshold reached, saving audio.")
                        save_audio(frames)
                    #else:
                        #print("⚠️ Silence but no speech detected — skipping save.")
                    frames = []
                    silence_duration = 0.0
                    speech_detected_in_session = False
                    prob_buffer.clear()

async def main():
    async with serve(echo, "0.0.0.0", 8000) as server:
        global turboWhisper, llamaClient
        llamaClient = Llama3("meta-llama/Llama-3.2-1B-Instruct")
        turboWhisper = WhisperTurbo("openai/whisper-large-v3-turbo")
        print("🎙️ WebSocket VAD server started at ws://localhost:8000")
        await server.serve_forever()

asyncio.run(main())
