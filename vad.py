import asyncio
import datetime
import numpy as np
import torch
from collections import deque
import tempfile
import wave
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# Constants
SAMPLE_RATE = 16000
WINDOW_SIZE = 512
HOP_SIZE = 256
SILENCE_THRESHOLD_SECONDS = 1.0
SPEECH_PROB_THRESHOLD = 0.3
PROB_SMOOTHING_WINDOW = 5


class VadAudioProcessor:
    def __init__(self, vad_model):
        self.vad = vad_model
        self.buffer = b""
        self.frames = []
        self.temp_filename = None
        self.silence_duration = 0.0
        self.prob_buffer = deque(maxlen=PROB_SMOOTHING_WINDOW)
        self.speech_detected = False

    def process_audio(self, message):
        self.buffer += message
        self.frames.append(message)

        audio_to_return = None

        # Process overlapping windows
        while len(self.buffer) >= (WINDOW_SIZE * 2):
            avg_prob = self.audio_prbability_from_buffer(self.buffer)
            if avg_prob > SPEECH_PROB_THRESHOLD:
                print("✅ Speech detected")
                self.speech_detected = True
                self.silence_duration = 0.0
            else:
                self.silence_duration += HOP_SIZE / SAMPLE_RATE
                if self.silence_duration >= SILENCE_THRESHOLD_SECONDS:
                    if self.speech_detected:
                        print("🛑 Silence threshold reached, saving audio.")
                        self.temp_filename = self.save_audio_to_temp(self.frames)

        if self.temp_filename:
            audio_file = self.temp_filename
            self.reset()
            return audio_file

    def save_audio_to_temp(self, frames):
        """Save recorded audio frames to a temporary .wav file."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
            tmp_filename = tmpfile.name
            with wave.open(tmpfile, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  # 16-bit audio
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(b"".join(frames))
        print(f"💾 Audio temporarily saved to {tmp_filename}")
        return tmp_filename

    def audio_prbability_from_buffer(self, buffer):
        frame_bytes = self.buffer[: WINDOW_SIZE * 2]
        self.buffer = self.buffer[HOP_SIZE * 2 :]  # move forward

        # Convert to PCM float
        pcm16 = np.frombuffer(frame_bytes, dtype=np.int16).astype(np.float32)
        pcm_float = np.clip(pcm16 / 32768.0, -1.0, 1.0)
        tensor_chunk = torch.from_numpy(pcm_float)

        # Speech probability
        speech_prob = self.vad(tensor_chunk, SAMPLE_RATE).item()
        self.prob_buffer.append(speech_prob)
        return sum(self.prob_buffer) / len(self.prob_buffer)

    def reset(self):
        self.frames = []
        self.buffer = b""
        self.silence_duration = 0.0
        self.speech_detected = False
        self.prob_buffer.clear()
        self.temp_filename = None
