import asyncio
import os
from websockets.asyncio.server import serve
import warnings
import os
import torch
from turbo import WhisperTurbo
from llama import Llama3
from piper import PiperTTS
from vad import VadAudioProcessor

warnings.filterwarnings("ignore", category=FutureWarning)


async def echo(websocket):
    """Main websocket handler."""
    vadProcessor = VadAudioProcessor(vadModel)
    async for message in websocket:
        if message:
            temp_audio_file = vadProcessor.process_audio(message)
            if temp_audio_file:
                transcription = turboWhisper.transcribe(temp_audio_file)
                print(f"WHISPER: {transcription}")

                llama_response = llamaClient.invoke_bedrock(transcription)
                detected_lang = llama_response.get("language", "unknown")
                translation = llama_response.get("translation", "")
                if detected_lang != "unknown":
                    print(f"LLAMA: {translation}")

                    if detected_lang == "English":
                        piper_voice = "voices/en_US-hfc_male-medium.onnx"
                    elif detected_lang == "Spanish":
                        piper_voice = "voices/es_ES-carlfm-x_low.onnx"

                    audio_bytes = piperClient.synthesize(translation, piper_voice)
                    chunk_size = 4096  # 4KB at a time
                    for i in range(0, len(audio_bytes), chunk_size):
                        chunk = audio_bytes[i : i + chunk_size]
                        await websocket.send(chunk)
                else:
                    print("LLAMA: Unknown language detected, no response generated.")
                os.remove(temp_audio_file)


async def main():
    """Server entry point."""
    global vadModel, turboWhisper, llamaClient, piperClient
    # Initialize clients
    vadModel, _ = torch.hub.load(
        repo_or_dir="snakers4/silero-vad", model="silero_vad", force_reload=False
    )
    llamaClient = Llama3(
        "arn:aws:bedrock:us-west-2:934985413136:inference-profile/us.meta.llama3-3-70b-instruct-v1:0"
    )
    turboWhisper = WhisperTurbo("openai/whisper-large-v3-turbo")
    piperClient = PiperTTS()

    print("🎙️ WebSocket VAD server started at ws://0.0.0.0:8000")
    async with serve(echo, "0.0.0.0", 8000):
        await asyncio.Future()  # Run forever


if __name__ == "__main__":
    asyncio.run(main())
