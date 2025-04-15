import asyncio
import wave
from websockets.asyncio.server import serve

async def echo(websocket):
    frames = []
    async for message in websocket:
        print(f"Received: {message}")
        frames.append(message)

    # Save the received frames to a WAV file
    with wave.open("output.wav", "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # Assuming 16-bit audio
        wf.setframerate(16000)
        wf.writeframes(b''.join(frames))
    print("Audio saved to output.wav")

async def main():
    async with serve(echo, "localhost", 8000) as server:
        await server.serve_forever()

asyncio.run(main())