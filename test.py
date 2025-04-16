# from websockets.sync.client import connect

# def hello():
#     with connect("ws://localhost:8000") as websocket:
#         websocket.send("Hello world!")

# hello()

# import webrtcvad
# vad = webrtcvad.Vad()
# vad.set_mode(1)
# # Run the VAD on 10 ms of silence. The result should be False.
# sample_rate = 16000
# frame_duration = 10  # ms
# frame = b'\x00\x00' * int(sample_rate * frame_duration / 1000)
# print ('Contains speech: %s' % (vad.is_speech(frame, sample_rate)))

from silero_vad import load_silero_vad, read_audio, get_speech_timestamps
model = load_silero_vad()
wav = read_audio('output.wav')
speech_timestamps = get_speech_timestamps(
  wav,
  model,
  return_seconds=True,  # Return speech timestamps in seconds (default is samples)
)
print(speech_timestamps)