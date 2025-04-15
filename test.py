from websockets.sync.client import connect

def hello():
    with connect("ws://localhost:8000") as websocket:
        websocket.send("Hello world!")

hello()