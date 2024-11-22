from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from whisper_online import *
import argparse
import logging
import numpy as np
import io
import soundfile
import librosa
import sys
import os

app = FastAPI()

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Parameters
SAMPLING_RATE = 16000

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--host", type=str, default='0.0.0.0')
parser.add_argument("--port", type=int, default=43007)
parser.add_argument("--warmup-file", type=str, dest="warmup_file")

add_shared_args(parser)
args = parser.parse_args()

set_logging(args, logger, other="")

# Setting whisper object by args
size = args.model
language = args.lan
asr, online = asr_factory(args)
min_chunk = args.min_chunk_size

# Warm up the ASR
msg = "Whisper is not warmed up. The first chunk processing may take longer."
if args.warmup_file:
    if os.path.isfile(args.warmup_file):
        a = load_audio_chunk(args.warmup_file, 0, 1)
        asr.transcribe(a)
        logger.info("Whisper is warmed up.")
    else:
        logger.critical("The warm up file is not available. " + msg)
        sys.exit(1)
else:
    logger.warning(msg)

class ServerProcessor:
    def __init__(self, websocket, online_asr_proc, min_chunk):
        self.websocket = websocket
        self.online_asr_proc = online_asr_proc
        self.min_chunk = min_chunk
        self.last_end = None
        self.is_first = True

    async def receive_audio_chunk(self):
        minlimit = self.min_chunk * SAMPLING_RATE
        while True:
            raw_bytes = await self.websocket.receive_bytes()
            sf = soundfile.SoundFile(io.BytesIO(raw_bytes), channels=1, endian="LITTLE", samplerate=SAMPLING_RATE, subtype="PCM_16", format="RAW")
            audio, _ = librosa.load(sf, sr=SAMPLING_RATE, dtype=np.float32)
            if self.is_first and len(audio) < minlimit:
                continue
            self.is_first = False
            return audio

    

    async def send_result(self, o):
        await self.websocket.send_text(o[2])
        

    async def process(self):
        self.online_asr_proc.init()
        while True:
            a = await self.receive_audio_chunk()
            if a is None:
                break
            self.online_asr_proc.insert_audio_chunk(a)
            o = online.process_iter()
            try:
                await self.send_result(o)
            except WebSocketDisconnect:
                logger.info("WebSocket connection closed")
                break

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connection established")
    proc = ServerProcessor(websocket, online, args.min_chunk_size)
    await proc.process()
    logger.info("WebSocket connection closed")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)