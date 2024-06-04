from __future__ import annotations
import asyncio
import json
import base64
import numpy as np
import os
import audioop
from pydub import AudioSegment
import av
from fastapi import WebSocket
from typing import Optional
from vocode.streaming.output_device.base_output_device import BaseOutputDevice
from vocode.streaming.telephony.constants import DEFAULT_AUDIO_ENCODING
from vocode.streaming.models.audio_encoding import AudioEncoding
from vocode.streaming.utils.worker import ThreadAsyncWorker

class FileWriterWorker(ThreadAsyncWorker):
    def __init__(self, buffer_queue: asyncio.Queue, output_container) -> None:
        super().__init__(buffer_queue)
        self.output_container = output_container
        self.stream = self.output_container.add_stream('libopus', rate=8000)

    def _run_loop(self):
        while True:
            try:
                chunk_arr = self.input_janus_queue.sync_q.get()
                frame = av.AudioFrame.from_ndarray(chunk_arr, format='s16', layout='mono')
                frame.rate = 8000
                for packet in self.stream.encode(frame):
                    self.output_container.mux(packet)
            except asyncio.CancelledError:
                packets = self.stream.encode(None)
                for packet in packets:
                    self.output_container.mux(packet)
                return

    def terminate(self):
        super().terminate()
        self.output_container.close()

class TwilioOutputDevice(BaseOutputDevice):
    DEFAULT_SAMPLING_RATE = 8000

    def __init__(
        self, ws: Optional[WebSocket] = None, stream_sid: Optional[str] = None, 
        call_sid: Optional[str] = None, conversation_id: Optional[str] = None, 
        sampling_rate: int = DEFAULT_SAMPLING_RATE, audio_encoding: AudioEncoding = DEFAULT_AUDIO_ENCODING
    ):
        super().__init__(sampling_rate=sampling_rate, audio_encoding=audio_encoding)
        self.ws = ws
        self.stream_sid = stream_sid
        self._call_sid = call_sid
        self.active = True
        self.queue: asyncio.Queue[str] = asyncio.Queue()
        self.process_task = asyncio.create_task(self.process())
        self.buffer_queue: asyncio.Queue[np.ndarray] = asyncio.Queue()
        self.thread_worker = None
        if call_sid is not None:
            self.setup_file_writer(call_sid)

    @property
    def call_sid(self):
        return self._call_sid

    @call_sid.setter
    def call_sid(self, value):
        self._call_sid = value
        self.setup_file_writer(value)

    def setup_file_writer(self, call_sid):
        print("Setting up file writer")
        file_directory = f"/tmp/conversations/{call_sid}"
        if not os.path.exists(file_directory):
            os.makedirs(file_directory)
        file_path = os.path.join(file_directory, "audio.webm")
        output_container = av.open(file_path, mode='w')
        self.thread_worker = FileWriterWorker(self.buffer_queue, output_container)
        self.thread_worker.start()

    async def process(self):
        while self.active:
            message = await self.queue.get()
            await self.ws.send_text(message)

    def _decode_and_enqueue(self, chunk: bytes):
        try:
            decoded_chunk = audioop.ulaw2lin(chunk, 2)
        except audioop.error as e:
            print(f"Error decoding μ-law: {e}")
            return
        chunk_arr = np.frombuffer(decoded_chunk, dtype=np.int16)
        self.buffer_queue.put_nowait(chunk_arr)

    def consume_nonblocking(self, chunk: bytes):
        twilio_message = {
            "event": "media",
            "streamSid": self.stream_sid,
            "media": {"payload": base64.b64encode(chunk).decode("utf-8")},
        }
        self.queue.put_nowait(json.dumps(twilio_message))
        self._decode_and_enqueue(chunk)

    def terminate(self):
        self.thread_worker.terminate()
        self.process_task.cancel()