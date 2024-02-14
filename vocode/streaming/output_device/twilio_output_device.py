from __future__ import annotations
import json
import base64
import wave
import asyncio
import numpy as np
import os
import audioop  # For μ-law decoding
from fastapi import WebSocket
from typing import Optional
from vocode.streaming.output_device.base_output_device import BaseOutputDevice
from vocode.streaming.telephony.constants import DEFAULT_AUDIO_ENCODING
from vocode.streaming.models.audio_encoding import AudioEncoding
from vocode.streaming.utils.worker import ThreadAsyncWorker


class FileWriterWorker(ThreadAsyncWorker):
    def __init__(self, buffer_queue: asyncio.Queue, wav) -> None:
        super().__init__(buffer_queue)
        self.wav = wav

    def _run_loop(self):
        while True:
            try:
                chunk_arr = self.input_janus_queue.sync_q.get()
                self.wav.writeframes(chunk_arr.tobytes())
            except asyncio.CancelledError:
                return

    def terminate(self):
        super().terminate()
        self.wav.close()


class TwilioOutputDevice(BaseOutputDevice):
    DEFAULT_SAMPLING_RATE = 8000

    def __init__(
        self,
        ws: Optional[WebSocket] = None,
        stream_sid: Optional[str] = None,
        call_sid: Optional[str] = None,
        conversation_id: Optional[str] = None,
        sampling_rate: int = DEFAULT_SAMPLING_RATE,
        audio_encoding: AudioEncoding = DEFAULT_AUDIO_ENCODING,
    ):
        super().__init__(sampling_rate=sampling_rate, audio_encoding=audio_encoding)
        self.ws = ws
        self.stream_sid = stream_sid  # Use a private attribute
        self._call_sid = call_sid
        self.active = True
        self.queue: asyncio.Queue[str] = asyncio.Queue()
        self.process_task = asyncio.create_task(self.process())

        # Unified buffer queue
        self.buffer_queue: asyncio.Queue[np.ndarray] = asyncio.Queue()

        # Initialize the thread_worker as None
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
        # File writing setup
        file_directory = f"/tmp/conversations/{call_sid}"
        if not os.path.exists(file_directory):
            os.makedirs(file_directory)
        file_path = os.path.join(file_directory, "audio.wav")
        wav = wave.open(file_path, "wb")
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(self.sampling_rate)

        # Initialize FileWriterWorker with new file
        self.thread_worker = FileWriterWorker(self.buffer_queue, wav)
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
        # Existing Twilio message handling
        twilio_message = {
            "event": "media",
            "streamSid": self.stream_sid,
            "media": {"payload": base64.b64encode(chunk).decode("utf-8")},
        }
        self.queue.put_nowait(json.dumps(twilio_message))

        # Convert the decoded chunk to numpy array and write to file queue
        self._decode_and_enqueue(chunk)

    def is_silence(self, chunk: bytes, threshold: int = 500) -> bool:
        """
        Check if the given audio chunk is predominantly silence.
        The threshold can be adjusted based on requirements.
        """
        # Convert byte chunk to numpy array
        audio_array = np.frombuffer(chunk, dtype=np.int16)

        chunky = np.max(np.abs(audio_array))
        # Check if the maximum amplitude is below the threshold
        return chunky == 32510  # 🤷

    def consume_input_nonblocking(self, chunk: bytes):
        if not self.is_silence(chunk):
            # Process only if chunk is not silence
            self._decode_and_enqueue(chunk)

    def maybe_send_mark_nonblocking(self, message_sent):
        mark_message = {
            "event": "mark",
            "streamSid": self.stream_sid,
            "mark": {"name": f"Sent {message_sent}"},
        }
        self.queue.put_nowait(json.dumps(mark_message))

    def terminate(self):
        self.thread_worker.terminate()
        self.process_task.cancel()
