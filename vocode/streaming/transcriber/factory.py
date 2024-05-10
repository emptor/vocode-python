import logging
from typing import Optional, Dict, Type
from vocode.streaming.models.transcriber import (
    AssemblyAITranscriberConfig,
    AzureTranscriberConfig,
    DeepgramTranscriberConfig,
    GoogleTranscriberConfig,
    RevAITranscriberConfig,
    GladiaTranscriberConfig,
    TranscriberConfig,
    TranscriberType,
)
from vocode.streaming.transcriber.assembly_ai_transcriber import AssemblyAITranscriber
from vocode.streaming.transcriber.deepgram_transcriber import DeepgramTranscriber
from vocode.streaming.transcriber.google_transcriber import GoogleTranscriber
from vocode.streaming.transcriber.rev_ai_transcriber import RevAITranscriber
from vocode.streaming.transcriber.azure_transcriber import AzureTranscriber
from vocode.streaming.transcriber.gladia_transcriber import GladiaTranscriber

class TranscriberFactory:
    def __init__(self):
        self.transcriber_map: Dict[Type[TranscriberConfig], Type[TranscriberBase]] = {
            DeepgramTranscriberConfig: DeepgramTranscriber,
            GoogleTranscriberConfig: GoogleTranscriber,
            AssemblyAITranscriberConfig: AssemblyAITranscriber,
            RevAITranscriberConfig: RevAITranscriber,
            AzureTranscriberConfig: AzureTranscriber,
            GladiaTranscriberConfig: GladiaTranscriber,
        }

    def create_transcriber(
        self,
        transcriber_config: TranscriberConfig,
        logger: Optional[logging.Logger] = None,
    ):
        transcriber_class = self.transcriber_map.get(type(transcriber_config))
        if transcriber_class:
            return transcriber_class(transcriber_config, logger=logger)
        else:
            raise Exception("Invalid transcriber config")
