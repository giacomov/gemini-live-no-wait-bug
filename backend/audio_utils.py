"""Audio utilities: TTS conversion and Gemini audio streaming."""

import asyncio
import os
import random
from typing import Any

import numpy as np
from google.genai.live import AsyncSession
from google.genai.types import Blob


# 5 seconds of silence at 16kHz 16-bit PCM — signals end of speech to Gemini
_SILENCE = b"\x00" * (16000 * 2 * 5)

# ~80ms chunks, matching pocket-tts streaming output size
_AUDIO_CHUNK_BYTES = 2560


def _make_leading_silence(min_sec: float = 0.5, max_sec: float = 2.0) -> bytes:
    """Return a random-duration silence to simulate natural response latency."""
    duration = random.uniform(min_sec, max_sec)
    samples = int(duration * 16000)
    return b"\x00" * (samples * 2)


class AudioConverter:
    """Encapsulates TTS state and audio streaming to a Gemini Live session."""

    def __init__(self, voice: str = "alba") -> None:
        self._voice = voice
        self._tts_model: Any = None
        self._voice_state: Any = None

    def _load(self) -> None:
        """Lazy-load the TTS model on first use."""
        if self._tts_model is not None:
            return
        try:
            from pocket_tts import TTSModel  # type: ignore[import-untyped]
        except ImportError as e:
            raise ImportError("pocket-tts not found. Install with: uv add pocket-tts") from e
        self._tts_model = TTSModel.load_model()
        voice_name = os.getenv("POCKET_TTS_VOICE", self._voice)
        self._voice_state = self._tts_model.get_state_for_audio_prompt(voice_name)

    def text_to_audio_bytes(self, text: str, trailing_silence_sec: float = 0.5) -> bytes:
        """Convert text to PCM16 audio bytes at 16kHz."""
        self._load()
        model_sample_rate = self._tts_model.sample_rate

        audio_tensor = self._tts_model.generate_audio(self._voice_state, text)
        audio = audio_tensor.numpy()

        if model_sample_rate != 16000:
            duration = len(audio) / model_sample_rate
            new_length = int(duration * 16000)
            audio = np.interp(
                np.linspace(0, len(audio) - 1, new_length),
                np.arange(len(audio)),
                audio,
            )

        audio_int16 = (audio * 32767).astype(np.int16)

        if trailing_silence_sec > 0:
            silence = np.zeros(int(trailing_silence_sec * 16000), dtype=np.int16)
            audio_int16 = np.concatenate([audio_int16, silence])

        return audio_int16.tobytes()

    async def send_audio(self, session: AsyncSession, audio: bytes, watcher: asyncio.Event) -> bool:
        """Stream audio in ~80ms chunks, preceded by a random leading silence."""
        
        leading = _make_leading_silence()
        await session.send_realtime_input(media=Blob(data=leading, mime_type="audio/pcm;rate=16000"))

        for offset in range(0, len(audio), _AUDIO_CHUNK_BYTES):
            chunk = audio[offset : offset + _AUDIO_CHUNK_BYTES]

            # Keep emitting only while the model is still idle (hasn't started a new turn)
            if watcher.is_set():
                await session.send_realtime_input(media=Blob(data=chunk, mime_type="audio/pcm;rate=16000"))
            else:
                return False

        await session.send_realtime_input(media=Blob(data=_SILENCE, mime_type="audio/pcm;rate=16000"))

        return True