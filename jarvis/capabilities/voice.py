"""
Voice — Speech-to-Text (Whisper) + Text-to-Speech (pyttsx3).
Gracefully disabled when audio hardware is unavailable.
"""
from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Optional

from jarvis.config import config

logger = logging.getLogger(__name__)


class VoiceIO:
    def __init__(self):
        self._enabled = config.VOICE_ENABLED
        self._stt_engine = config.VOICE_STT_ENGINE
        self._tts_engine_name = config.VOICE_TTS_ENGINE
        self._whisper_model = None
        self._tts = None

        if self._enabled:
            self._init_tts()
            if self._stt_engine == "whisper":
                self._init_whisper()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_whisper(self) -> None:
        try:
            from faster_whisper import WhisperModel
            self._whisper_model = WhisperModel(
                config.VOICE_WHISPER_MODEL,
                device="cpu",
                compute_type="int8",
            )
            logger.info(f"Whisper STT loaded ({config.VOICE_WHISPER_MODEL})")
        except Exception as e:
            logger.warning(f"Whisper unavailable: {e}")
            self._enabled = False

    def _init_tts(self) -> None:
        try:
            import pyttsx3
            self._tts = pyttsx3.init()
            self._tts.setProperty("rate", 175)
            logger.info("pyttsx3 TTS ready.")
        except Exception as e:
            logger.warning(f"TTS unavailable: {e}")

    # ------------------------------------------------------------------
    # STT
    # ------------------------------------------------------------------

    def listen(self, timeout: int = 10, phrase_limit: int = 15) -> Optional[str]:
        """Listen from the microphone and return transcribed text."""
        if not self._enabled:
            return None
        try:
            import speech_recognition as sr
            recogniser = sr.Recognizer()
            with sr.Microphone() as source:
                logger.info("Listening…")
                recogniser.adjust_for_ambient_noise(source, duration=0.5)
                audio = recogniser.listen(source, timeout=timeout, phrase_time_limit=phrase_limit)

            if self._stt_engine == "whisper" and self._whisper_model:
                return self._transcribe_whisper(audio)
            else:
                return recogniser.recognize_google(audio)
        except Exception as e:
            logger.error(f"STT error: {e}")
            return None

    def _transcribe_whisper(self, audio) -> Optional[str]:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp = Path(f.name)
        try:
            tmp.write_bytes(audio.get_wav_data())
            segments, _ = self._whisper_model.transcribe(str(tmp), beam_size=5)
            return " ".join(s.text for s in segments).strip() or None
        finally:
            tmp.unlink(missing_ok=True)

    # ------------------------------------------------------------------
    # TTS
    # ------------------------------------------------------------------

    def speak(self, text: str) -> None:
        """Speak text aloud."""
        if not self._tts:
            return
        try:
            self._tts.say(text)
            self._tts.runAndWait()
        except Exception as e:
            logger.error(f"TTS error: {e}")

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    @property
    def available(self) -> bool:
        return self._enabled
