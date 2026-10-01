"""F6 voice: one persona, short lines, cached ElevenLabs audio, browser TTS fallback, barge-in.

The dashboard is the speaker. Every line is emitted as {"t": "say", text, audio?}; if ElevenLabs is
configured the audio URL points at a cached mp3, otherwise the browser speaks the text itself.
"""
import hashlib
import queue
import threading

import httpx

from . import config

CACHED_PHRASES = [
    "Stopped.", "Continuing.", "Gripping.", "Releasing.", "Can you show me?",
    "I haven't learned that yet. Can you show me?", "Done.", "Thanks. Try me again.", "Keep this demo?",
    "Discarded.",
    "Your turn.", "Looking…",
    "I'm fairly sure about this one.", "I'm confident about this one.",
]


class Voice:
    def __init__(self, emit):
        self.emit = emit
        self.enabled = bool(config.ELEVENLABS_API_KEY)
        self.q: queue.Queue = queue.Queue()
        self.last = ""
        if self.enabled:
            threading.Thread(target=self._worker, daemon=True).start()
            threading.Thread(target=self._precache, daemon=True).start()

    @staticmethod
    def _key(text: str) -> str:
        return hashlib.sha1(f"{config.ELEVENLABS_VOICE_ID}|{text}".encode()).hexdigest()[:16]

    def _path(self, text: str):
        return config.VOICE_CACHE / f"{self._key(text)}.mp3"

    def _synth(self, text: str) -> bool:
        p = self._path(text)
        if p.exists():
            return True
        try:
            r = httpx.post(
                f"https://api.elevenlabs.io/v1/text-to-speech/{config.ELEVENLABS_VOICE_ID}?output_format=mp3_44100_64",
                headers={"xi-api-key": config.ELEVENLABS_API_KEY},
                json={"text": text, "model_id": config.ELEVENLABS_MODEL,
                      "voice_settings": {"stability": 0.55, "similarity_boost": 0.8}},
                timeout=10)
            if r.status_code == 200:
                p.write_bytes(r.content)
                return True
            print("[voice] ElevenLabs", r.status_code, r.text[:120])
        except Exception as e:
            print("[voice] ElevenLabs error", e)
        return False

    def _precache(self):
        for line in CACHED_PHRASES:
            self._synth(line)

    def _worker(self):
        while True:
            text = self.q.get()
            ok = self._synth(text)
            self.emit({"t": "say", "text": text, "audio": f"/voice/{self._key(text)}.mp3" if ok else None})

    def say(self, text: str):
        self.last = text
        self.emit({"t": "log", "who": "robot", "text": text})
        if self.enabled and self._path(text).exists():
            self.emit({"t": "say", "text": text, "audio": f"/voice/{self._key(text)}.mp3"})
        elif self.enabled:
            self.q.put(text)
        else:
            self.emit({"t": "say", "text": text, "audio": None})

    def hush(self):
        with self.q.mutex:
            self.q.queue.clear()
        self.emit({"t": "hush"})
