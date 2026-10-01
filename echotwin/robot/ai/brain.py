"""Provider-agnostic AI: understands unusual phrasing about the objects on the table, names objects from a photo.

Only used when the rules fail or for 'what do you see?' enrichment. Everything has a template
fallback, so a missing key or a rate limit never blocks the demo.
Providers: nvidia (build.nvidia.com) and openai share the OpenAI chat-completions format.
"""
import asyncio
import base64
import json
import re
import time

import cv2
import httpx
import numpy as np

from .. import config

BASE = {"nvidia": "https://integrate.api.nvidia.com/v1", "openai": "https://api.openai.com/v1"}

CHAT_PROMPT = (f"You are {config.ROBOT_NAME}, a small friendly robot arm that moves everyday objects on a table. "
               "Answer in one short spoken sentence, under 15 words. No emojis.")


class Brain:
    def __init__(self):
        self.enabled = bool(config.AI_API_KEY) and config.FEATURE_BRAIN and config.AI_PROVIDER in BASE
        self.base = BASE.get(config.AI_PROVIDER, "")
        self._last = 0.0
        self._lock = asyncio.Lock()
        self._down: dict[str, float] = {}

    async def _chat(self, model: str, messages: list, max_tokens=120, timeout=12.0) -> str | None:
        """Try the configured model, then the fallbacks; a model that is down is skipped for 10 minutes."""
        if not self.enabled:
            return None
        for m in dict.fromkeys([model, *config.AI_FALLBACK_MODELS]):
            if self._down.get(m, 0) > time.time():
                continue
            out = await self._chat_one(m, messages, max_tokens, timeout)
            if out:
                return out
        return None

    async def _chat_one(self, model: str, messages: list, max_tokens=120, timeout=12.0) -> str | None:
        async with self._lock:  # free tier ~40 req/min: space calls out
            wait = 1.5 - (time.time() - self._last)
            if wait > 0:
                await asyncio.sleep(wait)
            self._last = time.time()
        try:
            async with httpx.AsyncClient(timeout=timeout) as c:
                r = await c.post(f"{self.base}/chat/completions",
                                 headers={"Authorization": f"Bearer {config.AI_API_KEY}"},
                                 json={"model": model, "messages": messages, "max_tokens": max_tokens,
                                       "temperature": 0.2,
                                       # hybrid reasoning models: answer directly, we need speed
                                       "chat_template_kwargs": {"enable_thinking": False}})
            if r.status_code != 200:
                print("[brain]", model, r.status_code, r.text[:200])
                if r.status_code in (404, 410):
                    self._down[model] = time.time() + 600
                return None
            choice = r.json()["choices"][0]
            text = choice["message"].get("content") or ""
            if not text.strip():
                print("[brain] empty answer, finish_reason =", choice.get("finish_reason"))
            return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip() or None
        except Exception as e:
            print("[brain]", model, "error", type(e).__name__, e)
            return None

    async def parse_everyday(self, text: str, names: list[str]) -> dict | None:
        """Instruction about the real objects on the table -> {object, relation, reference} (1-based numbers)."""
        listing = "\n".join(f"{i}. {n}" for i, n in enumerate(names, 1))
        prompt = ("A table-top robot can move these objects on the table:\n" + listing + "\n"
                  "Turn the user's instruction into JSON only, one of:\n"
                  '{"object": <number>, "relation": "next to|left of|right of|in front of|behind", "reference": <number>}\n'
                  '{"object": <number>, "relation": "left|right|front|back|middle"}\n'
                  '{"clarify": "<short question>"}\n'
                  '{"action": "none"}   (not a request to move one of these objects)\n'
                  "Match loosely: 'swiss chocolate' is the chocolate, 'earbuds' is the earbud case.")
        out = await self._chat(config.AI_MODEL, [{"role": "system", "content": prompt},
                                                 {"role": "user", "content": text}], max_tokens=300)
        m = re.search(r"\{.*\}", out or "", re.S)
        try:
            return json.loads(m.group(0)) if m else None
        except json.JSONDecodeError:
            return None

    async def chat(self, text: str, scene: str | None = None) -> str | None:
        if scene:
            sys_prompt = (f"You are {config.ROBOT_NAME}, a small friendly robot arm working on a digital twin of a real "
                          f"table. On the table: {scene}. You can move these objects next to each other, or left, "
                          "right, forward, back. Only mention objects from that list. "
                          "Answer in one short spoken sentence, under 15 words. No emojis.")
            return await self._chat(config.AI_MODEL, [{"role": "system", "content": sys_prompt},
                                                      {"role": "user", "content": text}], max_tokens=200)
        return await self._chat(config.AI_MODEL, [{"role": "system", "content": CHAT_PROMPT},
                                                  {"role": "user", "content": text}], max_tokens=60)

    async def describe(self, jpeg: bytes, question: str = "What do you see on the table?") -> str | None:
        img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return None
        s = 512 / max(img.shape[:2])
        if s < 1:
            img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
        b64 = base64.b64encode(buf.tobytes()).decode()
        prompt = (f"{question} You are a robot looking at a table with everyday objects on it. "
                  "Answer in at most two short spoken sentences.")
        return await self._chat(config.AI_VISION_MODEL, [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}]}],
            max_tokens=150, timeout=15.0)

    async def image_json(self, jpeg: bytes, prompt: str, max_tokens=1600) -> dict | None:
        """Ask the vision model a question about an image and parse a JSON answer."""
        b64 = base64.b64encode(jpeg).decode()
        out = await self._chat(config.AI_VISION_MODEL, [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}]}],
            max_tokens=max_tokens, timeout=45.0)
        m = re.search(r"\{.*\}", out or "", re.S)
        try:
            return json.loads(m.group(0)) if m else None
        except json.JSONDecodeError:
            return None
