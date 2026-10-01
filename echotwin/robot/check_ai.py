"""Check the AI setup:  python -m echotwin.robot.check_ai"""
import asyncio
import time

import cv2
import numpy as np

from . import config
from .ai.brain import Brain


async def main():
    b = Brain()
    print(f"provider={config.AI_PROVIDER} key={'set' if config.AI_API_KEY else 'MISSING'} enabled={b.enabled}")
    print(f"text model:   {config.AI_MODEL}")
    print(f"vision model: {config.AI_VISION_MODEL}")
    if not b.enabled:
        return
    t = time.time()
    print("intent ->", await b.parse_everyday("could you shove the glass over beside the chocolate", ["glass", "chocolate bar"]),
          f"({time.time() - t:.1f}s)")
    img = np.full((480, 640, 3), 235, np.uint8)
    cv2.rectangle(img, (100, 300), (150, 350), (40, 40, 220), -1)
    cv2.rectangle(img, (420, 80), (540, 200), (60, 190, 60), -1)
    ok, buf = cv2.imencode(".jpg", img)
    t = time.time()
    print("vision ->", await b.describe(buf.tobytes()), f"({time.time() - t:.1f}s)")


asyncio.run(main())
