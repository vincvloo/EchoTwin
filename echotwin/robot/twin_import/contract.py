"""The contract between importers and the rest of the app.

An importer turns outside data (photos, a layout file, a 3D scan...) into a Layout and hands it to the app
through a TwinContext. It never touches the simulator, the server or the websocket directly.

    async def my_photo_importer(frames: list[bytes], pitches: list[float | None], ctx: TwinContext) -> None:
        ...
        ctx.apply(layout, summary)          # show the twin (required)
        ctx.rename(props, "I see ...")      # optional: better names/shapes later (e.g. after an AI call)

Plug your own photo importer in without touching this package:
    TWIN_PHOTO_IMPORTER=my_package.my_module:my_photo_importer     (in .env)
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Awaitable, Callable

from ..scene import Layout


@dataclass
class TwinContext:
    apply: Callable[[Layout, dict], None]
    """Show a new twin. summary keys used by the dashboard: id, mode ('blocks'|'everyday'|'file'), greeting,
    twin (image url), texture (image url), props (list of names), objects, thumbs, views, frames, seconds."""
    rename: Callable[[list, str], None]
    """Replace the everyday objects (e.g. with AI names and shapes) and say a line."""
    say: Callable[[str], None]
    progress: Callable[[str, dict], None]
    """Dashboard scan events: stage in {'processing','frames','view','everyday','done','error'} + data."""
    new_dir: Callable[[], tuple[str, Path]]
    """-> (id, folder) for this import's files; the folder is served at /scans/<id>/."""
    ask_ai_json: Callable[[bytes, str], Awaitable[dict | None]] | None = None
    """Vision model: (jpeg, prompt) -> parsed JSON answer, or None when unavailable."""
    extra: dict = field(default_factory=dict)


@dataclass
class ImportError_(Exception):
    message: str

    def __str__(self):
        return self.message
