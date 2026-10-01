"""Twin import: everything that turns outside data into a digital twin (a Layout). See README.md here.

Public API (the only things the server uses):
    TwinContext, ImportError_                       the contract (contract.py)
    photo_importer()                                the active photo importer (default: photos.import_photos)
    layout_to_doc / doc_to_layout / export_zip / read_upload    twin files (layout_file.py)
    convert_mesh                                    3D scan -> mesh (meshes.py)
"""
import importlib
import os

from .contract import ImportError_, TwinContext
from .layout_file import doc_to_layout, export_zip, layout_to_doc, read_upload
from .meshes import convert as convert_mesh
from .meshes import fit_scale


def photo_importer():
    """TWIN_PHOTO_IMPORTER=package.module:function swaps in another photo importer (same signature)."""
    spec = os.getenv("TWIN_PHOTO_IMPORTER", "").strip()
    if spec:
        mod, _, fn = spec.partition(":")
        return getattr(importlib.import_module(mod), fn)
    from .photos import import_photos
    return import_photos


__all__ = ["TwinContext", "ImportError_", "photo_importer", "layout_to_doc", "doc_to_layout", "export_zip",
           "read_upload", "convert_mesh", "fit_scale"]
