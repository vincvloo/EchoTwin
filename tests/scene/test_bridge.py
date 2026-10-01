"""Tests for sonar_to_robot bridge."""

import json
import tempfile
from pathlib import Path
import pytest

from echotwin.scene.bridge import sonar_objects_to_twin_doc, export_sonar_twin_zip
from echotwin.robot import twin_import as TI
from echotwin.robot.world import World


def test_sonar_objects_conversion():
    sample_sonar_json = {
        "map": {"origin": [-2.0, -2.0], "res": 0.03, "width": 100, "height": 100},
        "objects": [
            {
                "class": "chair",
                "x": -1.0,
                "y": 0.5,
                "size_x": 0.5,
                "size_y": 0.5,
                "height": 0.8,
                "photos": 5,
                "points": 1000
            },
            {
                "class": "potted plant",
                "x": 0.5,
                "y": -0.5,
                "size_x": 0.3,
                "size_y": 0.3,
                "height": 0.4,
                "photos": 3,
                "points": 500
            }
        ]
    }
    
    doc = sonar_objects_to_twin_doc(sample_sonar_json)
    assert doc["format"] == "phone-puppeteer-twin"
    assert len(doc["objects"]) == 2
    
    chair = doc["objects"][0]
    assert chair["name"] == "chair 1"
    assert chair["shape"] == "box"
    assert chair["size_cm"] == [50.0, 50.0, 80.0]
    
    plant = doc["objects"][1]
    assert plant["name"] == "potted plant 2"
    assert plant["shape"] == "cylinder"
    assert plant["size_cm"] == [30.0, 30.0, 40.0]


def test_sonar_export_and_puppeteer_import():
    sample_sonar_json = {
        "map": {"origin": [0.0, 0.0], "res": 0.03, "width": 100, "height": 100},
        "objects": [
            {
                "class": "couch",
                "x": 0.2,
                "y": 0.3,
                "size_x": 1.2,
                "size_y": 0.8,
                "height": 0.7,
                "photos": 8,
                "points": 5000
            }
        ]
    }
    
    d = Path(tempfile.mkdtemp())
    zip_path = d / "sonar_twin.zip"
    export_sonar_twin_zip(sample_sonar_json, cloud_path=None, out_zip_path=zip_path)
    
    assert zip_path.exists()
    
    # Verify reading with TI.read_upload and doc_to_layout
    doc = TI.read_upload(zip_path.read_bytes(), "sonar_twin.zip", d / "extracted")
    assert doc["format"] == "phone-puppeteer-twin"
    
    lay = TI.doc_to_layout(doc, d / "extracted")
    world = World(lay)
    world.settle(10)
    assert len(world.layout.props) == 1
    assert world.layout.props[0]["name"] == "couch 1"
