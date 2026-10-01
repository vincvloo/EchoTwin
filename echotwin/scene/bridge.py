"""Sonar-to-Robot Bridge: Converts Sonar 3D object map outputs into Robot Puppeteer twin format.

Usage:
    python -m echotwin.scene.bridge --objects path/to/lounge_objects_objects.json --cloud path/to/lounge_vggt.ply --out out_twin.zip
    python -m echotwin.scene.bridge --objects path/to/objects.json --server https://localhost:8443
"""

import argparse
import json
import os
import shutil
import sys
import zipfile
from pathlib import Path

# Map common YOLO / segmentation classes to Puppeteer 3D shapes & colors
CLASS_CONFIG = {
    # Boxes / Furniture
    "chair": {"shape": "box", "color": "#4287f5"},
    "couch": {"shape": "box", "color": "#34a853"},
    "sofa": {"shape": "box", "color": "#34a853"},
    "table": {"shape": "box", "color": "#8d6e63"},
    "desk": {"shape": "box", "color": "#8d6e63"},
    "bed": {"shape": "box", "color": "#ab47bc"},
    "cabinet": {"shape": "box", "color": "#78909c"},
    "box": {"shape": "box", "color": "#fbc02d"},
    "laptop": {"shape": "box", "color": "#607d8b"},
    
    # Cylindrical objects
    "potted plant": {"shape": "cylinder", "color": "#2e7d32"},
    "plant": {"shape": "cylinder", "color": "#2e7d32"},
    "bottle": {"shape": "cylinder", "color": "#0288d1"},
    "cup": {"shape": "cylinder", "color": "#e64a19"},
    "vase": {"shape": "cylinder", "color": "#8e24aa"},
    "can": {"shape": "cylinder", "color": "#d81b60"},
    
    # Flat objects
    "book": {"shape": "flat", "color": "#5d4037"},
    "paper": {"shape": "flat", "color": "#eceff1"},
    "cell phone": {"shape": "flat", "color": "#212121"},
    
    # Round objects
    "sports ball": {"shape": "round", "color": "#f57c00"},
    "apple": {"shape": "round", "color": "#d32f2f"},
    "orange": {"shape": "round", "color": "#f57c00"},
}

DEFAULT_PROP = {"shape": "box", "color": "#9e9e9e"}


def sonar_objects_to_twin_doc(sonar_data: dict, cloud_path: Path | None = None, name: str = "Sonar Twin", sim_scale: float = 2.0) -> dict:
    """Converts Sonar objects JSON data to a Puppeteer twin document dictionary."""
    objects_in = sonar_data.get("objects", [])
    map_meta = sonar_data.get("map", {})
    
    # Find origin / centroid of objects to center scene around table/room center
    if objects_in:
        avg_x = sum(o["x"] for o in objects_in) / len(objects_in)
        avg_y = sum(o["y"] for o in objects_in) / len(objects_in)
    else:
        avg_x = map_meta.get("origin", [0, 0])[0] + (map_meta.get("width", 100) * map_meta.get("res", 0.03)) / 2
        avg_y = map_meta.get("origin", [0, 0])[1] + (map_meta.get("height", 100) * map_meta.get("res", 0.03)) / 2

    twin_objects = []
    for i, obj in enumerate(objects_in):
        cls_name = obj.get("class", "object").lower()
        cfg = CLASS_CONFIG.get(cls_name, DEFAULT_PROP)
        
        # Sonar values are in metres, relative to map frame.
        # Shift relative to average center, convert to cm
        rel_x_m = obj["x"] - avg_x
        rel_y_m = obj["y"] - avg_y
        
        size_x_cm = round(max(2.0, float(obj.get("size_x", 0.1)) * 100), 1)
        size_y_cm = round(max(2.0, float(obj.get("size_y", 0.1)) * 100), 1)
        height_cm = round(max(1.0, float(obj.get("height", 0.1) or 0.1) * 100), 1)
        
        pos_x_cm = round(rel_x_m * 100, 1)
        pos_y_cm = round(rel_y_m * 100, 1)
        
        twin_objects.append({
            "name": f"{cls_name} {i + 1}",
            "shape": cfg["shape"],
            "size_cm": [size_x_cm, size_y_cm, height_cm],
            "pos_cm": [pos_x_cm, pos_y_cm],
            "yaw_deg": 0.0,
            "color": cfg["color"]
        })

    doc = {
        "format": "phone-puppeteer-twin",
        "version": 1,
        "name": name,
        "sim_scale": sim_scale,
        "show_zones": False,
        "objects": twin_objects,
        "blocks": [],
        "zones": [],
        "scene": []
    }

    if cloud_path and cloud_path.exists():
        rel_cloud = f"scene/{cloud_path.name}"
        doc["scene"].append({
            "mesh": rel_cloud,
            "pos_cm": [0.0, 0.0, -10.0],
            "euler_deg": [0, 0, 0],
            "scale": 1.0
        })

    return doc


def export_sonar_twin_zip(sonar_data: dict, cloud_path: Path | None, out_zip_path: Path, name: str = "Sonar Twin") -> None:
    """Exports a twin .zip file with twin.json and referenced 3D point cloud mesh."""
    out_zip_path.parent.mkdir(parents=True, exist_ok=True)
    doc = sonar_objects_to_twin_doc(sonar_data, cloud_path=cloud_path, name=name)
    
    with zipfile.ZipFile(out_zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        # Write twin.json
        z.writestr("twin.json", json.dumps(doc, indent=2))
        
        # Copy mesh file if present
        if cloud_path and cloud_path.exists():
            z.write(cloud_path, f"scene/{cloud_path.name}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Convert Sonar 3D object detections to Robot Puppeteer twin format.")
    parser.add_argument("--objects", required=True, help="Path to <out>_objects.json file from Sonar")
    parser.add_argument("--cloud", help="Optional path to 3D cloud (.ply / .glb)")
    parser.add_argument("--out", default="sonar_twin.zip", help="Output .zip or .json twin path (default: sonar_twin.zip)")
    parser.add_argument("--name", default="Sonar Scan Twin", help="Twin scene name")
    
    args = parser.parse_args(argv)
    
    obj_path = Path(args.objects)
    if not obj_path.exists():
        print(f"Error: Objects file not found at {obj_path}", file=sys.stderr)
        sys.exit(1)
        
    sonar_data = json.loads(obj_path.read_text(encoding="utf-8"))
    cloud_p = Path(args.cloud) if args.cloud else None
    
    out_p = Path(args.out)
    if out_p.suffix == ".zip":
        export_sonar_twin_zip(sonar_data, cloud_p, out_p, name=args.name)
        print(f"Successfully generated twin package: {out_p.resolve()}")
    else:
        doc = sonar_objects_to_twin_doc(sonar_data, cloud_p, name=args.name)
        out_p.write_text(json.dumps(doc, indent=2), encoding="utf-8")
        print(f"Successfully generated twin JSON: {out_p.resolve()}")


if __name__ == "__main__":
    main()
