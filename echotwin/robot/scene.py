"""MuJoCo scene generation: table, everyday objects (props), static scenery, floating gripper.

Coordinates: the table centre is the origin, x to the right, y away from the user ("back"), z up.
Sim metres are real metres. The table size is part of the layout (`Layout.table_half`).
"""
from dataclasses import dataclass, field

import numpy as np

DEFAULT_TABLE_HALF = (0.40, 0.30)       # half size of the table (x, y) in metres, when nothing says otherwise
TABLE_ASPECT = 1188 / 840               # width / height of the table texture the quick importer builds
SOLID_TABLE_MIN = DEFAULT_TABLE_HALF    # the table is never smaller than this: a photo shows a patch of it, not all of it


@dataclass
class Layout:
    """Where things are. props: everyday objects {name, pos, size (half xyz), rgb, shape, ...}."""
    texture: str | None = None  # absolute path to a top-down table texture (from a scan)
    props: list = field(default_factory=list)
    view: dict | None = None  # camera where the phone was: {pos, xyaxes, fovy}
    # 3D scans: scenery meshes (visual only): {file, texture?, pos(3), euler(3, deg), scale}
    scene: list = field(default_factory=list)
    meta: dict = field(default_factory=dict)  # e.g. {"name": ...}
    # fixed furniture next to the table: {name, shape, pos, size (half xyz), rgb, yaw}; solid, never moved
    obstacles: list = field(default_factory=list)
    table_half: tuple = DEFAULT_TABLE_HALF   # real metres

    def copy(self) -> "Layout":
        return Layout(self.texture, [dict(p) for p in self.props], self.view,
                      [dict(m) for m in self.scene], dict(self.meta), [dict(o) for o in self.obstacles],
                      tuple(self.table_half))

    def scaled(self, k: float) -> "Layout":
        """The whole scene k times bigger (or smaller): table, objects, furniture, scanned meshes and the phone's camera.
        The arm is not touched, so this is also "the arm k times smaller". `meta["scale"]` keeps the product so far."""
        lay = self.copy()
        lay.table_half = tuple(v * k for v in self.table_half)
        for p in lay.props:
            p["size"] = tuple(v * k for v in p["size"])
            p["pos"] = tuple(v * k for v in p["pos"])
            if p.get("mesh_scale"):
                p["mesh_scale"] = tuple(v * k for v in p["mesh_scale"])
        for o in lay.obstacles:
            o["size"] = tuple(v * k for v in o["size"])
            o["pos"] = tuple(v * k for v in o["pos"])
        for m in lay.scene:
            m["scale"] = m.get("scale", 1.0) * k
            if m.get("pos") is not None:
                m["pos"] = tuple(v * k for v in m["pos"])
        if lay.view:
            lay.view = {**lay.view, "pos": [v * k for v in lay.view["pos"]]}
        lay.meta["scale"] = round(float(self.meta.get("scale", 1.0)) * k, 6)
        return lay


DENSITY = 300.0     # kg/m3, everyday objects are mostly hollow or light


def prop_mass(pr: dict) -> float:
    """Mass from size and shape (kg), between 20 and 400 g."""
    hx, hy, hz = pr["size"]
    shape = pr.get("shape")
    if shape == "cylinder":
        vol = np.pi * ((hx + hy) / 2) ** 2 * 2 * hz
    elif shape == "round":
        vol = 4 / 3 * np.pi * hx * hy * hz
    else:
        vol = 8 * hx * hy * hz
    return float(np.clip(vol * DENSITY, 0.02, 0.4))


def _f(*v) -> str:
    return " ".join(f"{x:.4f}" for x in v)


ATLAS_SUFFIX = "_cube.png"    # a skin with this name holds the six faces (3 rows x 4 columns), see features/everyday.cube_atlas


def _skin_texture(i: int, path: str) -> str:
    """The object's own look. One image on all six faces, or (name ends in _cube.png) one tile per face."""
    path = path.replace("\\", "/")
    grid = ' gridsize="3 4" gridlayout=".U..LFRB.D.."' if path.endswith(ATLAS_SUFFIX) else ""
    return f'<texture name="skin_{i}" type="cube" file="{path}"{grid}/>'


def _edge_colour(texture: str | None) -> str:
    """The colour around the patch the photo covers: the average of the texture's border (a wood brown without a texture)."""
    if texture:
        try:
            import cv2
            img = cv2.imread(texture)
            if img is not None:
                b = max(2, min(img.shape[:2]) // 40)
                edge = np.concatenate([img[:b].reshape(-1, 3), img[-b:].reshape(-1, 3),
                                       img[:, :b].reshape(-1, 3), img[:, -b:].reshape(-1, 3)])
                r, g, bl = (edge.mean(0)[::-1] / 255).tolist()
                return f"{r:.3f} {g:.3f} {bl:.3f} 1"
        except Exception:                          # an unreadable texture: the plain colour below
            pass
    return "0.62 0.50 0.38 1"


def build_xml(layout: Layout) -> str:
    tw, th = layout.table_half                      # the patch that is mapped (texture, goals, where the arm stands)
    sw, sh = max(tw, SOLID_TABLE_MIN[0]), max(th, SOLID_TABLE_MIN[1])     # the table itself, at least a normal one
    around = (f'<geom name="tablesurround" type="plane" pos="0 0 0.0001" size="{_f(sw, sh, 0.01)}" '
              f'rgba="{_edge_colour(layout.texture)}" contype="0" conaffinity="0"/>') if (sw, sh) != (tw, th) else ""
    if layout.texture:
        tex = layout.texture.replace("\\", "/")
        table_asset = (f'<texture name="tabletex" type="2d" file="{tex}"/>'
                       '<material name="tabletop" texture="tabletex" texrepeat="1 1" texuniform="false"/>')
    else:
        table_asset = '<material name="tabletop" rgba="0.62 0.50 0.38 1"/>'

    props, prop_assets = [], []
    for i, pr in enumerate(layout.props):
        hx, hy, hz = pr["size"]
        r, g, b = pr["rgb"]
        shape = pr.get("shape")
        if pr.get("mesh"):  # a 3D-scanned object: mesh centred on its bounding box, scaled to fit "size"
            sc = pr.get("mesh_scale", (1, 1, 1))
            prop_assets.append(f'<mesh name="mesh_{i}" file="{pr["mesh"].replace(chr(92), "/")}" scale="{_f(*sc)}"/>')
            geom = f'type="mesh" mesh="mesh_{i}"'
        elif shape == "cylinder":
            geom = f'type="cylinder" size="{_f((hx + hy) / 2, hz)}"'
        elif shape == "round":
            geom = f'type="ellipsoid" size="{_f(hx, hy, hz)}"'
        else:
            geom = f'type="box" size="{_f(hx, hy, hz)}"'
        alpha = 0.6 if shape == "cylinder" else 1
        if pr.get("mesh_texture"):
            prop_assets.append(f'<texture name="skin_{i}" type="2d" file="{pr["mesh_texture"].replace(chr(92), "/")}"/>'
                               f'<material name="skin_{i}" texture="skin_{i}" rgba="1 1 1 1" specular="0.15" shininess="0.1"/>')
            look = f'material="skin_{i}"'
        elif pr.get("skin") and not pr.get("mesh"):
            prop_assets.append(_skin_texture(i, pr["skin"]) +
                               f'<material name="skin_{i}" texture="skin_{i}" rgba="1 1 1 {alpha}" specular="0.15" shininess="0.1"/>')
            look = f'material="skin_{i}"'
        else:
            look = f'rgba="{_f(r, g, b)} {alpha}"'
        props.append(f'<body name="prop_{i}" pos="{_f(pr["pos"][0], pr["pos"][1], hz + 0.0005)}" euler="0 0 {pr.get("yaw", 0):.1f}">'
                     f'<freejoint name="prop_{i}"/>'
                     f'<geom {geom} {look} mass="{prop_mass(pr):.4f}" friction="1.5 0.05 0.01" condim="6"/></body>')

    fixed = []
    for i, ob in enumerate(layout.obstacles):
        hx, hy, hz = ob["size"]
        r, g, b = ob["rgb"]
        shape = f'type="cylinder" size="{_f((hx + hy) / 2, hz)}"' if ob.get("shape") == "cylinder"             else f'type="box" size="{_f(hx, hy, hz)}"'
        fixed.append(f'<geom name="obstacle_{i}" {shape} pos="{_f(ob["pos"][0], ob["pos"][1], hz)}" '
                     f'euler="0 0 {ob.get("yaw", 0):.1f}" rgba="{_f(r, g, b)} 0.4"/>')

    scenery = []
    for k, m in enumerate(layout.scene):
        prop_assets.append(f'<mesh name="scene_{k}" file="{m["file"].replace(chr(92), "/")}" scale="{_f(*([m.get("scale", 1.0)] * 3))}"/>')
        if m.get("texture"):
            prop_assets.append(f'<texture name="scene_tex_{k}" type="2d" file="{m["texture"].replace(chr(92), "/")}"/>'
                               f'<material name="scene_mat_{k}" texture="scene_tex_{k}"/>')
            look = f'material="scene_mat_{k}"'
        else:
            look = 'rgba="0.75 0.75 0.78 1"'
        scenery.append(f'<geom name="scene_{k}" type="mesh" mesh="scene_{k}" pos="{_f(*m.get("pos", (0, 0, 0)))}" '
                       f'euler="{_f(*m.get("euler", (0, 0, 0)))}" {look} contype="0" conaffinity="0"/>')

    return f"""
<mujoco model="puppeteer">
  <option timestep="0.002" cone="elliptic" impratio="10"/>
  <visual>
    <global offwidth="1280" offheight="960"/>
    <quality shadowsize="4096"/>
    <headlight ambient="0.35 0.35 0.35" diffuse="0.45 0.45 0.45" specular="0.1 0.1 0.1"/>
  </visual>
  <asset>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.24 0.27 0.32" rgb2="0.07 0.08 0.10" width="256" height="256"/>
    <texture name="floor" type="2d" builtin="checker" rgb1="0.20 0.22 0.25" rgb2="0.16 0.17 0.20" width="256" height="256"/>
    <material name="floor" texture="floor" texrepeat="10 10"/>
    {table_asset}
    {"".join(prop_assets)}
  </asset>
  <worldbody>
    <light pos="0.3 -0.6 1.4" dir="-0.15 0.35 -1" diffuse="0.7 0.7 0.7" castshadow="true"/>
    <geom name="floor" type="plane" pos="0 0 -0.75" size="4 4 0.1" {'rgba="0 0 0 0"' if layout.scene else 'material="floor"'}/>
    <geom name="table" type="box" pos="0 0 -0.02" size="{_f(sw + 0.02, sh + 0.02, 0.02)}" rgba="0.36 0.27 0.20 1"/>
    {around}
    <geom name="tabletop" type="plane" pos="0 0 0.0002" size="{_f(tw, th, 0.01)}" material="tabletop" contype="0" conaffinity="0"/>
    {"".join(f'<geom type="box" pos="{_f(sx * (sw - 0.04), sy * (sh - 0.04), -0.39)}" size="0.025 0.025 0.35" rgba="0.3 0.23 0.17 {0 if layout.scene else 1}"/>' for sx in (-1, 1) for sy in (-1, 1))}
    {"".join(scenery)}
    {"".join(fixed)}
    {"".join(props)}
    <camera name="main" pos="0.55 -0.62 0.55" xyaxes="0.758 0.652 -0.000 -0.347 0.403 0.847"/>
    <camera name="top" pos="0 0 1.0" xyaxes="1 0 0 0 1 0"/>
    {f'<camera name="photo" pos="{_f(*layout.view["pos"])}" xyaxes="{_f(*layout.view["xyaxes"])}" fovy="{layout.view["fovy"]:.1f}"/>' if layout.view else ""}
  </worldbody>
</mujoco>"""
