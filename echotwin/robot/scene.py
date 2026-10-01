"""MuJoCo scene generation: table, coloured cubes, movable zones, floating gripper.

Coordinates: the A4 sheet (297 x 210 mm) is the workspace, centred at the origin,
x to the right, y away from the user ("back"), z up. The sim is SCALE x real size so
small props become comfortably graspable cubes.
"""
from dataclasses import dataclass, field

SCALE = 2.0
SHEET_MM = (297.0, 210.0)
WS_HALF = (SHEET_MM[0] / 2000 * SCALE, SHEET_MM[1] / 2000 * SCALE)  # 0.297 x 0.21 m
TABLE_HALF = (2 * WS_HALF[0], 2 * WS_HALF[1])                      # scan texture covers this
CUBE_HALF = 0.022
HOME = (0.0, -0.16, 0.20)

OBJECT_NAMES = ("red", "blue", "yellow")
ZONE_NAMES = ("green", "tray")
ZONE_LABEL = {"green": "green zone", "tray": "blue tray"}

RGB = {
    "red": (0.86, 0.14, 0.12),
    "blue": (0.14, 0.34, 0.88),
    "yellow": (0.96, 0.80, 0.12),
    "green": (0.16, 0.72, 0.30),
    "tray": (0.14, 0.40, 0.95),
}

DEFAULT_ZONES = {
    "green": {"pos": (0.17, 0.12), "half": (0.07, 0.06)},
    "tray": {"pos": (-0.17, 0.12), "half": (0.07, 0.06)},
}


@dataclass
class Layout:
    """Where things are. Objects: name -> {pos:(x,y), half, present}. Zones: name -> {pos, half}."""
    objects: dict = field(default_factory=lambda: {
        n: {"pos": (x, -0.08), "half": CUBE_HALF, "present": True}
        for n, x in zip(OBJECT_NAMES, (-0.15, 0.0, 0.15))})
    zones: dict = field(default_factory=lambda: {k: dict(v) for k, v in DEFAULT_ZONES.items()})
    texture: str | None = None  # absolute path to a top-down table texture (from a scan)
    props: list = field(default_factory=list)  # everyday objects: {name, pos, size(half xyz), rgb, shape}
    show_zones: bool = True
    view: dict | None = None  # camera where the phone was: {pos, xyaxes, fovy}
    # 3D scans: scenery meshes (visual only): {file, texture?, pos(3), euler(3, deg), scale}
    scene: list = field(default_factory=list)
    meta: dict = field(default_factory=dict)  # e.g. {"sim_scale": sim metres per real metre, "name": ...}

    def copy(self) -> "Layout":
        return Layout({k: dict(v) for k, v in self.objects.items()},
                      {k: dict(v) for k, v in self.zones.items()}, self.texture,
                      [dict(p) for p in self.props], self.show_zones, self.view,
                      [dict(m) for m in self.scene], dict(self.meta))


def _f(*v) -> str:
    return " ".join(f"{x:.4f}" for x in v)


def build_xml(layout: Layout) -> str:
    tw, th = TABLE_HALF
    if layout.texture:
        tex = layout.texture.replace("\\", "/")
        table_asset = (f'<texture name="tabletex" type="2d" file="{tex}"/>'
                       '<material name="tabletop" texture="tabletex" texrepeat="1 1" texuniform="false"/>')
        sheet = ""
    else:
        table_asset = '<material name="tabletop" rgba="0.62 0.50 0.38 1"/>'
        sheet = (f'<geom name="sheet" type="box" pos="0 0 0.0006" size="{_f(WS_HALF[0], WS_HALF[1], 0.0005)}" '
                 'rgba="0.95 0.95 0.93 1" contype="0" conaffinity="0"/>')

    zones = []
    for name, z in layout.zones.items():
        r, g, b = RGB[name]
        hx, hy = z["half"]
        zones.append(
            f'<body name="zone_{name}" pos="{_f(z["pos"][0], z["pos"][1], 0)}">'
            f'<geom name="zone_{name}" type="box" pos="0 0 0.0015" size="{_f(hx, hy, 0.001)}" '
            f'rgba="{_f(r, g, b)} {0.75 if layout.show_zones else 0}" contype="0" conaffinity="0"/></body>')

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
                               f'<material name="skin_{i}" texture="skin_{i}" rgba="1 1 1 1"/>')
            look = f'material="skin_{i}"'
        elif pr.get("skin") and not pr.get("mesh"):
            # the object's own photo on every face (cube texture, same image on all six faces)
            prop_assets.append(f'<texture name="skin_{i}" type="cube" file="{pr["skin"].replace(chr(92), "/")}"/>'
                               f'<material name="skin_{i}" texture="skin_{i}" rgba="1 1 1 {alpha}"/>')
            look = f'material="skin_{i}"'
        else:
            look = f'rgba="{_f(r, g, b)} {alpha}"'
        props.append(f'<body name="prop_{i}" pos="{_f(pr["pos"][0], pr["pos"][1], hz + 0.0005)}" euler="0 0 {pr.get("yaw", 0):.1f}">'
                     f'<freejoint name="prop_{i}"/>'
                     f'<geom {geom} {look} mass="0.08" friction="1.5 0.05 0.01" condim="6"/></body>')

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

    objs = []
    for i, (name, o) in enumerate(layout.objects.items()):
        r, g, b = RGB[name]
        h = o["half"]
        if o.get("present", True):
            pos, alpha = (o["pos"][0], o["pos"][1], h + 0.0005), 1
        else:  # parked on the floor, invisible
            pos, alpha = (2.0 + 0.2 * i, 2.0, -0.75 + h), 0
        objs.append(
            f'<body name="obj_{name}" pos="{_f(*pos)}"><freejoint name="obj_{name}"/>'
            f'<geom name="obj_{name}" type="box" size="{_f(h, h, h)}" rgba="{_f(r, g, b)} {alpha}" '
            'mass="0.05" friction="1.2 0.02 0.001" condim="4"/></body>')

    fingers = "".join(
        f'<body name="finger_{s}" mocap="true" pos="{_f(HOME[0] + dx, HOME[1], HOME[2] + 0.008)}">'
        '<geom type="box" size="0.006 0.013 0.026" rgba="0.82 0.84 0.88 1" contype="0" conaffinity="0"/></body>'
        for s, dx in (("l", -0.045), ("r", 0.045)))

    return f"""
<mujoco model="puppeteer">
  <option timestep="0.002"/>
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
    <light pos="0.3 -0.6 1.6" dir="-0.15 0.35 -1" diffuse="0.7 0.7 0.7" castshadow="true"/>
    <geom name="floor" type="plane" pos="0 0 -0.75" size="4 4 0.1" {'rgba="0 0 0 0"' if layout.scene else 'material="floor"'}/>
    <geom name="table" type="box" pos="0 0 -0.02" size="{_f(tw + 0.02, th + 0.02, 0.02)}" rgba="0.36 0.27 0.20 1"/>
    <geom name="tabletop" type="plane" pos="0 0 0.0002" size="{_f(tw, th, 0.01)}" material="tabletop" contype="0" conaffinity="0"/>
    {"".join(f'<geom type="box" pos="{_f(sx * (tw - 0.04), sy * (th - 0.04), -0.39)}" size="0.025 0.025 0.35" rgba="0.3 0.23 0.17 {0 if layout.scene else 1}"/>' for sx in (-1, 1) for sy in (-1, 1))}
    {sheet}
    {"".join(zones)}
    {"".join(scenery)}
    {"".join(props)}
    {"".join(objs)}
    <body name="hand" mocap="true" pos="{_f(*HOME)}">
      <geom type="box" pos="0 0 0.042" size="0.056 0.016 0.009" rgba="0.18 0.19 0.22 1" contype="0" conaffinity="0"/>
      <geom type="cylinder" pos="0 0 0.16" size="0.014 0.11" rgba="0.30 0.32 0.36 1" contype="0" conaffinity="0"/>
      <geom type="sphere" pos="0 0 0" size="0.004" rgba="1 1 1 0.5" contype="0" conaffinity="0"/>
    </body>
    {fingers}
    <camera name="main" pos="0 -0.95 0.78" xyaxes="1 0 0 0 0.64 0.77"/>
    <camera name="top" pos="0 0 1.35" xyaxes="1 0 0 0 1 0"/>
    {f'<camera name="photo" pos="{_f(*layout.view["pos"])}" xyaxes="{_f(*layout.view["xyaxes"])}" fovy="{layout.view["fovy"]:.1f}"/>' if layout.view else ""}
  </worldbody>
</mujoco>"""
