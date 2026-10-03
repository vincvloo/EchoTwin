"""MuJoCo scene generation: table, everyday objects (props), static scenery, floating gripper.

Coordinates: the table centre is the origin, x to the right, y away from the user ("back"), z up.
Sim metres are real metres. The table size is part of the layout (`Layout.table_half`).
"""
from dataclasses import dataclass, field

DEFAULT_TABLE_HALF = (0.40, 0.30)       # half size of the table (x, y) in metres, when nothing says otherwise
TABLE_ASPECT = 1188 / 840               # width / height of the table texture the quick importer builds
HOME = (0.0, -0.16, 0.20)


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


def _f(*v) -> str:
    return " ".join(f"{x:.4f}" for x in v)


def build_xml(layout: Layout) -> str:
    tw, th = layout.table_half
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
    {"".join(scenery)}
    {"".join(fixed)}
    {"".join(props)}
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
