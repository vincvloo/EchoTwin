"""Looking at the table: a camera, and an observer that turns a frame into where an object is.

`RenderSource` is the simulation's camera: a top-down render of a MuJoCo world (the mock arm's own world, say). A real
camera would be a `CameraSource` plus a `PlaneMap` calibrated to the table (a homography; nothing here calibrates one, and
no real camera was available to test it). `CameraObserver` does the looking: it finds the object near where it is expected
(`features/locate.py`) and says where it is on the table. The robot then moves its belief (the twin) to that place.
"""
from __future__ import annotations

import mujoco
import numpy as np

from .features import locate as L

PAD_DZ = 0.03             # the pads seen from above are centred about this far above the tool point (calibrated on the render)
SIZE = (640, 480)            # frame size (width, height)
CAMERA = "top"


class FrameSource:
    def frame(self) -> np.ndarray:
        """A BGR frame of the table, looking at it now."""
        raise NotImplementedError


class RenderSource(FrameSource):
    """A render of one of the simulation's cameras. The renderer is made on first use, on the thread that asks (OpenGL
    contexts belong to a thread), and again when the world was rebuilt."""

    def __init__(self, world_of, camera: str = CAMERA, size=SIZE):
        self.world_of, self.camera, self.size = world_of, camera, size
        self._r = None
        self._model = None

    def frame(self) -> np.ndarray:
        w = self.world_of()
        if self._r is None or self._model is not w.model:
            if self._r is not None:
                self._r.close()
            self._r = mujoco.Renderer(w.model, self.size[1], self.size[0])
            self._model = w.model
        self._r.update_scene(w.data, camera=self.camera)
        self._r.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False         # soft room light: shadows would fatten every object
        return self._r.render()[:, :, ::-1].copy()

    def plane(self, world) -> L.PlaneMap:
        """The pixel <-> table map of this camera in `world` (a MuJoCo camera: position, axes, field of view)."""
        m = world.model
        cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, self.camera)
        d = world.data
        R = d.cam_xmat[cid].reshape(3, 3)                       # columns: right, up, backwards
        return L.PlaneMap.from_camera(d.cam_xpos[cid], np.r_[R[:, 0], R[:, 1]], float(m.cam_fovy[cid]), self.size)


class CameraSource(FrameSource):
    """A real camera through OpenCV. Not tested (no camera here)."""

    def __init__(self, index: int = 0):
        import cv2
        self.cap = cv2.VideoCapture(index)

    def frame(self) -> np.ndarray:
        ok, img = self.cap.read()
        if not ok:
            raise RuntimeError("the camera gave no frame")
        return img


class CameraObserver:
    """Where is this object, by looking? `plane` maps the camera's pixels to the table; it may be a PlaneMap or a function
    returning one (the simulation's camera moves with a rebuilt world)."""

    def __init__(self, source: FrameSource, plane):
        self.source, self._plane = source, plane

    @property
    def plane(self) -> L.PlaneMap:
        return self._plane() if callable(self._plane) else self._plane

    def observe(self, name: str, expected_xy, size_m: float, z: float, others=()) -> np.ndarray | None:
        """The table (x, y) of the object seen near `expected_xy`, or None if the camera does not see it there.
        `size_m`: about how wide it is, `z`: the height of its middle. `others`: [(x, y, z)] of the other known objects."""
        plane = self.plane
        frame = self.source.frame()
        px = plane.to_pixel(expected_xy, z)
        size_px = plane.metres_to_pixels(size_m, expected_xy, z)
        near = [plane.to_pixel(o[:2], o[2]) for o in others]
        f = L.locate(frame, px, size_px, max_area=3.5 * size_px ** 2, others=near)      # a blob much bigger than the object is something else (the arm)
        if f is None:
            return None
        return plane.to_table(f.px, z)

    def observe_tool(self, expected_xy, z: float, jaw_sep_m: float, yaw: float = 0.0) -> np.ndarray | None:
        """Where the gripper really is: the table (x, y) under the midpoint of its two pads, seen by the camera, or None.
        `expected_xy`, `z`: where the joints say the tool is; `jaw_sep_m`: the distance between the pads; `yaw`: the direction they close along."""
        plane = self.plane
        zp = z + PAD_DZ
        px = plane.to_pixel(expected_xy, zp)
        mid = L.find_pads(self.source.frame(), px, plane.metres_to_pixels(jaw_sep_m, expected_xy, zp), axis=(np.cos(yaw), -np.sin(yaw)),   # image y points down
                          need_pair=True)       # a lone pad is a guess: too coarse to steer a gripper by
        return None if mid is None else plane.to_table(mid, zp)
