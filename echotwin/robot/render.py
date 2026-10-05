"""The picture of the twin that both pages show: the scene, the planned path as dots, and a banner when it matters.

Everything here runs in the sim thread (an OpenGL context belongs to the thread that made it).
"""
import cv2
import mujoco
import numpy as np

W, H = 800, 600
RED = (40, 40, 230)     # BGR


def _text(img, text, xy, scale, color, outline, thick):
    cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), outline, cv2.LINE_AA)
    cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


class FrameRenderer:
    def __init__(self, model):
        self._r = mujoco.Renderer(model, H, W)

    def reopen(self, model):
        """The twin was rebuilt: the renderer needs the new model."""
        self._r.close()
        self._r = mujoco.Renderer(model, H, W)

    def jpeg(self, view, ghost, uncertainty: float, overlay: tuple | None) -> bytes | None:
        """One frame as JPEG. `ghost`: points of the planned path, coloured by how unsure the plan is.
        `overlay`: ("halted", reason) | ("practice", progress, this try) | ("rec",) | None."""
        r = self._r
        r.update_scene(view.data, camera="photo" if view.layout.view else "main")
        if ghost is not None and len(ghost):
            scn = r.scene
            rgba = np.array([0.2, 0.9, 0.5, 0.55] if uncertainty < 0.45 else [1.0, 0.75, 0.2, 0.55] if uncertainty < 0.75
                            else [1.0, 0.35, 0.3, 0.55], dtype=np.float32)
            for p in ghost[::3]:
                if scn.ngeom >= scn.maxgeom:
                    break
                mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                                    np.array([0.005, 0, 0]), p.astype(np.float64), np.eye(3).flatten(), rgba)
                scn.ngeom += 1
        bgr = cv2.cvtColor(r.render(), cv2.COLOR_RGB2BGR)
        if overlay:
            self._draw(bgr, overlay)
        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return buf.tobytes() if ok else None

    @staticmethod
    def _draw(bgr, overlay):
        kind = overlay[0]
        if kind == "halted":
            cv2.rectangle(bgr, (0, 0), (W - 1, H - 1), RED, 10)
            cv2.putText(bgr, f"STOPPED ({overlay[1]})", (24, 52), cv2.FONT_HERSHEY_SIMPLEX, 1.1, RED, 3, cv2.LINE_AA)
        elif kind == "practice":
            _text(bgr, overlay[1], (20, H - 70), 0.8, (255, 200, 120), 5, 2)
            if overlay[2]:
                _text(bgr, overlay[2], (20, H - 36), 0.65, (255, 255, 255), 4, 1)
        elif kind == "rec":
            cv2.circle(bgr, (30, 34), 11, RED, -1)
            cv2.putText(bgr, "REC", (50, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, RED, 2, cv2.LINE_AA)
