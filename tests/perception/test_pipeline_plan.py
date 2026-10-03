"""The command list for the photo pipeline (standard library only)."""
from pathlib import Path

from echotwin.perception import pipeline as PL


def test_plan_runs_gpu_steps_in_the_gpu_env_and_the_rest_in_the_maps_env():
    steps = PL.plan(Path("in"), Path("out"), gpu_py="gpu.exe", maps_py="maps.exe", frames=10, cam_height=0.5)
    assert [s.argv[0] for s in steps] == ["gpu.exe", "gpu.exe", "maps.exe", "maps.exe"]
    assert [s.argv[2] for s in steps] == ["echotwin.perception." + m for m in ("reconstruct", "detect", "objects", "review")]
    assert steps[0].argv[steps[0].argv.index("--frames") + 1] == "10"
    assert steps[-1].optional and not any(s.optional for s in steps[:-1])
    assert "{scale}" in steps[2].argv and "--min-area" in steps[2].argv      # tabletop settings
    assert Path(steps[3].argv[-1]) == PL.scene_path(Path("out")) or steps[3].argv[-1].endswith("objects_scene.json")


def test_plan_without_review_and_for_a_room():
    steps = PL.plan(Path("in"), Path("out"), gpu_py="g", review=False, table=False)
    assert len(steps) == 3 and "--min-area" not in steps[2].argv and steps[2].argv[0] == "g"


def test_scale_is_read_from_the_reconstruction_output_and_filled_in():
    out = "camera height above floor: 0.88 units | scale hint if you held the phone at 0.45 m: --scale 0.512\n"
    assert PL.find_scale(out) == 0.512 and PL.find_scale("nothing here") is None
    assert PL.fill(["a", "{scale}"], 0.512) == ["a", "0.5120"]
    assert PL.fill(["a", "{scale}"], None) == ["a", "1.0"]


def test_settings_and_reasons_it_cannot_run(tmp_path):
    cfg = PL.settings({"PERCEPTION_PY": "x", "SCAN_FRAMES": "8", "SCAN_MODE": "QUICK"})
    assert cfg["frames"] == 8 and cfg["mode"] == "quick" and cfg["maps_py"] == cfg["gpu_py"]
    assert PL.settings({})["mode"] == "auto" and "PERCEPTION_PY" in PL.usable(PL.settings({}), 10)
    ok = {**PL.settings({}), "gpu_py": str(tmp_path)}
    assert PL.usable(ok, 10) is None and "3 photos" in PL.usable(ok, 2)


def test_paths_in_env_are_relative_to_the_repository_so_they_work_anywhere(tmp_path):
    cfg = PL.settings({"PERCEPTION_PY": ".venv-perception/bin/python", "PERCEPTION_MAPS_PY": "maps/python"})
    assert Path(cfg["gpu_py"]) == PL.REPO / ".venv-perception/bin/python"
    assert Path(cfg["maps_py"]) == PL.REPO / "maps/python"
    absolute = str(tmp_path / "py")
    assert PL.settings({"PERCEPTION_PY": absolute})["gpu_py"] == absolute        # absolute paths are left alone
    assert PL.resolve("") == ""


def test_commands_use_paths_relative_to_the_repository():
    out = PL.REPO / "data" / "robot" / "scans" / "s1"
    steps = PL.plan(out / "input", out, gpu_py="g", vggt_path="")
    joined = " ".join(" ".join(s.argv) for s in steps)
    assert str(PL.REPO) not in joined and "data/robot/scans/s1/cloud.ply" in joined
    assert steps[0].env is None


def test_vggt_source_folder_goes_on_the_gpu_steps_python_path(tmp_path):
    vggt = tmp_path / "vggt"
    vggt.mkdir()
    steps = PL.plan(Path("in"), Path("out"), gpu_py="g", maps_py="m", vggt_path=str(vggt))
    assert steps[0].env["PYTHONPATH"].startswith(str(vggt)) and steps[1].env == steps[0].env
    assert steps[2].env is None                                                  # maps steps do not need it
    assert PL.settings({"VGGT_PATH": str(vggt)})["vggt_path"] == str(vggt)
    assert PL.settings({"VGGT_PATH": str(tmp_path / "missing")})["vggt_path"] == ""
