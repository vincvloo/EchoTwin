"""An arm descriptor can say how its pads touch things ("pad_contact"), without editing the arm's own file."""
import mujoco
import numpy as np
import pytest

from echotwin.robot import arm as A


def _pads(model, arm):
    return [g for g in range(model.ngeom) if model.geom(g).name.startswith(arm.prefix) and any(p in model.geom(g).name for p in arm.pads)]


def _compile(arm):
    spec = mujoco.MjSpec.from_string("<mujoco><worldbody/></mujoco>")
    A.compose(spec, arm)
    return spec.compile()


def test_pad_contact_sets_friction_and_condim_on_the_pads_only():
    base = A.load("builtin")
    tuned = A.parse({**_descriptor("builtin"), "pad_contact": {"friction": [2.0, 0.03, 0.003], "condim": 6}})
    m0, m1 = _compile(base), _compile(tuned)
    pads = _pads(m1, tuned)
    assert len(pads) == 2
    for g in pads:
        assert m1.geom_friction[g] == pytest.approx([2.0, 0.03, 0.003]) and m1.geom_condim[g] == 6
    others = [g for g in range(m1.ngeom) if g not in pads]
    assert (m1.geom_friction[others] == m0.geom_friction[others]).all()                   # nothing else changed


def test_unknown_keys_or_no_pads_are_errors():
    with pytest.raises(A.ArmError, match="unknown key"):
        _compile(A.parse({**_descriptor("builtin"), "pad_contact": {"grip": 1}}))
    with pytest.raises(A.ArmError, match="no geom name"):
        _compile(A.parse({**_descriptor("builtin"), "pads": ["nothing_like_this"], "pad_contact": {"condim": 4}}))


def _descriptor(name):
    import json
    return json.loads((A.ARMS_DIR / f"{name}.json").read_text(encoding="utf-8"))
