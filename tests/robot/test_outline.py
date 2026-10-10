"""Which objects are built from their own outline: decided by the outline, not only by the shape label."""
import numpy as np

from echotwin.robot.features.everyday import built_from_outline


def _mask(widths):
    """A silhouette from its width at each row (top row first), centred."""
    w = max(widths)
    m = np.zeros((len(widths), w + 4), bool)
    for i, k in enumerate(widths):
        a = (w + 4 - k) // 2
        m[i, a:a + k] = True
    return m


def test_a_straight_sided_box_stays_a_box():
    assert not built_from_outline("box", _mask([40] * 50))


def test_a_tapered_tub_called_a_box_is_built_from_its_outline():
    assert built_from_outline("box", _mask(list(np.linspace(46, 28, 50).astype(int))))


def test_a_rounded_blob_called_a_box_is_built_from_its_outline():
    yy, xx = np.mgrid[-25:26, -25:26]
    assert built_from_outline("box", xx ** 2 + yy ** 2 <= 25 ** 2)


def test_round_and_cylinder_labels_still_are_and_flat_never_is():
    assert built_from_outline("cylinder", _mask([40] * 50)) and built_from_outline("round", _mask([40] * 50))
    assert not built_from_outline("flat", _mask(list(np.linspace(46, 28, 50).astype(int))))
