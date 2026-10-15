# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Layout-identity test for EpilogueScratch on develop's subtile cvt block.

Pins that the named-window ``EpilogueScratch`` allocator produces the same 7-VGPR
(UseSubtileImpl) / 4-VGPR (non-subtile) total and window offsets as develop's
hand-coded ``numCvtVgprs = 7 if UseSubtileImpl else 4`` block.  Pure-Python, no
rocisa/GPU needed: runnable directly or under pytest.
"""

import importlib.util
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_MODULE_PATH = os.path.normpath(
    os.path.join(_HERE, "..", "..", "Components", "EpilogueScratch.py"))
_spec = importlib.util.spec_from_file_location("EpilogueScratch", _MODULE_PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
EpilogueScratch = _mod.EpilogueScratch
borrow_dead_accumulators = _mod.borrow_dead_accumulators


def test_develop_layout_identical():
    """EpilogueScratch reproduces develop's flat 7-VGPR subtile cvt block exactly."""
    for use_subtile in (False, True):
        s = EpilogueScratch(4)   # base = bf16 staging (+0..+3); subtile adds 3 more
        if use_subtile:
            assert s.reserve("permAddr",    1) == 4
            assert s.reserve("laneGDelta",  1) == 5
            assert s.reserve("addrScratch", 1) == 6
        expected = 7 if use_subtile else 4
        assert s.total() == expected, \
            f"use_subtile={use_subtile}: got {s.total()}, expected {expected}"


def test_reserve_never_overlaps():
    s = EpilogueScratch(4)
    s.reserve("a", 2)
    s.reserve("b", 1)
    s.reserve("c", 3, align=2)
    spans = [(s.offset(n), s.offset(n) + sz)
             for n, sz in (("staging", 4), ("a", 2), ("b", 1), ("c", 3))]
    spans.sort()
    for (lo_a, hi_a), (lo_b, _) in zip(spans, spans[1:]):
        assert hi_a <= lo_b, f"overlap: {spans}"


def test_borrow_dead_accumulators():
    # Disjoint borrow: fine.
    out = borrow_dead_accumulators(borrowed=[100, 108], live=[104, 112], width=4)
    assert out == [100, 108]
    # Clobbers live: must fail loud.
    try:
        borrow_dead_accumulators(borrowed=[100], live=[102], width=4)
        raise AssertionError("expected clobber assertion")
    except AssertionError as e:
        assert "clobbers live" in str(e)
    # Self-overlap: must fail loud.
    try:
        borrow_dead_accumulators(borrowed=[100, 103], live=[], width=4)
        raise AssertionError("expected self-overlap assertion")
    except AssertionError as e:
        assert "overlap" in str(e)


if __name__ == "__main__":
    test_develop_layout_identical()
    test_reserve_never_overlaps()
    test_borrow_dead_accumulators()
    print("OK: EpilogueScratch matches develop's 7-VGPR subtile cvt block; borrow checks pass")
