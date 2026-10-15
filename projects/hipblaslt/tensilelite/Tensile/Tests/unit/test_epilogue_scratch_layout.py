# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Layout-identity test for EpilogueScratch (Rung 1 of the subtile regalloc refactor).

Pins that the named-window ``EpilogueScratch`` allocator produces byte-for-byte the same
cvt-block offsets and total size as the former hand-advanced ``nxt`` cursor in
``KernelWriterAssembly.getGlobalWriteBatchComponents`` -- i.e. that introducing it is a
bit-exact refactor at the layout-arithmetic level.  Pure-Python, no rocisa/GPU needed:
runnable directly (``python3 test_epilogue_scratch_layout.py``) or under pytest.
"""

import importlib.util
import os

# Load EpilogueScratch.py directly by path so the test has zero package-import side
# effects (no heavy Tensile __init__, no rocisa).
_HERE = os.path.dirname(os.path.abspath(__file__))
_MODULE_PATH = os.path.normpath(
    os.path.join(_HERE, "..", "..", "Components", "EpilogueScratch.py"))
_spec = importlib.util.spec_from_file_location("EpilogueScratch", _MODULE_PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
EpilogueScratch = _mod.EpilogueScratch


def legacy_layout(use_subtile, is_fold, scalar, pack_pairs, col128):
    """Verbatim reproduction of the pre-refactor hand cursor (the code being replaced)."""
    base = 7 if use_subtile else 4
    nxt = base
    foldVoff = foldBlend = -1
    if is_fold:
        foldVoff, foldBlend = nxt, nxt + 1
        nxt += 2
    scalarAddrOff = scalarRingOff = -1
    if scalar:
        scalarAddrOff, scalarRingOff = nxt, nxt + 1
        nxt += 1 + 2 * pack_pairs
    col128Base = -1
    if use_subtile and col128:
        col128Base = (nxt + 1) & ~1
        nxt = col128Base + 7
    numCvtVgprs = nxt if use_subtile else 4
    return (foldVoff, foldBlend, scalarAddrOff, scalarRingOff, col128Base, numCvtVgprs)


def new_layout(use_subtile, is_fold, scalar, pack_pairs, col128):
    """The refactored call site, reproduced exactly as it now stands in KWA."""
    base = 7 if use_subtile else 4
    scratch = EpilogueScratch(base)
    foldVoff = foldBlend = -1
    if is_fold:
        foldVoff = scratch.reserve("foldVoff", 1)
        foldBlend = scratch.reserve("foldBlend", 1)
    scalarAddrOff = scalarRingOff = -1
    if scalar:
        scalarAddrOff = scratch.reserve("scalarAddr", 1)
        scalarRingOff = scratch.reserve("scalarRing", 2 * pack_pairs)
    col128Base = -1
    if use_subtile and col128:
        col128Base = scratch.reserve("col128", 7, align=2)
    numCvtVgprs = scratch.total() if use_subtile else 4
    return (foldVoff, foldBlend, scalarAddrOff, scalarRingOff, col128Base, numCvtVgprs)


def _sweep():
    for use_subtile in (False, True):
        for is_fold in (False, True):
            for scalar in (False, True):
                for pack_pairs in (1, 2, 4, 8):
                    for col128 in (False, True):
                        yield (use_subtile, is_fold, scalar, pack_pairs, col128)


def test_epilogue_scratch_matches_legacy_cursor():
    for combo in _sweep():
        assert new_layout(*combo) == legacy_layout(*combo), \
            f"layout mismatch for (use_subtile,is_fold,scalar,pack_pairs,col128)={combo}: " \
            f"new={new_layout(*combo)} legacy={legacy_layout(*combo)}"


def test_reserve_rejects_overlap_and_bad_size():
    s = EpilogueScratch(7)
    s.reserve("a", 2)
    try:
        s.reserve("a", 1)          # duplicate name
        raise AssertionError("expected ValueError on duplicate window")
    except ValueError:
        pass
    try:
        s.reserve("b", 0)          # bad size
        raise AssertionError("expected ValueError on size 0")
    except ValueError:
        pass


def test_forward_only_windows_never_overlap():
    # Any sequence of reserves yields strictly non-overlapping, ordered windows.
    s = EpilogueScratch(7)
    s.reserve("x", 1)
    s.reserve("y", 3, align=2)
    s.reserve("z", 2)
    spans = []
    for name in ("staging", "x", "y", "z"):
        off = s.offset(name)
        size = {"staging": 7, "x": 1, "y": 3, "z": 2}[name]
        spans.append((off, off + size))
    spans.sort()
    for (a_lo, a_hi), (b_lo, b_hi) in zip(spans, spans[1:]):
        assert a_hi <= b_lo, f"overlap: {spans}"


if __name__ == "__main__":
    test_epilogue_scratch_matches_legacy_cursor()
    test_reserve_rejects_overlap_and_bad_size()
    test_forward_only_windows_never_overlap()
    n = sum(1 for _ in _sweep())
    print(f"OK: EpilogueScratch layout == legacy cursor across {n} parameter combinations")
