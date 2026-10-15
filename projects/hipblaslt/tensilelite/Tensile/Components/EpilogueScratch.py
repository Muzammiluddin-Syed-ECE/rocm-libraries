# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Named-window scratch allocator for the subtile store epilogue.

The subtile 16-bit store checks out one contiguous, aligned block of "cvt" VGPRs and
sub-divides it into per-feature windows (bf16-convert staging, ds_bpermute address,
the DPP-fold voffset/blend temps, the scalar/pair pack rings, ...).  Historically those
sub-windows were carved by a hand-advanced integer cursor (``nxt``) with each feature's
offset computed inline, and disjointness guaranteed only by a comment.

``EpilogueScratch`` replaces that cursor with a small forward-only allocator:

* every window is **named**, and the layout is **dumpable** -- disjointness is a
  property of the allocator, not a promise in a comment;
* two windows can never overlap (the cursor only moves forward);
* it produces the **identical** offsets/total the hand cursor did (see
  ``Tensile/Tests/unit/test_epilogue_scratch_layout.py``), so introducing it is a
  bit-exact refactor;
* it gives the DPP fold a first-class, asserted way to borrow provably-dead ValuC
  accumulator slots instead of a raw register-index aliasing (``borrow_dead_window``).

This owns only the *layout* arithmetic (offsets relative to the block base).  The actual
pool checkout and the ``base + offset`` absolute-VGPR math stay in the caller, so this
helper has no dependency on the register pool or rocisa and is trivially testable.
"""


class EpilogueScratch:
    """Forward-only named-window layout over a single checked-out VGPR block.

    Offsets are relative to the block base (the caller adds the absolute ``cvtVgpr``).
    The leading ``base`` registers are the always-present shared staging window
    (``+0..+base-1``): the bf16-convert temps, reused as pack/permute scratch.
    """

    def __init__(self, base):
        self._base = base
        self._cursor = base
        # name -> (offset, size). Insertion order preserved for dump().
        self._windows = {"staging": (0, base)}

    def reserve(self, name, size, align=1):
        """Reserve a ``size``-wide window, ``align``-aligned; return its offset.

        Matches the legacy cursor exactly: ``align`` rounds the start up to the next
        multiple of ``align`` (e.g. the Col128 quad used ``(nxt + 1) & ~1``).
        """
        if size <= 0:
            raise ValueError(f"EpilogueScratch.reserve({name!r}, {size}): size must be > 0")
        if name in self._windows:
            raise ValueError(f"EpilogueScratch.reserve: window {name!r} already reserved")
        off = (self._cursor + (align - 1)) & ~(align - 1)
        self._windows[name] = (off, size)
        self._cursor = off + size
        return off

    def borrow_dead_window(self, name, base_offset, size):
        """Register a window that lives OUTSIDE this block (e.g. provably-dead ValuC
        accumulator slots the fold reuses).  Records the borrow by name for the layout
        dump and overlap bookkeeping, but does NOT advance the cursor and does NOT grow
        the block -- the registers are owned elsewhere.  The caller is responsible for
        asserting the liveness precondition that makes the borrow safe.
        """
        if name in self._windows:
            raise ValueError(f"EpilogueScratch.borrow_dead_window: {name!r} already present")
        self._windows[name] = (base_offset, size)
        return base_offset

    def offset(self, name):
        """Offset of a reserved window relative to the block base, or -1 if absent."""
        w = self._windows.get(name)
        return w[0] if w is not None else -1

    def total(self):
        """Total block size to check out (== the legacy ``numCvtVgprs``)."""
        return self._cursor

    def dump(self):
        """Human-readable layout, ordered by offset -- for logging / review."""
        items = sorted(self._windows.items(), key=lambda kv: (kv[1][0], kv[0]))
        return " ".join(f"{n}@+{o}:{s}" for n, (o, s) in items)
