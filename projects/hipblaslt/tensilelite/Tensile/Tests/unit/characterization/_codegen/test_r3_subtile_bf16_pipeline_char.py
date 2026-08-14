################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""Characterization for the bf16 UseSubtileImpl store-transpose PIPELINE (gfx950, wave64).

Guards the depth-2 double-buffered ``ds_bpermute`` pipeline on the guard-free interior
store body: the paired store is split into an ISSUE phase (pack + ``ds_bpermute`` +
address) and a COMMIT phase (``s_waitcnt`` + ``v_permlane32_swap_b32`` + dwordx4 store),
and group N+1 is issued before group N is committed so N's LDS round-trip overlaps N+1's
issue.

Target: Tensile/Components/GlobalWriteBatch.py, the ranges this change adds:
  2706-2760  ``_buildSubtileInteriorStores`` driver — buffer rotation plus the nested
             ``commitPending`` (2725) / ``issuePaired`` (2731) that order issue before commit
  2932-2971  ``_emitPairedStoreIssue`` — pack + 4 ``ds_bpermute`` + address, no wait
  2972-2998  ``_emitPairedStoreCommit`` — wait leaving ``dscnt`` younger ds ops in flight,
             then the permlane swaps and the dwordx4 store

The load-bearing assertion is ``test_commit_wait_leaves_exactly_the_younger_groups_in_flight``:
a commit must wait to ``lgkmcnt(4 * younger_groups_still_in_flight)``. Getting that count too
HIGH is silent data corruption rather than a hang — the wave proceeds to permute and store a
buffer whose ``ds_bpermute`` has not landed, so D holds untransposed data. Counting the
in-flight groups from the instruction stream is what makes the assertion able to see that,
where "some kernel contains lgkmcnt(4)" could not.

Assertions are on opcodes, register names and instruction ORDER, never on emitted comment
prose, so rewording a comment cannot break CI.

Reuses the peel sweep's designed config (already registered in ``input_yaml_files.txt``)
because the pipeline lives inside the body that sweep emits; forking a near-duplicate config
would double emit cost for the same coverage.

CPU-only — no GPU, no compile, no hardware access.
"""

import os
import re
from typing import List, NamedTuple

import pytest

from config_harness import emit_kernels_from_config

pytestmark = pytest.mark.unit

_ARCH = "gfx950"

# 3 MFMA/WaveGroup shapes x StreamK{0,5} x PGR{1,2} = 12 fork permutations.
_LIMIT = 12

_CONFIG = os.path.join(
    os.path.dirname(__file__),
    "data",
    "test_data",
    "_designed",
    "gfx950",
    "subtile_bf16_peel.yaml",
)

# Opcode signature of the 16bit paired subtile store; selects the relevant kernels.
_PAIRED_STORE_OPS = (
    "v_permlane32_swap_b32",
    "ds_bpermute_b32",
)

# One transpose group = 4 in-place ds_bpermute (one per packed dword).
_BPERMUTE_PER_GROUP = 4

_BPERMUTE_RE = re.compile(r"\s*ds_bpermute_b32\b")
_WAIT_LGKM_RE = re.compile(r"\s*s_waitcnt\s+lgkmcnt\((\d+)\)")
# Stores through the D SRD only: the StreamK partials path also emits dwordx4, to SrdWS.
_STORE_D_RE = re.compile(r"\s*buffer_store_(\w+)\s+[^,]+,\s*[^,]+,\s*s\[sgprSrdD:")
# 64-bit advance of the D base address (deferred row increment / batch offset).
_SRD_D_ADVANCE_RE = re.compile(r"\s*s_addc?_u32\s+s\[sgprSrdD\+[01]\]")


class _Commit(NamedTuple):
    line: int
    lgkmcnt: int
    groups_in_flight: int  # this group + every younger group issued but not yet stored


class _Trace(NamedTuple):
    groups_issued: int
    commits: List[_Commit]
    # (line, what, groups_still_pending) for a buffer-reuse or SRD-advance drain breach.
    drain_breaches: List[tuple]
    end_pending: int

    @property
    def overlapped(self):
        return [c for c in self.commits if c.groups_in_flight >= 2]

    @property
    def drained(self):
        return [c for c in self.commits if c.groups_in_flight == 1]


# --- fixtures (emit once per module; the emit pipeline is the expensive part) ----------


@pytest.fixture(scope="module")
def emitted():
    """[(basename, src, err), ...] for the whole designed fork sweep."""
    return emit_kernels_from_config(_CONFIG, limit=_LIMIT, arch=_ARCH)


@pytest.fixture(scope="module")
def paired_kernels(emitted):
    """Every emitted kernel that took the 16bit subtile paired-store path."""
    hits = [(b, s) for (b, s, _e) in emitted if all(op in s for op in _PAIRED_STORE_OPS)]
    assert hits, (
        "no kernel took the 16bit subtile paired-store path -- is16bitSubtile never "
        f"engaged (config: {_CONFIG}); emitted: {[b for b, _s, _e in emitted]}"
    )
    return hits


@pytest.fixture(scope="module")
def traces(paired_kernels):
    """basename -> _Trace of the store pipeline, walked from the instruction stream."""
    return {base: _trace_store_pipeline(src) for base, src in paired_kernels}


# --- helpers --------------------------------------------------------------------------


def _trace_store_pipeline(src):
    """Walk the emitted stream and reconstruct the transpose pipeline's depth over time.

    A group becomes *pending* once its 4 ``ds_bpermute`` are issued and stops being pending
    when its dwordx4 store retires it, so ``pending`` at any point is the number of groups
    whose LDS round-trip is outstanding. That is the quantity a commit's ``lgkmcnt`` has to
    agree with, and the quantity that must be zero wherever buffer 0 is reused or the D SRD
    moves underneath a deferred store.
    """
    pending = 0
    bpermute_run = 0
    groups_issued = 0
    last_wait = None
    commits, drain_breaches = [], []

    for i, line in enumerate(src.splitlines()):
        if _BPERMUTE_RE.match(line):
            bpermute_run += 1
            if bpermute_run == _BPERMUTE_PER_GROUP:
                pending += 1
                groups_issued += 1
                bpermute_run = 0
            continue

        wait = _WAIT_LGKM_RE.match(line)
        if wait:
            last_wait = int(wait.group(1))
            continue

        store = _STORE_D_RE.match(line)
        if store:
            width = store.group(1)
            if width == "dwordx4" and pending:
                # A paired-store commit: retires the oldest outstanding group.
                commits.append(_Commit(line=i, lgkmcnt=last_wait, groups_in_flight=pending))
                pending -= 1
            elif pending:
                # Scalar/orphan stores reuse buffer 0, so nothing may be outstanding here.
                drain_breaches.append((i, f"buffer_store_{width}", pending))
            continue

        if _SRD_D_ADVANCE_RE.match(line) and pending:
            drain_breaches.append((i, "SrdD advance", pending))

    return _Trace(groups_issued, commits, drain_breaches, pending)


# --- tests ----------------------------------------------------------------------------


def test_pipeline_walk_is_not_vacuous(traces):
    """The walk found issued groups and committing stores in every paired kernel.

    Everything below quantifies over ``commits``; if the opcode matchers ever stop matching
    (an opcode rename, a change of store width or SRD) those assertions would pass over an
    empty list. This is the guard that makes them mean something.
    """
    for base, t in traces.items():
        assert t.groups_issued > 0, (
            f"{base}: no complete {_BPERMUTE_PER_GROUP}x ds_bpermute group was found -- the "
            "transpose-group matcher no longer matches the emitted stream"
        )
        assert t.commits, (
            f"{base}: {t.groups_issued} transpose groups were issued but no dwordx4 store "
            "through SrdD retired any of them -- the commit matcher is stale"
        )


def test_pipeline_engages_across_the_sweep(traces):
    """Issue-before-commit overlap is emitted, on more than one geometry.

    Overlap is a property of the tile geometry, not of the lever: a body whose N-group holds a
    single paired group drains at every commit and legitimately shows none, so this asserts a
    floor across the sweep rather than requiring every kernel to overlap. The exact split is
    pinned by the golden below.
    """
    overlapping = {b: len(t.overlapped) for b, t in traces.items() if t.overlapped}
    assert len(overlapping) >= 2, (
        "expected issue-before-commit overlap on >=2 kernels of the sweep, got "
        f"{len(overlapping)}: {overlapping}. Per-kernel commit counts: "
        f"{ {b: len(t.commits) for b, t in traces.items()} }. With no overlap anywhere the "
        "pipeline is emitting its ISSUE/COMMIT split but never hiding a round-trip."
    )


def test_commit_wait_leaves_exactly_the_younger_groups_in_flight(traces):
    """A commit waits for its own ds_bpermute and no further: lgkmcnt == 4 * younger groups.

    Over-waiting only costs the win (that is the un-pipelined ``lgkmcnt(0)``). UNDER-waiting is
    silent corruption: with lgkmcnt(4) and nothing younger in flight, the wave permutes and
    stores a buffer whose own ds_bpermute has not landed, writing untransposed D. The count is
    reconstructed from the stream, so the assertion tracks the real depth rather than a literal.
    """
    for base, t in traces.items():
        for c in t.commits:
            expected = _BPERMUTE_PER_GROUP * (c.groups_in_flight - 1)
            assert c.lgkmcnt == expected, (
                f"{base}: commit at line {c.line} waited lgkmcnt({c.lgkmcnt}) with "
                f"{c.groups_in_flight} transpose group(s) outstanding, so it must wait "
                f"lgkmcnt({expected}) to cover its own 4 ds_bpermute. "
                + (
                    "This wait does NOT cover this group's own transpose -- the store writes "
                    "untransposed data."
                    if c.lgkmcnt > expected
                    else "This wait is more conservative than the pipeline needs."
                )
            )


def test_pipeline_drains_before_srd_advance_and_buffer_reuse(traces):
    """No outstanding group may cross a D-SRD advance, a scalar store, or the body end.

    Deferred stores are addressed off the live ``SrdD``, so a group committed after the row
    increment would write the wrong row; scalar/orphan stores reuse buffer 0, so a group still
    outstanding there would have its packed dwords overwritten before the store reads them.
    Both are the ``commitPending(dscnt=0)`` drains in the driver.
    """
    for base, t in traces.items():
        assert not t.drain_breaches, (
            f"{base}: transpose group(s) still outstanding at a point that requires a drain: "
            f"{t.drain_breaches} (line, site, groups outstanding). The matching "
            "commitPending(dscnt=0) drain is missing."
        )
        assert t.end_pending == 0, (
            f"{base}: {t.end_pending} transpose group(s) issued but never stored at the end of "
            "the kernel -- the final pipeline drain is missing and those outputs are dropped."
        )


def test_r3_subtile_bf16_pipeline_gfx950_golden(traces, snapshot):
    """Order-invariant golden: pin the per-kernel overlapped/drained commit split.

    Deliberately not a {basename, err} digest: the sibling peel module already pins that for
    this same config, and re-pinning it here would catch nothing new. This pins the pipeline's
    own shape instead, so overlap silently collapsing to drain-only on some geometries -- which
    leaves every assertion above passing -- shows up as a golden diff. Counts are per-kernel
    totals, so they do not depend on emission order.
    """
    digest = sorted(
        (
            {
                "basename": base,
                "commits": len(t.commits),
                "overlapped": len(t.overlapped),
                "drained": len(t.drained),
            }
            for base, t in traces.items()
        ),
        key=lambda d: d["basename"],
    )
    assert digest == snapshot
