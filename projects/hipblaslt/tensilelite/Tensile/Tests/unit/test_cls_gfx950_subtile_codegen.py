#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
################################################################################
# Structural codegen tests for CompactLoopStore (CLS) on gfx950 -- SUBTILE path.
#
# They run entirely against the Python/rocisa codegen layer: the real store-D
# methods (mapAcctoArchRegs -> notLocalSplitUGlobalWriteIndices ->
# notLocalSplitUGlobalWrite) are invoked exactly as test_storeD_roundtrip.py
# does, but instead of assembling and running on a GPU we render the emitted
# Module to text (str(...)) and assert on the CLS structure. No GPU required.
#
# Why structural and not numerical: the non-subtile CLS coverage in-tree is
# itself structural / config-driven (the gfx1250 CLS YAMLs skip gfx950; the
# gfx950 store harness in test_storeD_roundtrip.py sets CompactLoopStore=False).
#
# Properties pinned, and why each one matters:
#   * compaction fired         -- a CLS body whose counter starts > 1 is the
#                                 whole point; iterCount == 1 everywhere means
#                                 the optimization silently did nothing.
#   * block-scoped brackets    -- one s_set_gpr_idx_on/off pair per contiguous
#                                 acc-read cluster. Strict alternation with an
#                                 unbalanced or nested bracket left index mode
#                                 on, which corrupts the SRC0 of every later
#                                 VALU instruction; one bracket per read undoes
#                                 the instruction-count win the loop buys.
#   * only acc-reads inside    -- while index mode is on EVERY VALU SRC0 is
#                                 M0-relative, so any other instruction that
#                                 drifts into the bracket is silently corrupted
#                                 once the loop compacts (M0 > 0).
#   * gpr_idx(SRC0) only       -- offsetting DST moves the store staging base
#                                 too, corrupting D at M0 > 0.
#   * M0 base 0 + fixed stride -- the loop reaches the later accumulator slices
#                                 only if M0 starts at 0 and steps uniformly.
#   * non-compacting edge      -- a tile whose outerTT1 does not divide
#                                 numBatches must still emit well-formed code.
#   * SRD advance self-contained -- CLS normally carries the next row increment
#                                 in s[stmp] across incrementToNextRow calls.
#                                 The subtile store/load bodies emitted between
#                                 two calls claim that same scratch as the wave64
#                                 exec mask, so on the subtile path the increment
#                                 must be computed immediately before the s_add
#                                 that consumes it. Getting this wrong advances
#                                 the SRD by a lane mask -> hipErrorIllegalAddress.
#
# Usage:
#   pytest test_cls_gfx950_subtile_codegen.py -v
################################################################################

import os
import re
import sys

import pytest

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TENSILE_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
sys.path.insert(0, TENSILE_ROOT)
sys.path.insert(0, SCRIPT_DIR)

GFX950_ISA = (9, 5, 0)
WAVESIZE_64 = 64

# Capability pairs that select the two CompactLoopStore acc-read mechanisms.
# Solution.py admits a CLS solution on either of them.
CAPS_INDEX_MODE = {"HasMovRelsD2B32": False, "HasVgprIndexMode": True}   # gfx950
CAPS_MOVRELS = {"HasMovRelsD2B32": True, "HasVgprIndexMode": False}      # gfx10+


def _gfx950_cls_supported():
    """True if rocisa exposes the CLS index-mode instructions and gfx950 asm.

    The gfx950 CLS acc-read uses s_set_gpr_idx_on/off (VGPR Index Mode). Those
    rocisa bindings (SSetGprIdxOn/Off) are only present in a rocisa built with
    the CLS codegen; a stale rocisa cannot even import KernelWriterModules.
    """
    try:
        from rocisa.instruction import SSetGprIdxOn, SSetGprIdxOff  # noqa: F401
    except ImportError:
        return False
    try:
        from gpu_test_helpers import init_rocisa
        init_rocisa(target="gfx950", wavesize=WAVESIZE_64)
        from rocisa import rocIsa
        caps = rocIsa.getInstance().getAsmCaps()
        # any well-known gfx9 asm cap proves the assembler initialised for gfx950
        return bool(caps)
    except (ImportError, RuntimeError, OSError):
        # no assembler / no rocisa build for gfx950: skip rather than error
        return False


pytestmark = pytest.mark.skipif(
    not _gfx950_cls_supported(),
    reason="rocisa lacks gfx950 CLS index-mode bindings / assembler",
)


@pytest.fixture(scope="module", autouse=True)
def _rocisa_once():
    from gpu_test_helpers import init_rocisa
    init_rocisa(target="gfx950", wavesize=WAVESIZE_64)


# ---------------------------------------------------------------------------
# Store-module builder: identical construction to test_storeD_roundtrip._run_storeD
# up to the point where the assembly text is produced, but CompactLoopStore is
# toggleable and we stop at str(module) (no assemble / no GPU run).
#
# NOTE: this reuses test_storeD_roundtrip's kernel/writer builders. That module
# is GPU-marked, but the four builders themselves are pure codegen setup and run
# without a device; importing it here only needs its module-level imports.
# ---------------------------------------------------------------------------

# Compacting subtile tiles: MIWaveTile[1] (== outerTT1, since VW1=1) > 1 and
# divides numBatches, so computeCLSLayout case (a) fires (iterCount = outerTT1).
COMPACTING_CONFIGS = [
    (128, 128, 64),   # MIWaveTile [4, 4] -> iterCount 4
    (256, 128, 64),   # MIWaveTile [8, 4] -> iterCount 4
]

# Non-compacting subtile tile: outerTT1 (=8) does NOT divide numBatches here, so
# every CLS body stays iterCount == 1 -- must still be well-formed.
NONCOMPACTING_CONFIG = (128, 256, 64)


def _build_subtile_store_asm(mt_a, mt_b, depth_u, cls, mi_wave_group=None,
                             use_bf16=False, mi_arch_vgpr=False):
    """Build the gfx950 subtile store-D module and return its assembly text.

    Returns (asm_text, kernel). asm_text concatenates the store-index module,
    the store-write module, and any deferred edge modules (the subtile edge
    store batches reached via PC-relative jumps), matching how the roundtrip
    harness stitches the final kernel.
    """
    import test_storeD_roundtrip as T
    from gpu_test_helpers import create_writer
    from Tensile.KernelWriterModules import mapAcctoArchRegs

    cfg = T.TileConfig(mt_a=mt_a, mt_b=mt_b, depth_u=depth_u)

    kernel = T._build_store_kernel(cfg, mi_wave_group=mi_wave_group, use_bf16=use_bf16)
    kernel["CompactLoopStore"] = cls
    kernel["UseSubtileImpl"] = True
    kernel["MIArchVgpr"] = mi_arch_vgpr
    assert tuple(kernel["ISA"]) == GFX950_ISA

    writer, _, _, _ = create_writer(cfg, mi_wave_group=mi_wave_group)
    sgprs = T._build_sgprs_for_test(writer)
    tileInfoD, _agpr_indices = T._allocate_d_tile(kernel, writer)

    kw = T._build_kwa(kernel, writer, use_bf16=use_bf16)
    kw.states.d.tileInfo = tileInfoD
    kw.states.subtileM32ValidBlocksSgpr = sgprs["subtileMValidBlocks"]
    kw.states.subtileN16ValidBlocksSgpr = sgprs["subtileNValidBlocks"]
    kw.sgprs["SubtileMGuard"] = sgprs["subtileMValidBlocks"]
    kw.sgprs["SubtileNGuard"] = sgprs["subtileNValidBlocks"]
    kw.states.subtileMBlockSize = 16

    kw.codes.accVgprRead = mapAcctoArchRegs(kernel, kw.states.asmCaps, kw.states.maxLimitAgprs, write=False)
    idx_mod = kw.notLocalSplitUGlobalWriteIndices(kernel)
    kw.states.c.startVgprValu = 0
    write_mod, _ = kw.notLocalSplitUGlobalWrite(kernel, tPA=None, tPB=None)

    parts = [str(idx_mod), str(write_mod)]
    deferred = getattr(kw.states, "deferredEdgeModules", None)
    if deferred:
        parts.extend(str(m) for m in deferred)
    return "\n".join(parts), kernel


# ---- small assembly-string measurement helpers ----

_RE_IDX_ON = re.compile(r"s_set_gpr_idx_on")
_RE_IDX_OFF = re.compile(r"s_set_gpr_idx_off")
# The acc-read is v_accvgpr_read_b32 on the AGPR (MIArchVgpr:false) path and a
# plain v_mov_b32 on the MIArchVgpr:true path; both are tagged M0-indexed under
# CLS index mode, which is what distinguishes them from ordinary moves.
_RE_ACC_READ = re.compile(r"v_accvgpr_read|v_mov_b32.*M0-indexed")
_RE_CLS_COUNTER_INIT = re.compile(r"s\[sgprCLSLoopCounter\],\s*(0x[0-9a-fA-F]+)")
_RE_CLS_LABEL = re.compile(r"^label_CLS\w*:", re.MULTILINE)
_RE_CLS_BACKEDGE = re.compile(r"s_cbranch_scc0\s+label_CLS")
_RE_M0BASE_INIT = re.compile(r"s\[sgprCLSm0Base\],\s*0x0")
_RE_M0BASE_STEP = re.compile(r"s_add_u32\s+s\[sgprCLSm0Base\],\s*s\[sgprCLSm0Base\],\s*(\d+)")


# An incToNextRow SRD advance and the two instruction forms that may legally
# produce the value it consumes: a stride compute (s_mul/s_lshl off a Stride
# sgpr) or an explicit zero (the "row 0, no advance" seed).
_RE_SRD_ADVANCE = re.compile(
    r"^\s*s_add_u32 s\[sgprSrd([CD])\+0\], s\[sgprSrd\1\+0\], (s\d+)\b.*incToNextRow")
_RE_STRIDE_COMPUTE = re.compile(r"^\s*s_(?:mul_i32|lshl_b32) s\d+, s\[sgprStride")
_RE_ZERO_WRITE = re.compile(r"^\s*s_mov_b32 s\d+, 0\s*(?://.*)?$")


def _counts(asm):
    on = len(_RE_IDX_ON.findall(asm))
    off = len(_RE_IDX_OFF.findall(asm))
    reads = len(_RE_ACC_READ.findall(asm))
    return on, off, reads


def _writes_sgpr(line, reg):
    """True if `line` writes scalar register `reg` (single dest or b64 pair).

    The b64 form matters: the subtile exec mask is written as a wave64 pair
    (s_mov_b64/s_lshr_b64 s[N:N+1]), which clobbers both halves.
    """
    n = int(reg[1:])
    code = line.split("//")[0]
    if re.match(rf"^\s*s_\w+ {reg}\s*,", code):
        return True
    pair = re.match(r"^\s*s_\w+ s\[(\d+):(\d+)\]\s*,", code)
    return bool(pair and int(pair.group(1)) <= n <= int(pair.group(2)))


def _srd_advance_producers(asm):
    """Pair every incToNextRow SRD advance with the value it actually adds.

    Returns [(lineno, reg, producer_line)], where producer_line is the nearest
    preceding instruction that writes the consumed register -- i.e. what the
    hardware really adds to the SRD, not what the emitter intended.
    """
    lines = asm.splitlines()
    sites = []
    for i, raw in enumerate(lines):
        m = _RE_SRD_ADVANCE.match(raw)
        if not m:
            continue
        reg = m.group(2)
        producer = next((lines[j] for j in range(i - 1, -1, -1)
                         if _writes_sgpr(lines[j], reg)), None)
        sites.append((i + 1, reg, producer))
    return sites


def _counter_inits(asm):
    return [int(v, 16) for v in _RE_CLS_COUNTER_INIT.findall(asm)]


def _scan_brackets(asm):
    """Walk the asm in emission order and return the contents of each bracket.

    Raises AssertionError on any nesting/ordering defect: an `on` while already
    inside a bracket, an `off` outside one, or a bracket still open at the end.
    Returns a list (one entry per bracket) of the source lines between the `on`
    and its matching `off`.
    """
    depth = 0
    brackets = []
    current = None
    for lineno, raw in enumerate(asm.splitlines(), start=1):
        line = raw.split("//")[0].strip()
        if not line or line.startswith("/*"):
            continue
        if line.startswith("s_set_gpr_idx_on"):
            assert depth == 0, f"nested s_set_gpr_idx_on at line {lineno}: {raw.strip()}"
            depth = 1
            current = []
        elif line.startswith("s_set_gpr_idx_off"):
            assert depth == 1, f"s_set_gpr_idx_off outside a bracket at line {lineno}"
            depth = 0
            brackets.append(current)
            current = None
        elif current is not None:
            current.append(raw)
    assert depth == 0, "index mode left on: a bracket was never closed"
    return brackets


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestGfx950SubtileCLSCodegen:
    """Structural CLS codegen assertions for the gfx950 subtile store path."""

    # -- baseline contrast: CLS-off emits none of the CLS scaffolding --

    @pytest.mark.parametrize("use_bf16", [False, True], ids=["f32", "bf16"])
    def test_cls_off_has_no_index_mode_or_loop(self, use_bf16):
        """CLS-OFF subtile store: zero index-mode brackets, no CLS loop/counter."""
        asm, _ = _build_subtile_store_asm(*COMPACTING_CONFIGS[0], cls=False,
                                          use_bf16=use_bf16)
        on, off, _ = _counts(asm)
        assert on == 0 and off == 0, "CLS-off must not emit s_set_gpr_idx brackets"
        assert "sgprCLSLoopCounter" not in asm
        assert _RE_CLS_LABEL.search(asm) is None

    # -- compaction fired --

    @pytest.mark.parametrize("mt_a,mt_b,depth_u", COMPACTING_CONFIGS,
                             ids=[f"{a}x{b}" for a, b, _ in COMPACTING_CONFIGS])
    def test_cls_on_compacts(self, mt_a, mt_b, depth_u):
        """A compacting subtile tile emits a CLS store loop with iterCount > 1."""
        asm, kernel = _build_subtile_store_asm(mt_a, mt_b, depth_u, cls=True)
        inits = _counter_inits(asm)
        assert inits, "expected at least one CLSLoopCounter initialisation"
        assert max(inits) > 1, (
            f"expected a compacting CLS body (iterCount > 1); got inits={inits}. "
            f"MIWaveTile={kernel['MIWaveTile']}")
        # the compacting body must be a real countdown loop with a back-edge
        assert _RE_CLS_LABEL.search(asm) is not None, "missing label_CLS loop label"
        assert _RE_CLS_BACKEDGE.search(asm) is not None, "missing s_cbranch_scc0 label_CLS back-edge"

    # -- block-scoped brackets: strict alternation, nothing but acc-reads inside --

    @pytest.mark.parametrize("mi_arch_vgpr", [False, True], ids=["agpr", "archvgpr"])
    @pytest.mark.parametrize("use_bf16", [False, True], ids=["f32", "bf16"])
    @pytest.mark.parametrize("mt_a,mt_b,depth_u", COMPACTING_CONFIGS,
                             ids=[f"{a}x{b}" for a, b, _ in COMPACTING_CONFIGS])
    def test_cls_brackets_are_block_scoped_clusters(self, mt_a, mt_b, depth_u,
                                                    use_bf16, mi_arch_vgpr):
        """Brackets strictly alternate on/off and contain only acc-reads.

        _scan_brackets fails on a dangling, doubled or mis-nested bracket (which
        would leave index mode on and M0-offset the SRC0 of every later VALU).
        On top of that: every acc-read must be inside a bracket (otherwise it
        reads the wrong slice once the loop compacts), and no bracket may hold a
        single read (that is the per-read bracketing this change replaced, which
        costs two SALU per read and cancels the compaction win).

        Both read forms are covered: v_accvgpr_read_b32 on the AGPR path and a
        plain v_mov_b32 when MIArchVgpr keeps the accumulators in arch vgprs.
        """
        asm, _ = _build_subtile_store_asm(mt_a, mt_b, depth_u, cls=True,
                                          use_bf16=use_bf16,
                                          mi_arch_vgpr=mi_arch_vgpr)
        on, off, reads = _counts(asm)
        assert reads > 0, "expected M0-indexed acc-reads in the store body"
        mnemonic = "v_mov_b32" if mi_arch_vgpr else "v_accvgpr_read_b32"
        assert mnemonic in asm, f"expected the {mnemonic} acc-read form"

        brackets = _scan_brackets(asm)
        assert len(brackets) == on == off, (
            f"bracket scan disagrees with raw counts: {len(brackets)} vs on={on} off={off}")
        assert brackets, "expected at least one index-mode bracket"

        inside = [ln for b in brackets for ln in b]
        for ln in inside:
            assert _RE_ACC_READ.search(ln), (
                f"non-acc-read inside an index-mode bracket (its SRC0 would be "
                f"M0-relative): {ln.strip()}")
        assert len(inside) == reads, (
            f"{reads - len(inside)} acc-read(s) emitted outside a bracket")
        for i, b in enumerate(brackets):
            assert len(b) > 1, (
                f"bracket {i} wraps a single read -- brackets must be per-cluster, "
                f"not per-read")

    # -- SRC0-only operand --

    @pytest.mark.parametrize("mt_a,mt_b,depth_u", COMPACTING_CONFIGS,
                             ids=[f"{a}x{b}" for a, b, _ in COMPACTING_CONFIGS])
    def test_cls_index_mode_is_src0_only(self, mt_a, mt_b, depth_u):
        """Every s_set_gpr_idx_on must be gpr_idx(SRC0) only (never DST/SRC0,DST).

        Offsetting the destination store-staging base by M0 corrupts D once the
        loop compacts (M0 > 0).
        """
        asm, _ = _build_subtile_store_asm(mt_a, mt_b, depth_u, cls=True)
        on_lines = [ln for ln in asm.splitlines() if "s_set_gpr_idx_on" in ln]
        assert on_lines, "expected s_set_gpr_idx_on lines"
        for ln in on_lines:
            assert "gpr_idx(SRC0)" in ln, f"expected SRC0-only operand: {ln.strip()}"
            assert "DST" not in ln, f"index mode must not offset DST: {ln.strip()}"

    # -- M0 base init + fixed stride --

    @pytest.mark.parametrize("mt_a,mt_b,depth_u", COMPACTING_CONFIGS,
                             ids=[f"{a}x{b}" for a, b, _ in COMPACTING_CONFIGS])
    def test_cls_m0_base_init_and_stride(self, mt_a, mt_b, depth_u):
        """CLSm0Base initialised to 0 and advanced by a fixed positive stride."""
        asm, _ = _build_subtile_store_asm(mt_a, mt_b, depth_u, cls=True)
        assert _RE_M0BASE_INIT.search(asm), "expected s[sgprCLSm0Base] init to 0x0"
        steps = [int(s) for s in _RE_M0BASE_STEP.findall(asm)]
        assert steps, "expected an s_add_u32 CLSm0Base stride per iteration"
        assert all(s > 0 for s in steps), f"CLS M0 step must be positive: {steps}"

    # -- non-compacting subtile edge stays well-formed --

    def test_cls_noncompacting_edge_wellformed(self):
        """A tile where outerTT1 does not divide numBatches: iterCount stays 1
        everywhere, but the emitted CLS scaffolding is still well-formed
        (strictly alternating brackets, SRC0-only, label_CLS with a back-edge)."""
        asm, kernel = _build_subtile_store_asm(*NONCOMPACTING_CONFIG, cls=True)
        inits = _counter_inits(asm)
        assert inits, "expected CLSLoopCounter initialisations even when not compacting"
        assert max(inits) == 1, (
            f"expected no compaction for this tile; got inits={inits}, "
            f"MIWaveTile={kernel['MIWaveTile']}")
        brackets = _scan_brackets(asm)
        assert brackets, "expected at least one index-mode bracket"
        for ln in (l for l in asm.splitlines() if "s_set_gpr_idx_on" in l):
            assert "gpr_idx(SRC0)" in ln and "DST" not in ln
        assert _RE_CLS_LABEL.search(asm) is not None
        assert _RE_CLS_BACKEDGE.search(asm) is not None


class TestAccVgprReadMechanismSelection:
    """mapAcctoArchRegs picks the acc-read mechanism from the same capabilities
    Solution.py gates CompactLoopStore on, and never bakes a bracket into the
    shared read Module (it would be lost when the Module is sliced downstream)."""

    MAX_AGPRS = 256  # gfx950 accvgpr file size, as used by the store harness

    @classmethod
    def _read_module(cls, clsOn, caps, mi_arch_vgpr):
        import test_storeD_roundtrip as T
        from Tensile.KernelWriterModules import mapAcctoArchRegs

        cfg = T.TileConfig(mt_a=128, mt_b=128, depth_u=64)
        kernel = T._build_store_kernel(cfg)
        kernel["CompactLoopStore"] = clsOn
        kernel["MIArchVgpr"] = mi_arch_vgpr
        return mapAcctoArchRegs(kernel, caps, cls.MAX_AGPRS, write=False), kernel

    @pytest.mark.parametrize("mi_arch_vgpr", [False, True], ids=["agpr", "archvgpr"])
    def test_index_mode_caps_emit_bare_reads(self, mi_arch_vgpr):
        """Index-mode arch: bare reads, no per-read bracket, module tagged."""
        from Tensile.KernelWriterModules import CLS_M0_RELATIVE_READS
        mod, _ = self._read_module(True, CAPS_INDEX_MODE, mi_arch_vgpr)
        text = str(mod)
        assert "s_set_gpr_idx" not in text, (
            "mapAcctoArchRegs must emit bare reads; the bracket belongs at the "
            "consumer, which is the only place a closing _off survives slicing")
        assert "v_movrelsd_2_b32" not in text
        assert "(src M0-indexed)" in text
        assert mod.name == CLS_M0_RELATIVE_READS, (
            "the read Module must advertise that its items are only correct "
            "inside an index-mode bracket")

    @pytest.mark.parametrize("mi_arch_vgpr", [False, True], ids=["agpr", "archvgpr"])
    def test_movrels_caps_never_emit_index_mode(self, mi_arch_vgpr):
        """An arch with v_movrelsd_2_b32 must not be handed index-mode brackets.

        Solution.py admits CLS on `HasMovRelsD2B32 or HasVgprIndexMode`, so this
        arch passes the solution gate; it must reach the movrels path, not an
        instruction its assembler does not have.
        """
        from Tensile.KernelWriterModules import CLS_M0_RELATIVE_READS
        mod, _ = self._read_module(True, CAPS_MOVRELS, mi_arch_vgpr)
        text = str(mod)
        assert "s_set_gpr_idx" not in text
        assert mod.name != CLS_M0_RELATIVE_READS
        if mi_arch_vgpr:
            # only the MIArchVgpr path has a movrels form; the AGPR path keeps
            # plain accvgpr reads and simply does not compact.
            assert "v_movrelsd_2_b32" in text

    def test_cls_off_reads_are_plain(self):
        """CLS off: no index mode, no movrels, no M0-indexed tagging."""
        mod, _ = self._read_module(False, CAPS_INDEX_MODE, False)
        text = str(mod)
        assert "s_set_gpr_idx" not in text
        assert "v_movrelsd_2_b32" not in text
        assert "M0-indexed" not in text


class TestClsIdxClusterHelper:
    """clsWrapIdxCluster is the only supported way to open a bracket, so the
    emptiness rule and the 'only acc-reads inside' rule live in it."""

    @staticmethod
    def _kernel(cls=True):
        return {"CompactLoopStore": cls}

    @staticmethod
    def _read():
        from rocisa.container import accvgpr, vgpr
        from rocisa.instruction import VAccvgprReadB32
        return VAccvgprReadB32(dst=vgpr(0), src=accvgpr(0), comment="acc read")

    def test_empty_cluster_gets_no_bracket(self):
        """A batch can pop zero reads (regsPerScalar is an integer division that
        can be 0), and an empty bracket must not be emitted for it."""
        from rocisa.code import Module
        from Tensile.KernelWriterModules import clsWrapIdxCluster
        cluster = Module("AccVgprReadCluster")
        out = clsWrapIdxCluster(self._kernel(), CAPS_INDEX_MODE, cluster)
        assert out is cluster
        assert "s_set_gpr_idx" not in str(out)

    def test_wraps_cluster_once(self):
        from rocisa.code import Module
        from Tensile.KernelWriterModules import clsWrapIdxCluster
        cluster = Module("AccVgprReadCluster")
        for _ in range(3):
            cluster.add(self._read())
        text = str(clsWrapIdxCluster(self._kernel(), CAPS_INDEX_MODE, cluster))
        assert text.count("s_set_gpr_idx_on") == 1
        assert text.count("s_set_gpr_idx_off") == 1
        assert text.index("s_set_gpr_idx_on") < text.index("v_accvgpr_read") \
               < text.index("s_set_gpr_idx_off")

    @pytest.mark.parametrize("kernel,caps", [
        ({"CompactLoopStore": False}, CAPS_INDEX_MODE),
        ({"CompactLoopStore": True}, CAPS_MOVRELS),
    ], ids=["cls_off", "movrels_arch"])
    def test_no_bracket_when_mechanism_not_selected(self, kernel, caps):
        from rocisa.code import Module
        from Tensile.KernelWriterModules import clsWrapIdxCluster
        cluster = Module("AccVgprReadCluster")
        cluster.add(self._read())
        out = clsWrapIdxCluster(kernel, caps, cluster)
        assert out is cluster
        assert "s_set_gpr_idx" not in str(out)

    def test_rejects_non_read_in_cluster(self):
        """Index mode M0-offsets the SRC0 of every VALU instruction in the
        bracket, so anything that is not an acc-read must fail at generation
        time rather than silently corrupt at M0 > 0."""
        from rocisa.code import Module
        from rocisa.container import vgpr, sgpr
        from rocisa.instruction import VMulF32
        from Tensile.KernelWriterModules import clsWrapIdxCluster
        cluster = Module("AccVgprReadCluster")
        cluster.add(self._read())
        cluster.add(VMulF32(dst=vgpr(0), src0=sgpr("Alpha"), src1=vgpr(0)))
        with pytest.raises(AssertionError, match="only acc-reads"):
            clsWrapIdxCluster(self._kernel(), CAPS_INDEX_MODE, cluster)


class TestBareAccVgprReadConsumers:
    """A consumer must bracket its reads IFF it runs inside the CLS countdown
    loop and so pops only a truncated prefix of the read list. StreamK, GSU and
    LSU pop the full list linearly outside the loop; their reads must stay bare,
    because a bracket there is a no-op that makes an otherwise M0-immune read
    depend on M0 (which on GFX9 is also the LDS base/limit register)."""

    # (module, class, method) of every accVgprRead consumer that runs outside
    # the CLS loop. Kept as an explicit list so adding a store path forces a
    # decision about which side of the rule it falls on.
    BARE_CONSUMERS = [
        ("Tensile.Components.StreamK", "StreamK", "partialsWriteBatch"),
        ("Tensile.Components.StreamK", "StreamK", "fixupBatch"),
        ("Tensile.Components.GSU", "GSUOn", "partialWriteBatch"),
        ("Tensile.Components.GSU", "GSUOn", "lastGsuWgReduction"),
        ("Tensile.Components.LSU", "LSUOn", "writeReadReduction"),
    ]

    @staticmethod
    def _source_of(module_name, class_name, method_name):
        import importlib
        import inspect
        mod = importlib.import_module(module_name)
        return inspect.getsource(getattr(getattr(mod, class_name), method_name))

    @pytest.mark.parametrize("module_name,class_name,method_name", BARE_CONSUMERS,
                             ids=[f"{c}.{m}" for _, c, m in BARE_CONSUMERS])
    def test_bare_consumers_emit_no_bracket(self, module_name, class_name, method_name):
        src = self._source_of(module_name, class_name, method_name)
        for token in ("clsWrapIdxCluster", "clsIdxModeOn", "clsIdxModeOff",
                      "SSetGprIdxOn", "SSetGprIdxOff"):
            assert token not in src, (
                f"{class_name}.{method_name} emits {token}: it consumes the "
                f"acc-read list in full outside the CLS loop, so its reads are "
                f"already at literal indices and must stay M0-immune")

    def test_in_loop_consumer_uses_the_shared_helper(self):
        """GlobalWriteBatch is the one consumer inside the CLS loop, and it must
        get its bracket from the shared helper (which owns the emptiness rule and
        the 'only acc-reads inside' assertion) rather than open-coding it."""
        src = self._source_of("Tensile.Components.GlobalWriteBatch",
                              "GlobalWriteBatchWriter", "_prolog")
        assert "clsWrapIdxCluster" in src
        assert "SSetGprIdxOn" not in src


class TestComputeCLSLayoutSubtile:
    """Pure-function tests of the subtile CLS layout math (no rocisa needed at
    call time). computeCLSLayout is the single source of truth for iterCount;
    these pin the subtile case (a) compaction predicate."""

    @staticmethod
    def _subtile_kernel(mi_wave_tile):
        # Minimal kernel dict for computeCLSLayout case (a): VW1 == 1, so
        # outerTT1 == MIWaveTile[1]. 16x16 MI, wave64, no SourceSwap/StreamK.
        return {
            "EnableMatrixInstruction": True,
            "VectorWidthA": 1,
            "VectorWidthB": 1,
            "MIWaveTile": list(mi_wave_tile),
            "MatrixInstM": 16,
            "MatrixInstN": 16,
            "MatrixInstBM": 1,
            "MatrixInstBN": 1,
            "WavefrontSize": WAVESIZE_64,
            "NumElementsPerBatchStore": 8,
            "SourceSwap": False,
            "StoreRemapVectorWidth": 0,
            "StreamK": 0,
        }

    def test_case_a_compacts_when_outertt1_divides_numbatches(self):
        from Tensile.Components.GlobalWriteBatch import GlobalWriteBatchWriter
        kernel = self._subtile_kernel([4, 4])  # outerTT1 = 4
        # numBatches divisible by outerTT1 -> compaction, iterCount == outerTT1.
        bpb, iterCount, m0Step = GlobalWriteBatchWriter.computeCLSLayout(kernel, numBatches=8)
        assert iterCount == 4, (bpb, iterCount, m0Step)
        assert bpb == 2
        assert m0Step > 0

    def test_case_a_no_compaction_when_not_divisible(self):
        from Tensile.Components.GlobalWriteBatch import GlobalWriteBatchWriter
        kernel = self._subtile_kernel([4, 8])  # outerTT1 = 8
        # numBatches not divisible by outerTT1 -> stays a single trip.
        bpb, iterCount, _ = GlobalWriteBatchWriter.computeCLSLayout(kernel, numBatches=12)
        assert iterCount == 1, (bpb, iterCount)
        assert bpb == 12

    def test_no_compaction_when_outertt1_is_one(self):
        from Tensile.Components.GlobalWriteBatch import GlobalWriteBatchWriter
        kernel = self._subtile_kernel([4, 1])  # outerTT1 = 1, VW1 == 1, !SourceSwap
        # None of the compacting cases apply -> iterCount 1.
        _, iterCount, _ = GlobalWriteBatchWriter.computeCLSLayout(kernel, numBatches=8)
        assert iterCount == 1

    def test_streamk_forces_single_iteration(self):
        from Tensile.Components.GlobalWriteBatch import GlobalWriteBatchWriter
        kernel = self._subtile_kernel([4, 4])  # would otherwise compact
        kernel["StreamK"] = 3
        # The StreamK store path runs outside the CLS loop, so it is not covered
        # by the countdown loop and must not be compacted.
        bpb, iterCount, _ = GlobalWriteBatchWriter.computeCLSLayout(kernel, numBatches=8)
        assert iterCount == 1 and bpb == 8

    @pytest.mark.parametrize("mi_wave_tile", [[1, 1], [2, 4], [4, 4], [8, 4], [4, 8]])
    @pytest.mark.parametrize("source_swap", [False, True])
    @pytest.mark.parametrize("num_batches", [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 16])
    def test_layout_always_covers_every_batch(self, mi_wave_tile, source_swap,
                                              num_batches):
        """batchesPerCLSBody * iterCount must equal numBatches, for every arm.

        notLocalSplitUGlobalWrite emits only batchesPerCLSBody batches and the
        CLS countdown re-executes exactly that body, so any layout whose
        coverage falls short of numBatches drops whole batches of stores from
        the kernel -- with no diagnostic, just missing output.
        """
        from Tensile.Components.GlobalWriteBatch import GlobalWriteBatchWriter
        kernel = self._subtile_kernel(mi_wave_tile)
        kernel["SourceSwap"] = source_swap
        bpb, iterCount, _ = GlobalWriteBatchWriter.computeCLSLayout(kernel, num_batches)
        assert bpb * iterCount == num_batches, (bpb, iterCount, num_batches)


class TestComputeCLSLayoutSourceSwap:
    """Pure-function tests of the SourceSwap arm, case (c) of computeCLSLayout.

    Case (c) is selected by outerTT1 == 1, VW1 == 1 and SourceSwap. It is
    reachable from in-tree configs: Tests/common/gemm/gfx12/bf16_CLS_gfx1250.yaml
    forks CompactLoopStore=True x SourceSwap=[True,False] over MatrixInstruction
    rows whose MIWaveTile[1] is 1, and VectorWidthB resolves to 1 there
    (Solution.py halves the candidate width until MIWaveTile[1] % vw == 0).

    With outerTT1 == VW1 == 1 the SourceSwap branch of accToArchMapper collapses
    to
        dst = vw0 + VW0*(bIdx0 + BM*(wgIdx0 + outerTT0*(tIdx + OPM*bIdx1)))
        src = tIdx + OPM*(bIdx0 + BM*(bIdx1 + BN*(vw0 + VW0*wgIdx0)))
    so with matrixInstBN == 1 the outermost (slowest) dst dim is tIdx, of extent
    OutputsPerMFMA1B and src stride 1. A CLS body is a prefix of the element
    list re-executed with M0 shifted, so tIdx is the only dim the loop can
    iterate, the iteration extent is OutputsPerMFMA1B, and m0Step is 1.
    """

    @staticmethod
    def _kernel(mi_wave_tile, vwa=1, nepbs=0, wave=WAVESIZE_64, mi=(16, 16),
                bm=1, bn=1):
        return {
            "EnableMatrixInstruction": True,
            "VectorWidthA": vwa,
            "VectorWidthB": 1,
            "MIWaveTile": list(mi_wave_tile),
            "MatrixInstM": mi[0],
            "MatrixInstN": mi[1],
            "MatrixInstBM": bm,
            "MatrixInstBN": bn,
            "WavefrontSize": wave,
            "NumElementsPerBatchStore": nepbs,
            "SourceSwap": True,
            "StoreRemapVectorWidth": 0,
            "StreamK": 0,
        }

    def test_default_num_elements_per_batch_store_does_not_raise(self):
        """NumElementsPerBatchStore == 0 is the default and must be harmless.

        The arm used to guard on `NEPBS % inner_dims == 0`, which is true for
        NEPBS == 0, and then divided numBatches by NEPBS -- so every CLS +
        SourceSwap solution that left NumElementsPerBatchStore at its default
        died in codegen with ZeroDivisionError.
        """
        from Tensile.Components.GlobalWriteBatch import GlobalWriteBatchWriter
        kernel = self._kernel([4, 1], nepbs=0)
        for numBatches in (1, 3, 4, 8, 12):
            bpb, iterCount, m0Step = GlobalWriteBatchWriter.computeCLSLayout(
                kernel, numBatches)
            assert bpb * iterCount == numBatches
            assert m0Step >= 1

    @pytest.mark.parametrize("nepbs", [0, 1, 2, 4, 8, 12])
    def test_layout_does_not_depend_on_num_elements_per_batch_store(self, nepbs):
        """NumElementsPerBatchStore caps the batch size upstream (it is one of
        the clamps that produce numBatches in refineOccupancy); it is not an
        extent of the accumulator layout, so it must not select the iteration
        dim. The old arm divided a *batch* count by this *element* count."""
        from Tensile.Components.GlobalWriteBatch import GlobalWriteBatchWriter
        base = self._kernel([4, 1], nepbs=0)
        variant = self._kernel([4, 1], nepbs=nepbs)
        for numBatches in (1, 2, 4, 8, 12, 16):
            assert (GlobalWriteBatchWriter.computeCLSLayout(base, numBatches)
                    == GlobalWriteBatchWriter.computeCLSLayout(variant, numBatches))

    def test_iterates_tidx_with_unit_m0_step(self):
        """16x16 MI on wave64 gives OutputsPerMFMA1B == 4, so a numBatches that
        is a multiple of 4 compacts 4x with a unit M0 step."""
        from Tensile.Components.GlobalWriteBatch import GlobalWriteBatchWriter
        kernel = self._kernel([4, 1])
        assert GlobalWriteBatchWriter.computeCLSLayout(kernel, 8) == (2, 4, 1)
        assert GlobalWriteBatchWriter.computeCLSLayout(kernel, 4) == (1, 4, 1)

    def test_no_compaction_when_tidx_extent_does_not_divide_numbatches(self):
        from Tensile.Components.GlobalWriteBatch import GlobalWriteBatchWriter
        kernel = self._kernel([4, 1])            # tIdx extent 4
        bpb, iterCount, _ = GlobalWriteBatchWriter.computeCLSLayout(kernel, 10)
        assert (bpb, iterCount) == (10, 1)

    def test_matrix_inst_bn_above_one_stays_single_iteration(self):
        """matrixInstBN > 1 puts bIdx1 outside tIdx in dst, so a tIdx-sized body
        is no longer a prefix of the element list and must not compact."""
        from Tensile.Components.GlobalWriteBatch import GlobalWriteBatchWriter
        kernel = self._kernel([4, 1], bn=2)
        _, iterCount, _ = GlobalWriteBatchWriter.computeCLSLayout(kernel, 8)
        assert iterCount == 1

    @pytest.mark.parametrize("mi_wave_tile,vwa,wave",
                             [([1, 1], 1, WAVESIZE_64),
                              ([2, 1], 2, WAVESIZE_64),
                              ([4, 1], 1, WAVESIZE_64),
                              ([8, 1], 4, WAVESIZE_64),
                              ([1, 1], 1, 32),
                              ([2, 1], 2, 32),
                              ([4, 1], 4, 32)])
    @pytest.mark.parametrize("num_batches", [1, 2, 3, 4, 5, 6, 8, 9, 12, 16, 24, 32])
    def test_m0_step_reaches_the_slice_the_loop_claims(self, mi_wave_tile, vwa,
                                                      wave, num_batches):
        """The M0 step must be exactly the acc-src delta between consecutive
        bodies, checked against the real accToArchMapper.

        This is the property that makes `s_set_gpr_idx_on m0` land on iteration
        j's accumulator slice: for every read position p in the body,
        arch2acc[p + j*readsPerBody] == arch2acc[p] + j*m0Step. Pinning it
        against the mapping (rather than against a number) is what catches an
        m0Step derived from an unrelated quantity.
        """
        from Tensile.Components.GlobalWriteBatch import GlobalWriteBatchWriter
        from Tensile.KernelWriterModules import accToArchMapper, getAccToArchLen

        kernel = self._kernel(mi_wave_tile, vwa=vwa, wave=wave)
        bpb, iterCount, m0Step = GlobalWriteBatchWriter.computeCLSLayout(
            kernel, num_batches)
        assert bpb * iterCount == num_batches
        if iterCount == 1:
            return
        _, arch2acc = accToArchMapper(kernel)
        accLen = getAccToArchLen(kernel)
        assert accLen % iterCount == 0, (accLen, iterCount)
        readsPerBody = accLen // iterCount
        for p in range(readsPerBody):
            for it in range(1, iterCount):
                assert arch2acc[p + it * readsPerBody] == arch2acc[p] + it * m0Step, (
                    f"m0Step {m0Step} does not reach slice {it} from read {p}: "
                    f"src({p + it * readsPerBody})={arch2acc[p + it * readsPerBody]} "
                    f"vs src({p})+{it}*{m0Step}={arch2acc[p] + it * m0Step}")


class TestSubtileSrdAdvanceIsSelfContained:
    """The store/load SRD must be advanced by a stride, never by leftover scratch.

    Under CompactLoopStore incrementToNextRow normally software-pipelines the
    row increment: call N leaves the increment call N+1 needs in s[stmp]. On the
    subtile path that carrier does not survive -- the paired/scalar store bodies
    emitted between two calls reuse s[stmp] (and s[stmp+1]) as the wave64 exec
    mask and as address scratch. The subtile sites therefore have to compute
    their increment immediately before the s_add that consumes it.

    A regression here is not a codegen-quality nit: the SRD gets advanced by a
    lane mask, the address walks off the buffer, and the kernel dies with
    hipErrorIllegalAddress once the CLS loop compacts.
    """

    @pytest.mark.parametrize("cls", [False, True], ids=["cls_off", "cls_on"])
    @pytest.mark.parametrize("use_bf16", [False, True], ids=["f32", "bf16"])
    @pytest.mark.parametrize("mt_a,mt_b,depth_u", COMPACTING_CONFIGS,
                             ids=[f"{a}x{b}" for a, b, _ in COMPACTING_CONFIGS])
    def test_srd_advance_consumes_a_stride_not_scratch(self, mt_a, mt_b, depth_u,
                                                      use_bf16, cls):
        """Every SRD advance adds a freshly computed stride (or an explicit 0).

        This is the property the post-assembly cls_subtile_stridefix.py used to
        establish by hand: walk back from each `s_add_u32 s[sgprSrdX+0], ..., sN`
        to the nearest write of sN and require it to be a stride compute or a
        zero seed. Anything else (an exec-mask `s_and`, a waveN-stride constant,
        an `s_mov_b64` lane pair) means the advance consumes scratch.
        """
        asm, _ = _build_subtile_store_asm(mt_a, mt_b, depth_u, cls=cls,
                                          use_bf16=use_bf16)
        sites = _srd_advance_producers(asm)
        assert sites, "expected at least one incToNextRow SRD advance"
        for lineno, reg, producer in sites:
            assert producer is not None, (
                f"SRD advance at line {lineno} consumes {reg}, which is never "
                f"written -- the increment is undefined")
            assert (_RE_STRIDE_COMPUTE.match(producer)
                    or _RE_ZERO_WRITE.match(producer)), (
                f"SRD advance at line {lineno} consumes {reg}, whose nearest "
                f"prior write is neither a stride compute nor an explicit zero: "
                f"{producer.strip()!r}. The SRD would advance by scratch "
                f"(-> hipErrorIllegalAddress once the CLS loop compacts).")

    @pytest.mark.parametrize("use_bf16", [False, True], ids=["f32", "bf16"])
    def test_cls_on_matches_cls_off_advance_count(self, use_bf16):
        """Turning CLS on must not drop or duplicate an SRD advance.

        The fix reorders instructions within a site; it must not change how many
        times the SRD is stepped, or D lands at the wrong rows.
        """
        off, _ = _build_subtile_store_asm(*COMPACTING_CONFIGS[0], cls=False,
                                          use_bf16=use_bf16)
        on, _ = _build_subtile_store_asm(*COMPACTING_CONFIGS[0], cls=True,
                                         use_bf16=use_bf16)
        n_off = len(_srd_advance_producers(off))
        n_on = len(_srd_advance_producers(on))
        # CLS may add exactly one extra leading seed advance (the chain seed that
        # forceinitrow0 opens); it must never remove one.
        assert n_on in (n_off, n_off + 1), (
            f"CLS changed the SRD advance count: off={n_off} on={n_on}")


class TestIncrementToNextRowSelfContainedStride:
    """Direct tests of the incrementToNextRow emit order.

    incrementToNextRow only reads `self.rowInc`, the bpe, and a few kernel keys,
    so it can be exercised without building a whole KernelWriter -- the same way
    TestComputeCLSLayoutSubtile calls computeCLSLayout as a pure function.
    """

    STMP = 40
    BPE = 2          # bf16 destination
    ROWS = 16        # subtile mBlockSize

    @classmethod
    def _emit(cls, clsOn, rowInc, selfContained, tc="D"):
        from types import SimpleNamespace
        from Tensile.AsmAddressCalculation import AddrCalculation

        addrCalc = AddrCalculation.__new__(AddrCalculation)
        addrCalc.rowInc = rowInc
        addrCalc.kernelWriter = SimpleNamespace(
            states=SimpleNamespace(bpeCexternal=cls.BPE,
                                   indexChars=list("IJKLMNOP")))
        kernel = {
            "CompactLoopStore": clsOn,
            "PackedC1IndicesX": [1],          # -> Stride<tc>J
            "_GlobalAccumulation": None,
            "WorkGroupReduction": False,
        }
        ss = SimpleNamespace(optSrdIncForRow=1)
        mod = addrCalc.incrementToNextRow(kernel, tc, ss, cls.STMP,
                                          forceinitrow0=1,
                                          selfContainedStride=selfContained)
        return [ln for ln in str(mod).splitlines() if ln.strip()]

    @staticmethod
    def _index_of(lines, needle):
        for i, ln in enumerate(lines):
            if needle in ln:
                return i
        return -1

    def test_cls_delayed_primer_computes_after_the_add(self):
        """Default CLS: the s_add consumes a value primed by an earlier call, and
        the stride compute at the end primes the next one."""
        lines = self._emit(clsOn=True, rowInc=self.ROWS, selfContained=False)
        add = self._index_of(lines, "s_add_u32")
        mul = self._index_of(lines, "s_mul_i32")
        assert add >= 0 and mul >= 0, lines
        assert add < mul, f"expected the delayed primer (add before mul): {lines}"

    @pytest.mark.parametrize("clsOn", [False, True], ids=["cls_off", "cls_on"])
    def test_self_contained_computes_before_the_add(self, clsOn):
        """selfContainedStride: the stride is computed first, so the s_add cannot
        consume anything another emitter left in s[stmp]."""
        lines = self._emit(clsOn=clsOn, rowInc=self.ROWS, selfContained=True)
        add = self._index_of(lines, "s_add_u32")
        mul = self._index_of(lines, "s_mul_i32")
        assert add >= 0 and mul >= 0, lines
        assert mul < add, f"expected mul before add: {lines}"
        assert sum("s_mul_i32" in ln for ln in lines) == 1, (
            f"the trailing primer must be gone -- a second stride compute would "
            f"leave a stale value for the next site to pick up: {lines}")

    def test_self_contained_scales_by_the_calls_own_rows(self):
        """The increment is this call's own rowInc * bpe, not a look-ahead."""
        lines = self._emit(clsOn=True, rowInc=self.ROWS, selfContained=True)
        mul = [ln for ln in lines if "s_mul_i32" in ln]
        assert len(mul) == 1, lines
        assert f", {self.ROWS * self.BPE}" in mul[0], (
            f"expected the stride scaled by rowInc({self.ROWS}) * bpe({self.BPE}) "
            f"= {self.ROWS * self.BPE}: {mul[0]!r}")

    def test_self_contained_row_zero_advances_by_nothing(self):
        """rowInc == 0 must produce a real zero, not a bpe-scaled stride.

        The shared stride builder folds numRows 0 and 1 into the same
        "scale by BPE" arm, which is correct for 1 and a whole spurious row for
        0. A self-contained seed site has to add exactly 0.
        """
        lines = self._emit(clsOn=True, rowInc=0, selfContained=True)
        add = self._index_of(lines, "s_add_u32")
        assert add >= 0, lines
        pre = lines[:add]
        assert any(re.search(rf"s_mov_b32 s{self.STMP}, 0\b", ln) for ln in pre), (
            f"expected an explicit zero increment before the s_add: {lines}")
        assert not any("s_lshl" in ln or "s_mul" in ln for ln in pre), (
            f"row 0 must not scale a stride: {lines}")

    def test_cls_off_is_unchanged_by_the_new_flag(self):
        """CLS-off already computed before the add; the flag must be inert there."""
        base = self._emit(clsOn=False, rowInc=self.ROWS, selfContained=False)
        flagged = self._emit(clsOn=False, rowInc=self.ROWS, selfContained=True)
        assert base == flagged, (base, flagged)


class TestSubtileSrdAdvanceCallSites:
    """The two emitters that step an SRD across subtile store bodies must opt out
    of the CLS delayed-primer chain. Pinned at the source level because the
    C-load site only fires on the beta path, which the store-module harness above
    does not build -- so nothing else in this file would notice it regressing.
    """

    @staticmethod
    def _source_of(module_name, class_name, method_name):
        import importlib
        import inspect
        mod = importlib.import_module(module_name)
        owner = getattr(mod, class_name) if class_name else mod
        return inspect.getsource(getattr(owner, method_name))

    def test_deferred_subtile_store_srd_inc_is_self_contained(self):
        """GlobalWriteBatch defers the sba=0 SrdD advance past the paired store
        that clobbers the scratch pair, so it must not use the chain."""
        src = self._source_of("Tensile.Components.GlobalWriteBatch",
                              "GlobalWriteBatchWriter", "_emitNonatomicAdd")
        calls = re.findall(
            r"_subtilePendingSrdDInc\s*=\s*addrCalc\.incrementToNextRow\([^)]*\)", src)
        assert len(calls) == 1, f"expected exactly one deferred SrdD advance: {calls}"
        assert "selfContainedStride=True" in calls[0], (
            "the deferred subtile SrdD advance must pass selfContainedStride=True; "
            "without it the s_add consumes the exec mask left in s[tmpS01]")

    def test_readinput_srd_inc_is_self_contained_on_subtile(self):
        """readInput's C/E/Gate advance carries its increment in the same scratch
        pair the subtile store uses for the exec mask."""
        src = self._source_of("Tensile.KernelWriterAssembly",
                              "KernelWriterAssembly", "readInput")
        assert "incrementToNextRow" in src
        assert 'selfContainedStride=kernel["UseSubtileImpl"]' in src, (
            "readInput must opt the subtile load-SRD advance out of the CLS "
            "delayed-primer chain")
