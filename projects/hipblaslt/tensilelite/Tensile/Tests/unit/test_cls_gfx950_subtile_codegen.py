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
# TODO: a subtile CLS-ON kernel is not yet numerically correct from codegen
# alone -- the store SRD for the A1 (second store) sub-batch is still emitted in
# the pre-CLS order, which faults with hipErrorIllegalAddress once the loop
# compacts. Until that reorder moves into codegen, only the layout/structure
# properties below (which ARE fully determined by codegen) can be asserted here.
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


def _counts(asm):
    on = len(_RE_IDX_ON.findall(asm))
    off = len(_RE_IDX_OFF.findall(asm))
    reads = len(_RE_ACC_READ.findall(asm))
    return on, off, reads


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
