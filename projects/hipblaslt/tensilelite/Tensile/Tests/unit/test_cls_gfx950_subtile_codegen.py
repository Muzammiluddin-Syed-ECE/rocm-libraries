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

import ast
import functools
import hashlib
import importlib
import inspect
import json
import os
import re
import shutil
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


# ---------------------------------------------------------------------------
# Preconditions, and why they are graded rather than uniformly skipped.
#
# This file used to hang entirely off one module-level
# `skipif(not _gfx950_cls_supported())`, where _gfx950_cls_supported() returned
# False for BOTH "this rocisa has no CLS bindings" and "this box has no gfx950
# assembler". Either one made every test in the file vanish and CI still report
# green -- a stale rocisa silently deleted the whole suite.
#
# The two conditions are not the same kind of thing, so they are graded:
#
#   * MISSING rocisa CLS BINDINGS -> hard error at import (collection error).
#     SSetGprIdxOn/Off are unconditional rocisa bindings: every rocisa built
#     from this tree has them, no capability or platform gates them, and nothing
#     about the host can make their absence legitimate. Their absence therefore
#     means a stale or broken install, never an unsupported platform, and the
#     only safe response is to be loud. Skipping here is what let 259 tests
#     disappear unnoticed.
#
#   * NO gfx950 ASSEMBLER (no amdclang++) -> skip, and only the classes that
#     actually render assembly. Building a store module needs rocisa's asm-caps
#     probe, which needs the assembler binary; a host without it genuinely
#     cannot run those tests. But most of this file does not render assembly at
#     all -- the computeCLSLayout math, the incrementToNextRow emitter, the
#     acc-read copy-site discovery and the source-level rules are pure Python --
#     so those keep running. A missing assembler now costs a clearly-reported
#     subset, not the whole file.
#
#   * ASSEMBLER PRESENT BUT rocisa INIT FAILED -> hard error. amdclang++ exists
#     yet the gfx950 assembler would not initialise: that is a broken toolchain,
#     not an unsupported one.
#
# test_cls_index_mode_bindings_are_importable and
# test_assembler_gated_classes_are_marked below are deliberately NOT gated on
# anything, so this file can never again collect zero tests.
# ---------------------------------------------------------------------------

CLS_BINDING_NAMES = ("SSetGprIdxOn", "SSetGprIdxOff")


def _classify_assembler(has_amdclang, init_error):
    """Grade the gfx950 assembler probe into ('active'|'unsupported'|'broken').

    Split out from the probe itself so the decision is testable without a
    toolchain: see TestPreconditionSemantics.
    """
    if init_error is None:
        return "active", None
    if not has_amdclang:
        return "unsupported", (
            "no gfx950 assembler on this host (amdclang++ not found); the "
            "assembly-rendering CLS tests cannot run here. Underlying error: %s"
            % (init_error,))
    return "broken", (
        "amdclang++ is present but the gfx950 assembler failed to initialise, "
        "which is a broken toolchain rather than an unsupported one: %s"
        % (init_error,))


def _import_cls_bindings(rocisa_instruction=None):
    """Import the rocisa CLS index-mode bindings. Returns (module, error_text).

    `rocisa_instruction` is injectable so the stale-install detection can be
    tested against a stub; it cannot be simulated through sys.modules, because
    `import rocisa.instruction` resolves through the already-bound parent
    package attribute.
    """
    if rocisa_instruction is None:
        try:
            import rocisa.instruction as rocisa_instruction
        except ImportError as exc:
            return None, "rocisa.instruction is not importable: %s" % (exc,)
    missing = [n for n in CLS_BINDING_NAMES if not hasattr(rocisa_instruction, n)]
    if missing:
        return None, (
            "rocisa at %s is missing the CompactLoopStore index-mode bindings %s"
            % (getattr(rocisa_instruction, "__file__", "<unknown>"), missing))
    return rocisa_instruction, None


_CLS_BINDINGS, _CLS_BINDINGS_ERROR = _import_cls_bindings()
if _CLS_BINDINGS_ERROR is not None:
    raise RuntimeError(
        "gfx950 CompactLoopStore tests cannot run: %s. These bindings are "
        "unconditional in rocisa, so this is a stale/broken rocisa build and "
        "not an unsupported platform -- rebuild rocisa (`invoke rocisa`). "
        "Failing loudly on purpose: skipping here silently deleted the whole "
        "CLS suite while CI reported green." % (_CLS_BINDINGS_ERROR,))


def _probe_gfx950_assembler():
    try:
        from gpu_test_helpers import init_rocisa
        init_rocisa(target="gfx950", wavesize=WAVESIZE_64)
        from rocisa import rocIsa
        if not rocIsa.getInstance().getAsmCaps():
            return "rocisa reported no gfx950 asm caps"
    except (ImportError, RuntimeError, OSError) as exc:
        return "%s: %s" % (type(exc).__name__, exc)
    return None


_ASM_STATE, _ASM_STATE_REASON = _classify_assembler(
    has_amdclang=bool(shutil.which("amdclang++") or os.path.exists("/usr/bin/amdclang++")),
    init_error=_probe_gfx950_assembler(),
)
if _ASM_STATE == "broken":
    raise RuntimeError("gfx950 CompactLoopStore tests cannot run: %s" % (_ASM_STATE_REASON,))

# Applied per class rather than as a module-level pytestmark: see the comment
# block above. Every class that renders assembly must carry it, and
# test_assembler_gated_classes_are_marked enforces that.
requires_gfx950_assembler = pytest.mark.skipif(
    _ASM_STATE != "active", reason=str(_ASM_STATE_REASON))

# Classes whose tests render assembly (directly or via mapAcctoArchRegs) and so
# need the gfx950 assembler. The rest of the file is pure Python.
ASSEMBLER_GATED_CLASSES = (
    "TestGfx950SubtileCLSCodegen",
    "TestAccVgprReadMechanismSelection",
    "TestClsIdxClusterHelper",
    "TestSubtileSrdAdvanceIsSelfContained",
    "TestIncrementToNextRowSelfContainedStride",
    "TestClsOffStoreGolden",
)


@pytest.fixture(scope="module", autouse=True)
def _rocisa_once():
    """Point rocisa at gfx950/wave64 for the whole module.

    Tolerates a missing assembler so the pure-Python classes still run: the
    classes that need it are gated by `requires_gfx950_assembler`.
    """
    if _ASM_STATE != "active":
        return
    from gpu_test_helpers import init_rocisa
    init_rocisa(target="gfx950", wavesize=WAVESIZE_64)


# ---------------------------------------------------------------------------
# Ungated sentinels: these two run on every host, in every tier. They are the
# reason this file can no longer be reduced to zero collected tests.
# ---------------------------------------------------------------------------

def test_cls_index_mode_bindings_are_importable():
    """The CLS bracket instructions exist and render as the ISA spells them.

    Ungated on purpose. If rocisa goes stale this is the test that fails; the
    import-time guard above turns the missing-binding case into a collection
    error, and this pins the rendering so a binding that exists but emits the
    wrong mnemonic/operand is caught too. Needs no assembler: constructing and
    rendering an instruction is pure rocisa.
    """
    from rocisa.container import mgpr
    on = str(_CLS_BINDINGS.SSetGprIdxOn(src=mgpr(0), mode="SRC0", comment="sentinel"))
    off = str(_CLS_BINDINGS.SSetGprIdxOff(comment="sentinel"))
    assert "s_set_gpr_idx_on" in on, on
    assert "gpr_idx(SRC0)" in on, on
    assert "m0" in on, "the index-mode source must be M0: %s" % (on,)
    assert "s_set_gpr_idx_off" in off, off


def test_assembler_gated_classes_are_marked():
    """Every assembly-rendering class carries `requires_gfx950_assembler`.

    The gate moved from one module-level pytestmark to per-class marks so a
    missing assembler no longer deletes the pure-Python half of the file. That
    trades one landmine for another -- a new class forgetting the mark -- so the
    mapping is asserted rather than trusted.
    """
    module = sys.modules[__name__]
    declared = {name for name, obj in vars(module).items()
                if name.startswith("Test") and inspect.isclass(obj)}
    unknown = set(ASSEMBLER_GATED_CLASSES) - declared
    assert not unknown, "ASSEMBLER_GATED_CLASSES names classes that do not exist: %s" % (unknown,)
    for name in ASSEMBLER_GATED_CLASSES:
        marks = getattr(vars(module)[name], "pytestmark", [])
        assert any(m.name == "skipif" for m in marks), (
            "%s renders assembly but is not gated on the gfx950 assembler" % (name,))


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

# Tiles used by the A1 SRD-advance assertions. The deferred
# selfContainedStride=True SrdD advance that A1 added lives on the 16-bit subtile
# store path (GlobalWriteBatch._emitNonatomicAdd, under `if is16bitSubtile`), so
# it is reachable only on the bf16 parametrizations -- the f32 assembly is
# byte-identical with and without it. With COMPACTING_CONFIGS alone that put the
# entire store half of A1 on 2 of 12 parametrizations, and dropping bf16 from the
# list would have silently un-covered it. These five tiles give it ten
# (5 tiles x cls_on/cls_off), and add the odd-MIWaveTile[0] geometries that route
# through the unpaired "orphan" subtile store.
A1_CONFIGS = COMPACTING_CONFIGS + [
    (96, 128, 64),    # MIWaveTile [3, 4] -- odd, orphan store path
    (160, 128, 64),   # MIWaveTile [5, 4] -- odd, orphan store path
    (256, 256, 64),   # MIWaveTile [8, 8] -- 7 row-group transitions, not 3
]
A1_CONFIG_IDS = [f"{a}x{b}" for a, b, _ in A1_CONFIGS]


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


@functools.lru_cache(maxsize=None)
def _store_asm(mt_a, mt_b, depth_u, cls, mi_wave_group=None, use_bf16=False,
               mi_arch_vgpr=False):
    """Memoized _build_subtile_store_asm.

    The builder is deterministic (verified: rebuilding a config after building
    others yields byte-identical text), and the widened parametrizations below
    ask for the same six or so modules many times over. Callers must not mutate
    the returned kernel dict.
    """
    asm, kernel = _build_subtile_store_asm(
        mt_a, mt_b, depth_u, cls,
        mi_wave_group=list(mi_wave_group) if mi_wave_group else None,
        use_bf16=use_bf16, mi_arch_vgpr=mi_arch_vgpr)
    return asm, kernel


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
#
# Both signs are matched. incrementToNextRow emits s_sub_u32 when numRows < 0
# (AsmAddressCalculation.py:1040); the add-only form of this pattern could not
# see a negative advance at all, so a defect confined to the subtract arm was
# invisible even in principle. No in-tree config the harness builds emits one
# today -- TestIncrementToNextRowSelfContainedStride covers the subtract arm
# directly -- but the predicate no longer looks away if one appears.
_RE_SRD_ADVANCE = re.compile(
    r"^\s*s_(add|sub)_u32 s\[sgprSrd([CD])\+0\], s\[sgprSrd\2\+0\], (s\d+)\b.*incToNextRow")
_RE_STRIDE_COMPUTE = re.compile(r"^\s*s_(?:mul_i32|lshl_b32) s\d+, s\[sgprStride")
_RE_ZERO_WRITE = re.compile(r"^\s*s_mov_b32 s\d+, 0\s*(?://.*)?$")

# The two stride-compute forms, with their magnitude operand captured. This is
# what _RE_STRIDE_COMPUTE deliberately ignores and what a shape-only predicate
# therefore cannot see: a 2x row stride emits the same s_mul_i32 off the same
# Stride sgpr and differs only in this operand.
_RE_STRIDE_MUL = re.compile(r"^\s*s_mul_i32 s\d+, s\[sgprStride\w+\], (\d+)\s*(?://.*)?$")
_RE_STRIDE_LSHL = re.compile(
    r"^\s*s_lshl_b32 s\d+, s\[sgprStride\w+\], (?:0x)?([0-9a-fA-F]+)\s*(?://.*)?$")


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
        reg = m.group(3)
        producer = next((lines[j] for j in range(i - 1, -1, -1)
                         if _writes_sgpr(lines[j], reg)), None)
        sites.append((i + 1, reg, producer))
    return sites


def _advance_bytes(producer):
    """How many bytes of stride the producer line actually contributes.

    Returns the multiplier of the row stride in bytes, 0 for an explicit zero
    seed, or None when the producer is not a recognisable stride source at all
    (i.e. the advance consumes scratch). The two stride forms encode the amount
    differently:
        s_mul_i32  sN, s[sgprStrideDJ], 32   -> 32 bytes  (numRows * bpe)
        s_lshl_b32 sN, s[sgprStrideDJ], 0x1  -> 2 bytes   (1 row, scaled by bpe)
    """
    if producer is None:
        return None
    code = producer.split("//")[0].rstrip()
    m = _RE_STRIDE_MUL.match(code)
    if m:
        return int(m.group(1))
    m = _RE_STRIDE_LSHL.match(code)
    if m:
        return 1 << int(m.group(1), 16)
    if _RE_ZERO_WRITE.match(code):
        return 0
    return None


def _srd_advance_amounts(asm):
    """[(lineno, sign, bytes_or_None)] for every incToNextRow SRD advance.

    sign is +1 for the s_add_u32 form and -1 for s_sub_u32.
    """
    lines = asm.splitlines()
    out = []
    for lineno, reg, producer in _srd_advance_producers(asm):
        sign = -1 if _RE_SRD_ADVANCE.match(lines[lineno - 1]).group(1) == "sub" else 1
        out.append((lineno, sign, _advance_bytes(producer)))
    return out


def _moving_advances(asm):
    """The advances that actually move the SRD, as signed byte amounts.

    Zero-amount sites are excluded: under CLS the first site is a chain seed
    whose s_add consumes a register that was zeroed in the preamble, so it steps
    the SRD by nothing. Excluding them is what lets the CLS-on / CLS-off
    comparison be an equality instead of the `n_off .. n_off + 1` band that the
    seed forced -- a band which, on f32, the legitimate +1 consumed entirely, so
    one dropped advance landed back inside it and passed.
    """
    return [sign * amount for _, sign, amount in _srd_advance_amounts(asm) if amount]


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


# ---- acc-read copy-site discovery ----
#
# The bare/bracketed rule is about the methods that take their own copy of
# codes.accVgprRead: that copy is the list a consumer pops reads off, and the
# copy site is where a bracket would naturally be introduced. Those methods are
# discovered from the AST instead of being named by hand, because naming them by
# hand is exactly how the rule came to be enforced against three methods that
# hold no copy site at all (partialsWriteBatch / fixupBatch / partialWriteBatch
# / lastGsuWgReduction are the emitters, but the deepcopy lives one level up in
# partialsWriteProcedure / fixupStep / reductionProcedure).

_ACC_READ_COPY_MODULES = (
    "Tensile.Components.GSU",
    "Tensile.Components.StreamK",
    "Tensile.Components.LSU",
    "Tensile.KernelWriterAssembly",
)


@functools.lru_cache(maxsize=None)
def _acc_read_copy_sites():
    """Every deepcopy of an accVgprRead module, as {(module, qualname): lineno}.

    Walks each store module's AST once (~0.25 s for all four, paid once per
    session) and records the enclosing function of every `deepcopy(...
    accVgprRead)` call.
    """
    sites = {}
    for module_name in _ACC_READ_COPY_MODULES:
        module = importlib.import_module(module_name)
        source = inspect.getsource(module)
        tree = ast.parse(source)

        def visit(node, prefix):
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    qualname = f"{prefix}.{child.name}" if prefix else child.name
                    if isinstance(child, ast.ClassDef):
                        visit(child, qualname)
                        continue
                    for sub in ast.walk(child):
                        if not isinstance(sub, ast.Call):
                            continue
                        func = sub.func
                        name = getattr(func, "id", None) or getattr(func, "attr", None)
                        if name != "deepcopy" or not sub.args:
                            continue
                        if "accVgprRead" in ast.unparse(sub.args[0]):
                            sites[(module_name, qualname)] = sub.lineno
                    visit(child, qualname)
                else:
                    visit(child, prefix)

        visit(tree, "")
    return sites


def _source_of_qualname(module_name, qualname):
    obj = importlib.import_module(module_name)
    for part in qualname.split("."):
        obj = getattr(obj, part)
    return inspect.getsource(obj)


# Tokens that would open or close a CLS index-mode bracket. A bare consumer must
# contain none of them.
BRACKET_TOKENS = ("clsWrapIdxCluster", "clsIdxModeOn", "clsIdxModeOff",
                  "SSetGprIdxOn", "SSetGprIdxOff")


# ---- CLS layout: the m0Step property, shared across all three arms ----

def assert_m0_step_reaches_every_slice(kernel, num_batches):
    """m0Step must be exactly the acc-src delta between consecutive CLS bodies.

    `s_set_gpr_idx_on m0` lands on iteration j's accumulator slice only if, for
    every read position p in the body,
        arch2acc[p + j*readsPerBody] == arch2acc[p] + j*m0Step
    against the real accToArchMapper. A wrong m0Step reads the wrong slice on
    every iteration past the first: silently wrong D, no fault, no diagnostic.

    Checked against the mapping rather than against a number, so an m0Step
    derived from an unrelated quantity is caught whatever value it happens to
    take. Returns True when the layout compacts (so callers can assert they are
    not vacuously passing), False when iterCount == 1 and m0Step is dead.
    """
    from Tensile.Components.GlobalWriteBatch import GlobalWriteBatchWriter
    from Tensile.KernelWriterModules import accToArchMapper, getAccToArchLen

    bpb, iterCount, m0Step = GlobalWriteBatchWriter.computeCLSLayout(kernel, num_batches)
    assert bpb * iterCount == num_batches, (bpb, iterCount, num_batches)
    if iterCount == 1:
        return False
    _, arch2acc = accToArchMapper(kernel)
    accLen = getAccToArchLen(kernel)
    assert accLen % iterCount == 0, (
        f"CLS body count {iterCount} does not divide the acc-read list length "
        f"{accLen}, so the body is not a whole prefix of it")
    readsPerBody = accLen // iterCount
    for p in range(readsPerBody):
        for it in range(1, iterCount):
            assert arch2acc[p + it * readsPerBody] == arch2acc[p] + it * m0Step, (
                f"m0Step {m0Step} does not reach slice {it} from read {p}: "
                f"src({p + it * readsPerBody})={arch2acc[p + it * readsPerBody]} "
                f"vs src({p})+{it}*{m0Step}={arch2acc[p] + it * m0Step} "
                f"(numBatches={num_batches}, iterCount={iterCount}, "
                f"MIWaveTile={kernel['MIWaveTile']}, VW=({kernel['VectorWidthA']},"
                f"{kernel['VectorWidthB']}), SourceSwap={kernel['SourceSwap']})")
    return True


def cls_layout_kernel(mi_wave_tile, vwa=1, vwb=1, source_swap=False,
                      wave=WAVESIZE_64, mi=(16, 16), bm=1, bn=1, nepbs=8):
    """Minimal kernel dict for computeCLSLayout, wide enough for all three arms.

    computeCLSLayout selects on (outerTT1, VW1, SourceSwap), so one factory that
    can express every combination is what lets the m0Step property be checked on
    cases (a) and (b) and not just (c).
    """
    return {
        "EnableMatrixInstruction": True,
        "VectorWidthA": vwa,
        "VectorWidthB": vwb,
        "MIWaveTile": list(mi_wave_tile),
        "MatrixInstM": mi[0],
        "MatrixInstN": mi[1],
        "MatrixInstBM": bm,
        "MatrixInstBN": bn,
        "WavefrontSize": wave,
        "NumElementsPerBatchStore": nepbs,
        "SourceSwap": source_swap,
        "StoreRemapVectorWidth": 0,
        "StreamK": 0,
    }


# ---- CLS-off assembly golden ----

GOLDEN_PATH = os.path.join(SCRIPT_DIR, "test_data",
                           "cls_gfx950_cls_off_store.golden.json")
GOLDEN_UPDATE_ENV = "CLS_GOLDEN_UPDATE"


def _normalize_asm(asm):
    """Canonical instruction text: no comments, no blank lines, no indentation.

    Comments are dropped because they carry emitter-internal wording that churns
    without changing a single emitted bit; everything else is kept, including
    labels and register numbers, so a real codegen change cannot hide.
    """
    out = []
    for raw in asm.splitlines():
        line = raw.split("//")[0].strip()
        if not line or line.startswith("/*"):
            continue
        out.append(line)
    return "\n".join(out)


def _asm_fingerprint(asm):
    """{digest, instructions, opcodes} for a normalized module.

    Three levels on purpose, so a mismatch says *what* moved rather than only
    that something did:
      * digest      -- sha256 of the canonical text: catches any change at all,
                       including an operand or register renumbering.
      * instructions -- how many instructions were emitted: catches a change in
                       code size.
      * opcodes     -- per-mnemonic histogram: catches a change in instruction
                       mix, and is the part a human can actually diff in review.
    """
    text = _normalize_asm(asm)
    opcodes = {}
    instructions = 0
    for line in text.splitlines():
        head = line.split()[0]
        if head.endswith(":") or head.startswith("."):
            continue          # label or assembler directive, not an instruction
        instructions += 1
        opcodes[head] = opcodes.get(head, 0) + 1
    return {
        "digest": hashlib.sha256(text.encode()).hexdigest(),
        "instructions": instructions,
        "opcodes": dict(sorted(opcodes.items())),
    }


def _load_golden():
    with open(GOLDEN_PATH) as fh:
        return json.load(fh)


def _describe_fingerprint_diff(expected, actual):
    """Human-readable account of which of the three levels moved."""
    notes = []
    if expected["instructions"] != actual["instructions"]:
        notes.append("instruction count %d -> %d"
                     % (expected["instructions"], actual["instructions"]))
    exp_ops, act_ops = expected["opcodes"], actual["opcodes"]
    deltas = {op: act_ops.get(op, 0) - exp_ops.get(op, 0)
              for op in set(exp_ops) | set(act_ops)
              if act_ops.get(op, 0) != exp_ops.get(op, 0)}
    if deltas:
        notes.append("opcode deltas %s" % (dict(sorted(deltas.items())),))
    elif expected["digest"] != actual["digest"]:
        notes.append("same instruction mix but different operands/order "
                     "(digest %s -> %s)" % (expected["digest"][:12], actual["digest"][:12]))
    return "; ".join(notes) or "no difference"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@requires_gfx950_assembler
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


@requires_gfx950_assembler
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


@requires_gfx950_assembler
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


class TestAccVgprReadCopySites:
    """The bare/bracketed rule, enforced on the methods that actually copy the
    acc-read list rather than on a hand-maintained list of emitter names.

    BARE_CONSUMERS above names the five methods that *emit* reads outside the CLS
    loop, and greping them is worth keeping -- a bracket named in any of them is
    wrong. But it is not where the rule can be broken most easily: the
    `deepcopy(codes.accVgprRead)` that produces the list a consumer pops from
    happens one level up for three of the five, in methods that list never
    mentioned (GSUOn.reductionProcedure, StreamK.partialsWriteProcedure,
    StreamK.fixupStep). A bracket opened at any of those copy sites passed every
    test in this file.

    On the source-grep question: a real codegen backstop -- building a StreamK,
    GSU or LSU store module with CLS on and asserting _scan_brackets finds
    nothing -- would be strictly better, and it is not affordable here.
    LSUOn.writeReadReduction is reached from KernelWriter's main-body emitter
    (KernelWriter.py:6611), not from a store entry point, and
    localSplitUGlobalWriteIndices already wants LSUValidOffset0 /
    LSUelementsPerLSUWave state that no in-tree GPU-free harness produces;
    GSU=2 and StreamK=1 both fault inside the store builder for want of
    _GlobalAccumulation and streamK writer state (verified). Standing all of
    that up is a per-component harness on the scale of test_storeD_roundtrip's
    builders, which this suite may not modify, for a rule whose in-loop half is
    already killed by 13 codegen tests. So the grep stays -- but it is now aimed
    by the AST at the methods that hold the copy sites, and a new copy site
    breaks the census below instead of quietly escaping the rule.
    """

    # Every acc-read copy site, and which side of the bracket rule it is on.
    # Discovered set is asserted against this, so a new consumer cannot be added
    # without classifying it.
    COPY_SITE_ROLES = {
        # Outside the CLS loop: pops the full read list at literal indices, so a
        # bracket here would make an M0-immune read depend on M0 -- which on GFX9
        # is also the LDS base/limit register, i.e. a wrong LDS window (possibly
        # a fault) in the reduction path rather than merely a wrong value.
        ("Tensile.Components.GSU", "GSUOn.reductionProcedure"): "bare",
        ("Tensile.Components.StreamK", "StreamK.partialsWriteProcedure"): "bare",
        ("Tensile.Components.StreamK", "StreamK.fixupStep"): "bare",
        ("Tensile.Components.LSU", "LSUOn.writeReadReduction"): "bare",
        # Inside the CLS loop: pops only the truncated prefix and relies on M0 to
        # reach the rest, so this one must bracket -- via the shared helper.
        ("Tensile.KernelWriterAssembly",
         "KernelWriterAssembly.globalWriteElementBatch"): "in-loop",
    }

    def test_copy_site_census_is_complete(self):
        """The discovered copy sites are exactly the classified ones.

        A new `deepcopy(codes.accVgprRead)` anywhere in the store components is a
        new consumer of the read list, and every consumer is on one side of the
        bracket rule or the other. Failing here forces that decision instead of
        letting the new site inherit whichever behaviour it happened to get.
        """
        discovered = set(_acc_read_copy_sites())
        classified = set(self.COPY_SITE_ROLES)
        assert discovered == classified, (
            "acc-read copy sites changed.\n"
            "  new, unclassified: %s\n"
            "  classified but gone: %s\n"
            "Each new site must be added to COPY_SITE_ROLES as 'bare' (pops the "
            "full list outside the CLS loop) or 'in-loop' (pops a prefix and "
            "relies on M0)."
            % (sorted(discovered - classified), sorted(classified - discovered)))

    @pytest.mark.parametrize(
        "module_name,qualname",
        [k for k, v in COPY_SITE_ROLES.items() if v == "bare"],
        ids=[k[1] for k, v in COPY_SITE_ROLES.items() if v == "bare"])
    def test_bare_copy_sites_open_no_bracket(self, module_name, qualname):
        """No bracket in the method that copies the read list for a bare consumer."""
        src = _source_of_qualname(module_name, qualname)
        assert "accVgprRead" in src, (
            f"{qualname} was discovered as an acc-read copy site but its source "
            f"does not mention accVgprRead -- the AST walk and the source lookup "
            f"disagree, so this test is not looking at the code it thinks it is")
        for token in BRACKET_TOKENS:
            assert token not in src, (
                f"{qualname} emits {token} around a copy of codes.accVgprRead: it "
                f"pops the read list in full outside the CLS loop, so its reads "
                f"are already at literal indices and must stay M0-immune")

    @pytest.mark.parametrize(
        "module_name,qualname",
        [k for k, v in COPY_SITE_ROLES.items() if v == "in-loop"],
        ids=[k[1] for k, v in COPY_SITE_ROLES.items() if v == "in-loop"])
    def test_in_loop_copy_site_does_not_open_its_own_bracket(self, module_name, qualname):
        """The in-loop copy site hands the list on; it must not bracket in place.

        globalWriteElementBatch copies the list and passes it to
        GlobalWriteBatchWriter, which brackets per contiguous cluster via
        clsWrapIdxCluster. A bracket opened here instead would wrap the whole
        batch -- including the non-acc-read instructions between clusters, whose
        SRC0 would then be M0-relative too.
        """
        src = _source_of_qualname(module_name, qualname)
        for token in ("SSetGprIdxOn", "SSetGprIdxOff", "clsIdxModeOn", "clsIdxModeOff"):
            assert token not in src, (
                f"{qualname} open-codes {token}; the bracket belongs to "
                f"clsWrapIdxCluster at the per-cluster consumer")


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

        The assertion body now lives in assert_m0_step_reaches_every_slice so
        cases (a) and (b) can be held to the same property; see
        TestComputeCLSLayoutM0StepAllArms.
        """
        assert_m0_step_reaches_every_slice(
            self._kernel(mi_wave_tile, vwa=vwa, wave=wave), num_batches)


class TestComputeCLSLayoutM0StepAllArms:
    """m0Step held to the accumulator map on ALL THREE arms of computeCLSLayout.

    Case (c) was cross-checked against the real accToArchMapper; cases (a) and
    (b) were checked against nothing stronger than `m0Step > 0`
    (test_cls_m0_base_init_and_stride). Injecting `m0Step += 1` on the
    non-SourceSwap arms therefore left the whole suite green, and a wrong m0Step
    there means every CLS iteration past the first reads the wrong accumulator
    slice: D is silently wrong, with no fault and no diagnostic. That is the
    highest-severity silent failure mode in the feature and the one defect class
    already found once by hand rather than by a test.

    The arms are selected by (outerTT1, VW1, SourceSwap), which is why the
    geometries below are grouped that way:
        (a) outerTT1 >  1, VW1 == 1              -> iter = wgIdx1
        (b) outerTT1 == 1, VW1 >  1, !SourceSwap -> iter = vw1
        (c) outerTT1 == 1, VW1 == 1,  SourceSwap -> iter = tIdx  (above)
    numBatches is swept inside each test rather than parametrized, to keep the
    property at full coverage without multiplying the reported test count.
    """

    NUM_BATCHES = (1, 2, 3, 4, 5, 6, 8, 9, 12, 16, 24, 32)

    # (a): outerTT1 == MIWaveTile[1] > 1 with VW1 == 1. Covers both MI shapes,
    # both wave sizes, and a VectorWidthA > 1 that makes outerTT0 < MIWaveTile[0].
    CASE_A = [
        ([1, 2], 1, (16, 16), WAVESIZE_64),
        ([2, 2], 2, (16, 16), WAVESIZE_64),
        ([4, 4], 1, (16, 16), WAVESIZE_64),
        ([8, 4], 2, (16, 16), WAVESIZE_64),
        ([2, 8], 1, (16, 16), WAVESIZE_64),
        ([4, 2], 4, (16, 16), WAVESIZE_64),
        ([4, 4], 1, (32, 32), WAVESIZE_64),
        ([8, 4], 2, (32, 32), WAVESIZE_64),
        ([4, 4], 1, (16, 16), 32),
        ([2, 2], 2, (32, 32), 32),
    ]

    # (b): outerTT1 == 1 via MIWaveTile[1] == VW1, VW1 > 1, no SourceSwap.
    CASE_B = [
        ([1, 2], 1, 2, (16, 16), WAVESIZE_64),
        ([2, 2], 2, 2, (16, 16), WAVESIZE_64),
        ([4, 4], 1, 4, (16, 16), WAVESIZE_64),
        ([8, 4], 4, 4, (16, 16), WAVESIZE_64),
        ([4, 2], 2, 2, (32, 32), WAVESIZE_64),
        ([2, 4], 1, 4, (16, 16), 32),
    ]

    @pytest.mark.parametrize("mi_wave_tile,vwa,mi,wave", CASE_A,
                             ids=[f"MIWT{t}-VWA{v}-MI{m[0]}x{m[1]}-w{w}"
                                  for t, v, m, w in CASE_A])
    def test_case_a_m0_step_reaches_the_slice_the_loop_claims(self, mi_wave_tile,
                                                              vwa, mi, wave):
        kernel = cls_layout_kernel(mi_wave_tile, vwa=vwa, vwb=1,
                                   source_swap=False, mi=mi, wave=wave)
        compacted = [nb for nb in self.NUM_BATCHES
                     if assert_m0_step_reaches_every_slice(kernel, nb)]
        assert compacted, (
            f"no numBatches in {self.NUM_BATCHES} compacts for MIWaveTile="
            f"{mi_wave_tile} VWA={vwa} MI={mi} wave={wave}, so this "
            f"parametrization asserts nothing about m0Step")

    @pytest.mark.parametrize("mi_wave_tile,vwa,vwb,mi,wave", CASE_B,
                             ids=[f"MIWT{t}-VWA{a}-VWB{b}-MI{m[0]}x{m[1]}-w{w}"
                                  for t, a, b, m, w in CASE_B])
    def test_case_b_m0_step_reaches_the_slice_the_loop_claims(self, mi_wave_tile,
                                                              vwa, vwb, mi, wave):
        kernel = cls_layout_kernel(mi_wave_tile, vwa=vwa, vwb=vwb,
                                   source_swap=False, mi=mi, wave=wave)
        assert kernel["MIWaveTile"][1] // vwb == 1, "not case (b): outerTT1 != 1"
        compacted = [nb for nb in self.NUM_BATCHES
                     if assert_m0_step_reaches_every_slice(kernel, nb)]
        assert compacted, (
            f"no numBatches in {self.NUM_BATCHES} compacts for MIWaveTile="
            f"{mi_wave_tile} VWA={vwa} VWB={vwb} MI={mi} wave={wave}, so this "
            f"parametrization asserts nothing about m0Step")

    @pytest.mark.parametrize("mi_wave_tile,vwa,mi,wave", CASE_A[:4],
                             ids=[f"MIWT{t}-VWA{v}-MI{m[0]}x{m[1]}-w{w}"
                                  for t, v, m, w in CASE_A[:4]])
    def test_case_a_arm_is_the_one_under_test(self, mi_wave_tile, vwa, mi, wave):
        """Guard the guard: these geometries really do select case (a).

        If a refactor moved the arm boundaries, the case (a) tests above would
        keep passing while silently exercising some other arm.
        """
        kernel = cls_layout_kernel(mi_wave_tile, vwa=vwa, vwb=1, source_swap=False,
                                   mi=mi, wave=wave)
        assert kernel["MIWaveTile"][1] // kernel["VectorWidthB"] > 1
        assert kernel["VectorWidthB"] == 1
        assert not kernel["SourceSwap"]


@requires_gfx950_assembler
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
    @pytest.mark.parametrize("mt_a,mt_b,depth_u", A1_CONFIGS, ids=A1_CONFIG_IDS)
    def test_srd_advance_consumes_a_stride_not_scratch(self, mt_a, mt_b, depth_u,
                                                      use_bf16, cls):
        """Every SRD advance adds a freshly computed stride (or an explicit 0).

        This is the property the post-assembly cls_subtile_stridefix.py used to
        establish by hand: walk back from each `s_add_u32 s[sgprSrdX+0], ..., sN`
        to the nearest write of sN and require it to be a stride compute or a
        zero seed. Anything else (an exec-mask `s_and`, a waveN-stride constant,
        an `s_mov_b64` lane pair) means the advance consumes scratch.
        """
        asm, _ = _store_asm(mt_a, mt_b, depth_u, cls=cls, use_bf16=use_bf16)
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

    # -- magnitude, not just shape --

    @pytest.mark.parametrize("cls", [False, True], ids=["cls_off", "cls_on"])
    @pytest.mark.parametrize("use_bf16", [False, True], ids=["f32", "bf16"])
    @pytest.mark.parametrize("mt_a,mt_b,depth_u", A1_CONFIGS, ids=A1_CONFIG_IDS)
    def test_srd_advance_steps_exactly_one_mi_row_block(self, mt_a, mt_b, depth_u,
                                                       use_bf16, cls):
        """Every advance moves the SRD by one MI output block of rows, not two.

        test_srd_advance_consumes_a_stride_not_scratch pins the *shape* of the
        advance -- that it consumes a freshly computed stride rather than
        scratch -- and deliberately ignores the multiplier. A 2x row stride emits
        the same s_mul_i32 off the same Stride sgpr and differs only in that
        operand, so it passed every assertion in this file while stepping D two
        rows per advance: on a bounds-checked buffer an illegal address,
        otherwise silent corruption of every row.

        The expectation is derived, not tabulated: one advance carries the SRD
        across one MatrixInst output block in the coord1 direction, so the byte
        amount is MatrixInstN * bpe. (Confirmed invariant under MIWaveGroup:
        [1,1], [2,2], [1,4] and [4,1] all emit the same MatrixInstN * bpe.)
        """
        asm, kernel = _store_asm(mt_a, mt_b, depth_u, cls=cls, use_bf16=use_bf16)
        bpe = int(kernel["ProblemType"]["DestDataType"].numBytes())
        expected = kernel["MatrixInstN"] * bpe

        amounts = _srd_advance_amounts(asm)
        assert amounts, "expected at least one incToNextRow SRD advance"
        moving = [(lineno, sign * amount) for lineno, sign, amount in amounts if amount]
        assert moving, (
            "every SRD advance steps by zero: the store never leaves row 0")
        for lineno, amount in moving:
            assert amount == expected, (
                f"SRD advance at line {lineno} moves the SRD by {amount} bytes, "
                f"expected {expected} = MatrixInstN({kernel['MatrixInstN']}) * "
                f"bpe({bpe}). A wrong magnitude lands D in the wrong rows on "
                f"every subtile CLS kernel.")

        # A zero-amount advance is the CLS delayed-primer chain seed: its s_add
        # consumes a register the preamble zeroed, so it steps by nothing. There
        # is at most one, and CLS-off has none. Bounding it is what keeps the
        # "explicit zero is an acceptable producer" allowance from excusing a
        # whole run of advances that quietly add nothing -- _RE_ZERO_WRITE
        # matches any `s_mov_b32 sN, 0` from any emitter, and on the f32 CLS-on
        # arm the one it matches is `// Init sgpr offset`, not the A1 seed.
        seeds = [lineno for lineno, sign, amount in amounts if amount == 0]
        assert len(seeds) <= 1, (
            f"{len(seeds)} SRD advances add an explicit zero (lines {seeds}); at "
            f"most one chain seed is legitimate")
        if not cls:
            assert not seeds, (
                f"CLS-off has no delayed-primer chain to seed, so no advance may "
                f"add zero (lines {seeds})")

    @pytest.mark.parametrize("cls", [False, True], ids=["cls_off", "cls_on"])
    @pytest.mark.parametrize("use_bf16", [False, True], ids=["f32", "bf16"])
    @pytest.mark.parametrize("mt_a,mt_b,depth_u", A1_CONFIGS, ids=A1_CONFIG_IDS)
    def test_srd_advance_count_matches_the_tiles_row_groups(self, mt_a, mt_b, depth_u,
                                                            use_bf16, cls):
        """The SRD is stepped once per row-group transition the tile implies.

        A wave owns outerTT1 = MIWaveTile[1] / VectorWidthB groups of rows in the
        coord1 direction and has to cross between them outerTT1 - 1 times. This
        is an absolute count, which is what a differential CLS-on-vs-CLS-off
        comparison cannot be: a drop that hits both arms is invisible to a
        differential by construction (the audit's a1_drop_advance_both mutant was
        caught only incidentally, by an emptiness guard, and only on bf16).
        """
        asm, kernel = _store_asm(mt_a, mt_b, depth_u, cls=cls, use_bf16=use_bf16)
        outer_tt1 = kernel["MIWaveTile"][1] // kernel["VectorWidthB"]
        expected = outer_tt1 - 1
        moving = _moving_advances(asm)
        assert len(moving) == expected, (
            f"expected {expected} SRD advances that move the SRD "
            f"(outerTT1={outer_tt1} row groups, so outerTT1-1 transitions), got "
            f"{len(moving)}: {moving}. A missing advance writes two row groups "
            f"on top of each other; an extra one skips a row group entirely.")

    @pytest.mark.parametrize("use_bf16", [False, True], ids=["f32", "bf16"])
    @pytest.mark.parametrize("mt_a,mt_b,depth_u", A1_CONFIGS, ids=A1_CONFIG_IDS)
    def test_cls_does_not_change_the_srd_advance_schedule(self, mt_a, mt_b, depth_u,
                                                          use_bf16):
        """CLS must not change which amounts the SRD is stepped by, or how often.

        Replaces test_cls_on_matches_cls_off_advance_count, which compared raw
        site counts with `n_on in (n_off, n_off + 1)` to make room for the CLS
        chain seed. That slack was not sound: on f32 the legitimate +1 consumed
        all of it, so a dropped advance landed back at n_off and passed. Counting
        only the advances that actually move the SRD identifies the seed by its
        zero amount instead of by budgeting for it, which turns the band into an
        equality -- and comparing the amounts, not just how many there are, also
        catches CLS changing a magnitude while preserving the count.
        """
        off, _ = _store_asm(mt_a, mt_b, depth_u, cls=False, use_bf16=use_bf16)
        on, _ = _store_asm(mt_a, mt_b, depth_u, cls=True, use_bf16=use_bf16)
        moving_off = sorted(_moving_advances(off))
        moving_on = sorted(_moving_advances(on))
        assert moving_on == moving_off, (
            f"CLS changed the SRD advance schedule: off={moving_off} "
            f"on={moving_on} (byte amounts, signed; zero-amount chain seeds "
            f"excluded)")


@requires_gfx950_assembler
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
    def _emit(cls, clsOn, rowInc, selfContained, tc="D", overrideAfterPrimerRows=0):
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
                                          overrideAfterPrimerRows=overrideAfterPrimerRows,
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

    # -- the delayed primer's look-ahead override --

    @pytest.mark.parametrize("override_rows", [1, 2, 16, 32])
    def test_override_after_primer_rows_primes_the_next_calls_advance(self, override_rows):
        """The AFTER primer scales by the override, not by this call's own rowInc.

        In the delayed-primer chain, call N's trailing stride compute is what call
        N+1's s_add consumes, so it must carry N+1's rowInc. `overrideAfterPrimerRows`
        is the caller-computed look-ahead that supplies it; without it the primer
        carries N's own rowInc and every advance is one element behind -- the
        off-by-one the commit message calls out. Nothing in-tree referenced this
        parameter from a test, in either direction.
        """
        own_rows = self.ROWS
        assert override_rows != own_rows or override_rows == self.ROWS
        lines = self._emit(clsOn=True, rowInc=own_rows, selfContained=False,
                           overrideAfterPrimerRows=override_rows)
        add = self._index_of(lines, "s_add_u32")
        assert add >= 0, lines
        primer = [ln for ln in lines[add:] if "s_mul_i32" in ln or "s_lshl_b32" in ln]
        assert len(primer) == 1, (
            f"expected exactly one AFTER primer following the s_add: {lines}")
        if override_rows > 1:
            assert f", {override_rows * self.BPE}" in primer[0], (
                f"expected the primer scaled by overrideAfterPrimerRows"
                f"({override_rows}) * bpe({self.BPE}) = {override_rows * self.BPE}, "
                f"not by this call's own rowInc({own_rows}): {primer[0]!r}")
        else:
            # numRows == 1 folds into the shift-by-log2(bpe) arm.
            assert "s_lshl_b32" in primer[0], primer[0]

    def test_override_is_ignored_by_a_self_contained_call(self):
        """A self-contained call emits no AFTER primer at all, override or not.

        The whole point of selfContainedStride is that nothing is left in s[stmp]
        for a later call to consume; honouring the look-ahead there would put a
        stale value back.
        """
        plain = self._emit(clsOn=True, rowInc=self.ROWS, selfContained=True)
        overridden = self._emit(clsOn=True, rowInc=self.ROWS, selfContained=True,
                                overrideAfterPrimerRows=32)
        assert plain == overridden, (plain, overridden)

    # -- the negative (s_sub_u32) advance form --

    def test_negative_row_inc_subtracts_its_own_magnitude(self):
        """rowInc < 0 steps the SRD backwards by |rowInc| * bpe.

        incrementToNextRow emits s_sub_u32 for numRows < 0 and scales the stride
        by (-numRows) * bpe. No configuration the store harness builds emits this
        form, so it is unreachable from the assembly predicates and pinned here
        instead: a sign error would walk the SRD off the front of the buffer.
        """
        lines = self._emit(clsOn=True, rowInc=-self.ROWS, selfContained=True)
        sub = self._index_of(lines, "s_sub_u32")
        assert sub >= 0, f"expected an s_sub_u32 for a negative rowInc: {lines}"
        assert self._index_of(lines, "s_add_u32") < 0, (
            f"a negative advance must not also emit an s_add_u32: {lines}")
        mul = [ln for ln in lines if "s_mul_i32" in ln]
        assert len(mul) == 1, lines
        assert self._index_of(lines, "s_mul_i32") < sub, (
            f"self-contained: the stride must be computed before the s_sub: {lines}")
        assert f", {self.ROWS * self.BPE}" in mul[0], (
            f"expected |rowInc|({self.ROWS}) * bpe({self.BPE}) = "
            f"{self.ROWS * self.BPE}: {mul[0]!r}")
        assert "s_subb_u32" in " ".join(lines), (
            f"the high half of the SRD must borrow: {lines}")


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


# Tiles the CLS-off golden covers: both compacting shapes and the
# non-compacting one, each at f32 and bf16.
GOLDEN_CONFIGS = COMPACTING_CONFIGS + [NONCOMPACTING_CONFIG]


@requires_gfx950_assembler
class TestClsOffStoreGolden:
    """CLS-OFF assembly non-regression golden.

    Every shipped logic file sets CompactLoopStore: false, so CLS-off is the
    production path -- and CLS-off shares the store emitter with CLS-on. Nothing
    in-tree noticed a CLS change perturbing it: the characterization codegen
    suite snapshots an order-invariant {basename, err} digest, so a CLS-off
    assembly change that still assembles cleanly produces an identical golden.
    The "the CLS-off assembly is byte-identical to before" claims in the commit
    messages were established by a human running a diff, with no automated
    successor. This is that successor.

    Why a fingerprint and not a text snapshot: the CLS-off subtile store is
    1100-2900 instructions per config, so a full text snapshot would be ~12k
    lines of golden that no reviewer can read and that churns on any unrelated
    store-emitter change. The fingerprint keeps the diagnostic value where it is
    useful -- a per-mnemonic histogram a reviewer can diff, plus an instruction
    count -- and puts the exactness in a sha256 of the canonical text, so nothing
    slips through.

    REGENERATING (deliberately, after reviewing what moved):

        cd projects/hipblaslt/tensilelite
        CLS_GOLDEN_UPDATE=1 python -m pytest \\
            Tensile/Tests/unit/test_cls_gfx950_subtile_codegen.py \\
            -k ClsOffStoreGolden -q

    That rewrites test_data/cls_gfx950_cls_off_store.golden.json. Read the diff
    before committing it: an opcode delta on the CLS-off path means the
    production store changed for kernels that never enable CLS.
    """

    @pytest.mark.parametrize("use_bf16", [False, True], ids=["f32", "bf16"])
    @pytest.mark.parametrize("mt_a,mt_b,depth_u", GOLDEN_CONFIGS,
                             ids=[f"{a}x{b}" for a, b, _ in GOLDEN_CONFIGS])
    def test_cls_off_assembly_matches_the_golden(self, mt_a, mt_b, depth_u, use_bf16):
        key = "%dx%d-%s" % (mt_a, mt_b, "bf16" if use_bf16 else "f32")
        asm, _ = _store_asm(mt_a, mt_b, depth_u, cls=False, use_bf16=use_bf16)
        actual = _asm_fingerprint(asm)

        if os.environ.get(GOLDEN_UPDATE_ENV):
            golden = _load_golden() if os.path.exists(GOLDEN_PATH) else {}
            golden[key] = actual
            with open(GOLDEN_PATH, "w") as fh:
                json.dump(golden, fh, indent=2, sort_keys=True)
                fh.write("\n")
            pytest.skip("%s set: rewrote %s[%s]" % (GOLDEN_UPDATE_ENV,
                                                    os.path.basename(GOLDEN_PATH), key))

        golden = _load_golden()
        assert key in golden, (
            "no CLS-off golden for %s; regenerate with %s=1 (see the class "
            "docstring)" % (key, GOLDEN_UPDATE_ENV))
        expected = golden[key]
        assert actual == expected, (
            "CLS-off subtile store assembly changed for %s: %s.\n"
            "CLS-off is the production path (every shipped logic file has "
            "CompactLoopStore: false), so this is a change to kernels that never "
            "enable CLS. If it is intended, regenerate with %s=1 and review the "
            "golden diff." % (key, _describe_fingerprint_diff(expected, actual),
                              GOLDEN_UPDATE_ENV))

    def test_golden_covers_every_config_and_nothing_else(self):
        """The golden file has exactly one entry per parametrization.

        A golden that silently loses an entry degrades to "no coverage for that
        config" without failing anything, which is the failure mode this whole
        file is trying to stop repeating.
        """
        expected = {"%dx%d-%s" % (a, b, dt)
                    for a, b, _ in GOLDEN_CONFIGS for dt in ("f32", "bf16")}
        assert set(_load_golden()) == expected

    def test_golden_fingerprint_is_sensitive_to_a_single_instruction(self):
        """The fingerprint would actually notice a one-instruction perturbation.

        A golden nobody has checked can be a golden of the wrong thing. This
        proves the comparison is live by perturbing the rendered text rather than
        the emitter: dropping one instruction must move all three levels.
        """
        asm, _ = _store_asm(*COMPACTING_CONFIGS[0], cls=False, use_bf16=False)
        lines = asm.splitlines()
        victim = next(i for i, ln in enumerate(lines)
                      if ln.strip().startswith("v_") and "//" in ln)
        perturbed = "\n".join(lines[:victim] + lines[victim + 1:])
        base, changed = _asm_fingerprint(asm), _asm_fingerprint(perturbed)
        assert changed["digest"] != base["digest"]
        assert changed["instructions"] == base["instructions"] - 1
        assert changed["opcodes"] != base["opcodes"]
        assert "instruction count" in _describe_fingerprint_diff(base, changed)


class TestPreconditionSemantics:
    """The graded precondition logic itself, exercised without a toolchain.

    _classify_assembler encodes the judgement call about which failures are
    legitimately unsupported and which are broken installs, and that decision is
    what the old single skipif got wrong. Pinning it here means the semantics are
    a tested property rather than a comment.
    """

    def test_working_assembler_is_active(self):
        assert _classify_assembler(has_amdclang=True, init_error=None)[0] == "active"

    def test_missing_assembler_is_unsupported_not_broken(self):
        """No amdclang++ at all: a genuine platform limit, so a skip is right."""
        state, reason = _classify_assembler(has_amdclang=False,
                                            init_error="RuntimeError: no assembler")
        assert state == "unsupported"
        assert "amdclang++ not found" in reason
        assert "no assembler" in reason, "the underlying error must survive into the skip reason"

    def test_present_but_failing_assembler_is_broken(self):
        """amdclang++ exists and init still failed: broken toolchain, be loud."""
        state, reason = _classify_assembler(has_amdclang=True,
                                            init_error="OSError: bad probe")
        assert state == "broken"
        assert "broken toolchain" in reason

    def test_no_assembler_but_init_worked_is_still_active(self):
        """The probe result wins over the heuristic: if it initialised, it works."""
        assert _classify_assembler(has_amdclang=False, init_error=None)[0] == "active"

    def test_missing_cls_bindings_are_reported_not_swallowed(self):
        """A rocisa without SSetGprIdxOn produces an error string, never None.

        The import-time guard turns that string into a RuntimeError; this pins the
        detection half, which is what has to notice a stale rocisa in the first
        place.
        """
        import types
        stale = types.ModuleType("rocisa.instruction")
        stale.__file__ = "/stale/rocisa/instruction.so"
        module, error = _import_cls_bindings(stale)
        assert module is None
        assert "SSetGprIdxOn" in error and "/stale/rocisa" in error

    def test_partially_stale_bindings_are_reported(self):
        """One of the two bindings missing is just as broken as both."""
        import types
        from rocisa.instruction import SSetGprIdxOn
        half = types.ModuleType("rocisa.instruction")
        half.__file__ = "/half/rocisa/instruction.so"
        half.SSetGprIdxOn = SSetGprIdxOn
        module, error = _import_cls_bindings(half)
        assert module is None
        assert "SSetGprIdxOff" in error and "SSetGprIdxOn" not in error

    def test_real_bindings_are_detected(self):
        module, error = _import_cls_bindings()
        assert error is None and module is not None
