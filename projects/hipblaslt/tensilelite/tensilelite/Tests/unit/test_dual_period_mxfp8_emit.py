################################################################################
# ISA-level regression for the dual-period MXFP8 subtile kernel.
#
# The dual-period path keeps the 128-K data loop while running a decoupled
# 256-K pre-shuffled scale epoch that is reused by two data iterations.  This
# test drives the REAL winning geometry (MT256x256, MIWT 8x4, MIWG 2x4, DU128,
# PGR2/PLR0, 4x2 partitions, N-major, WGM4) through CPU-only codegen and
# asserts the steady-state mainloop honors the scale contract:
#
#   * exactly SIX scale ds_read_b32 per 256-K epoch (4 SA groups + 2 SB groups)
#   * both K phases consumed by the MFMAs (K=0 for data iter 0, K=1 for iter 1)
#   * every scale GR / SRD-advance / LDS-swap fires exactly once per epoch
#   * architectural VGPR stays within the two-wave-slot budget (<= 128)
#
# It needs the gfx950 assembly toolchain (amdclang++); it skips otherwise.
################################################################################

import re
import shutil

import pytest


def _emit_dual_period_kernel():
    """Build the real dual-period MXFP8 solution and return its assembly text.

    Partition-directed PLR (next-partition prefetch, n-major traversal, 2x2
    partition grid) is selected automatically from the dual-period MXFP8 solution
    parameters -- no environment knobs are required.
    """
    from tensilelite.Common.Architectures import gfxToIsa
    from tensilelite.Common.Capabilities import makeIsaInfoMap
    from tensilelite.Toolchain.Assembly import makeAssemblyToolchain
    from tensilelite.Toolchain.Validators import validateToolchain, ToolchainDefaults
    from tensilelite.Common.GlobalParameters import assignGlobalParameters
    from tensilelite.SolutionStructs.Solution import Solution
    from tensilelite.SolutionStructs.Validators.MatrixInstruction import (
        matrixInstructionToMIParameters,
    )

    cxx = validateToolchain("amdclang++")
    if not cxx or not shutil.which(cxx):
        pytest.skip("gfx950 assembly toolchain (amdclang++) unavailable")

    isa = gfxToIsa("gfx950")
    iim = makeIsaInfoMap([isa], cxx)
    bundler = validateToolchain(ToolchainDefaults.OFFLOAD_BUNDLER)
    asm = makeAssemblyToolchain(cxx, bundler, "default").assembler
    assignGlobalParameters({}, iim)

    mi = [16, 16, 128, 1, 1, 8, 4, 2, 4]
    pt = {"OperationType": "GEMM", "DataType": "F8", "DestDataType": "b",
          "ComputeDataType": "s", "HighPrecisionAccumulate": True,
          "TransposeA": True, "TransposeB": False, "UseBeta": True,
          "Batched": True, "StridedBatched": True, "MXBlockA": 32, "MXBlockB": 32}
    params = {"ProblemType": pt, "ISA": isa, "MatrixInstruction": mi,
        "WorkGroup": [16, 16, 1], "WavefrontSize": 64, "DepthU": 128,
        "KernelLanguage": "Assembly", "PrefetchGlobalRead": 2, "PrefetchLocalRead": 0,
        "ScheduleIterAlg": 3, "StaggerU": 0, "GlobalSplitU": 1, "InnerUnroll": 1,
        "LdsPadA": -1, "LdsPadB": -1, "LdsBlockSizePerPadA": -1, "LdsBlockSizePerPadB": -1,
        "1LDSBuffer": 0, "VectorWidthA": -1, "VectorWidthB": -1, "StoreVectorWidth": -1,
        "GlobalReadVectorWidthA": -1, "GlobalReadVectorWidthB": -1, "LocalReadVectorWidth": -1,
        "SourceSwap": False, "ExpandPointerSwap": False,
        "GlobalSplitUAlgorithm": "MultipleBuffer",
        "DirectToLds": 1, "StreamK": 0, "PrefetchAcrossPersistent": 0,
        "UseSubtileImpl": True, "StoreRemapVectorWidth": 0,
        "DirectToVgprA": False, "DirectToVgprB": False,
        "DirectToVgprSparseMetadata": False, "WorkGroupMapping": 4}
    params.update(matrixInstructionToMIParameters(
        mi, isa, 64, pt, params["WorkGroup"], iim))
    sol = Solution(params, False, True, False, asm, iim)
    assert sol.get("Valid"), "dual-period MXFP8 solution did not validate"
    assert sol.get("_DualPeriodMXFP8"), "solution did not take the dual-period path"

    from tensilelite.TensileCreateLibrary.Run import (
        generateKernelObjectsFromSolutions, processKernelSource)
    from tensilelite.KernelWriterAssembly import KernelWriterAssembly
    from tensilelite.Common.Types import DebugConfig
    import rocisa

    kernels = generateKernelObjectsFromSolutions([sol])
    kwa = KernelWriterAssembly(asm, DebugConfig())
    for kernel in kernels:
        ri = rocisa.rocIsa.getInstance()
        ri.init(tuple(kernel["ISA"]), shutil.which(cxx))
        ri.setKernel(tuple(kernel["ISA"]), kernel["WavefrontSize"])
        kernel["BaseName"] = "dual_period_probe"
        try:
            kernel.duplicate = False
        except Exception:
            pass
        res = processKernelSource(
            kwa, ri.getData(), ri.getOutputOptions(), False, kernel)
        assert res.err == 0, "kernel codegen failed (err=%s)" % res.err
        src = res.src
        if isinstance(src, (bytes, bytearray)):
            src = src.decode(errors="replace")
        return src
    pytest.skip("no kernel object generated")


def _mainloop_lines(src):
    """Return one steady-state mainloop epoch through its back-branch."""
    lines = src.splitlines()
    loopback_label = (
        "label_LoopBackC0ScaleRefreshL:"
        if "label_LoopBackC0ScaleRefreshL:" in src
        else "label_LoopBeginL:"
    )
    loopback_branch = (
        "s_cbranch_scc0 label_LoopBackC0ScaleRefreshL"
        if "label_LoopBackC0ScaleRefreshL:" in src
        else "s_cbranch_scc0 label_LoopBeginL"
    )
    start = end = None
    for i, ln in enumerate(lines):
        if start is None and ln.strip() == loopback_label:
            start = i
        elif start is not None and loopback_branch in ln:
            end = i
            break
    assert start is not None and end is not None, "could not locate mainloop body"
    return lines[start:end + 1]


def _section_lines(src, label):
    begin = "/* %s start */" % label
    end = "/* %s end */" % label
    assert begin in src and end in src, "missing %s section" % label
    return src[src.index(begin) + len(begin):src.index(end)].splitlines()


@pytest.fixture(scope="module")
def dual_src():
    return _emit_dual_period_kernel()


def test_six_scale_reads_per_epoch(dual_src):
    body = _mainloop_lines(dual_src)
    reads = [
        ln for ln in body
        if re.search(r"^\s*ds_read_b32\b.*// S[AB] MT=", ln)
    ]
    assert len(reads) == 6, (
        "expected exactly six scale ds_reads per 256-K epoch, got %d:\n%s"
        % (len(reads), "\n".join(reads)))
    # LR granularity is two MFMA tiles: four SA groups and two SB groups.
    sa = sorted(int(re.search(r"// SA MT=.* tile=(\d+)", r).group(1))
                for r in reads if "// SA MT=" in r)
    sb = sorted(int(re.search(r"// SB MT=.* tile=(\d+)", r).group(1))
                for r in reads if "// SB MT=" in r)
    assert sa == [0, 2, 4, 6], "scale-A tile groups: %s" % sa
    assert sb == [0, 2], "scale-B tile groups: %s" % sb


def test_tail_scale_refresh_moves_from_c1_to_the_loopback_c0(dual_src):
    c0 = _section_lines(dual_src, "MAINLOOP_C0")
    c1 = _section_lines(dual_src, "MAINLOOP_C1")
    c0_refreshes = [
        line for line in c0
        if "late scale refresh" in line
    ]
    assert len(c0_refreshes) == 2
    assert any("// SA MT=n+2 tile=6" in line for line in c0_refreshes)
    assert any("// SB MT=n+2 tile=2" in line for line in c0_refreshes)

    refreshes = [
        (index, line)
        for index, line in enumerate(c1)
        if "late scale refresh" in line
    ]
    assert len(refreshes) == 4

    uses = {}
    for index, line in enumerate(c1):
        if "v_mfma_scale" not in line:
            continue
        match = re.search(r", (v\d+), (v\d+) op_sel:", line)
        assert match, line
        for register in match.groups():
            uses.setdefault(register, []).append(index)

    staged_registers = {
        re.search(r"ds_read_b32 (v\d+),", line).group(1)
        for _, line in refreshes
        if "via reusable staging VGPR" in line
    }
    assert len(staged_registers) == 1
    staging_register = next(iter(staged_registers))
    for refresh_index, line in refreshes:
        destination = re.search(r"ds_read_b32 (v\d+),", line)
        assert destination, line
        destination = destination.group(1)
        if destination == staging_register:
            publish_index = next(
                index
                for index, candidate in enumerate(
                    c1[refresh_index + 1:], refresh_index + 1)
                if re.search(
                    rf"v_mov_b32 v\d+, {staging_register}\b",
                    candidate)
            )
            publish = re.search(
                rf"v_mov_b32 (v\d+), {staging_register}\b",
                c1[publish_index])
            assert any(
                "s_waitcnt lgkmcnt" in candidate
                and "staged scale value" in candidate
                for candidate in c1[refresh_index:publish_index]
            )
            destination = publish.group(1)
            refresh_index = publish_index
        assert max(uses[destination]) < refresh_index

    first_refresh = refreshes[0][0]
    producer_wait = next(
        index for index, line in enumerate(c1[:first_refresh])
        if "s_waitcnt vmcnt" in line
    )
    producer_barrier = next(
        index for index in range(producer_wait + 1, first_refresh)
        if "s_barrier" in c1[index]
    )
    assert producer_wait < producer_barrier < first_refresh
    assert sum("swap MXSA LR vgpr" in line for line in c0) == 1
    assert sum("swap MXSB LR vgpr" in line for line in c0) == 1
    assert not any(
        "swap MXS" in line and "LR vgpr" in line
        for line in c1)

    assert sum(
        "late scale refresh" in line
        for line in _section_lines(dual_src, "NGLL_C0")
    ) == 6
    assert not any(
        "late scale refresh" in line
        for label in ("NGLL_C1", "NLL_C0", "NLL_C1")
        for line in _section_lines(dual_src, label)
    )


def test_both_scale_phases_consumed(dual_src):
    body = _mainloop_lines(dual_src)
    mfmas = [ln for ln in body if "v_mfma_scale" in ln]
    phases = {}
    for ln in mfmas:
        m = re.search(r"A\[\d+,K=(\d+)\]", ln)
        if m:
            phases[m.group(1)] = phases.get(m.group(1), 0) + 1
    assert set(phases) == {"0", "1"}, "MFMA scale phases present: %s" % phases
    # Symmetric: scaleReuse==2 data iterations => equal MFMA counts per phase.
    assert phases["0"] == phases["1"], "phase MFMA counts asymmetric: %s" % phases
    # Phase 1 MFMAs must select the high scale byte (op_sel_hi:[1,1]).
    for ln in mfmas:
        if re.search(r"A\[\d+,K=1\]", ln):
            assert "op_sel_hi:[1,1]" in ln, "phase-1 MFMA missing hi byte-sel: %s" % ln


def test_tail_dispatches_to_the_completed_depthu_scale_phase(dual_src):
    dispatch = next(
        ln for ln in dual_src.splitlines()
        if "dual-period tail: select completed-DepthU parity" in ln)
    assert dispatch.lstrip().startswith("s_bitcmp1_b32")
    assert re.search(r",\s*7(?:\s|//)", dispatch), dispatch

    p0_start = dual_src.index("/* TAILLOOP_PHASE0 start */")
    p0_end = dual_src.index("/* TAILLOOP_PHASE0 end */", p0_start)
    p1_start = dual_src.index("/* TAILLOOP_PHASE1 start */", p0_end)
    p1_end = dual_src.index("/* TAILLOOP_PHASE1 end */", p1_start)
    phase0 = dual_src[p0_start:p0_end]
    phase1 = dual_src[p1_start:p1_end]
    phase0_mfmas = [ln for ln in phase0.splitlines() if "v_mfma_scale" in ln]
    phase1_mfmas = [ln for ln in phase1.splitlines() if "v_mfma_scale" in ln]
    assert phase0_mfmas and phase1_mfmas
    assert all("K=0]" in ln for ln in phase0_mfmas)
    assert all("K=1]" in ln for ln in phase1_mfmas)
    assert all("op_sel_hi:[1,1]" in ln for ln in phase1_mfmas)

    # The phase override must affect only MFMA byte selection.  Both standalone
    # tail bodies still need the complete PGR0 scale load/read cadence.
    for phase_body in (phase0, phase1):
        assert len(re.findall(
            r"^\s*ds_read_b32\b.*// S[AB] MT=", phase_body,
            flags=re.MULTILINE)) == 6
        assert len([
            ln for ln in phase_body.splitlines()
            if re.search(r"^\s*buffer_load_dwordx4\b", ln)
            and ("sgprSrdMXSA" in ln or "sgprSrdMXSB" in ln)
            and re.search(r"\blds\b", ln)
        ]) == 2


def test_odd_phase_tail_rewinds_scale_srds_with_their_bounds(dual_src):
    align_start = dual_src.index(
        "dual-period tail: test completed-DepthU parity for scale SRDs")
    align_end = dual_src.index("label_SubtileTailScaleSrdAligned:", align_start)
    tail_start = dual_src.index("/* TAILLOOP */", align_end)
    alignment = dual_src[align_start:align_end]

    parity_test = next(
        ln for ln in dual_src.splitlines()
        if "dual-period tail: test completed-DepthU parity for scale SRDs" in ln)
    assert parity_test.lstrip().startswith("s_bitcmp1_b32")
    assert re.search(r",\s*7(?:\s|//)", parity_test), parity_test
    assert "s_cbranch_scc0 label_SubtileTailScaleSrdAligned" in alignment

    for tc in ("MXSA", "MXSB"):
        rewind = next(
            ln for ln in alignment.splitlines()
            if f"{tc}: rewind to odd-DU scale epoch" in ln)
        borrow = next(
            ln for ln in alignment.splitlines()
            if f"{tc}: scale SRD rewind borrow" in ln)
        restore = next(
            ln for ln in alignment.splitlines()
            if f"{tc}: restore rewound num_records" in ln)
        assert rewind.lstrip().startswith("s_sub_u32")
        assert "256" in rewind
        assert borrow.lstrip().startswith("s_subb_u32")
        assert restore.lstrip().startswith("s_add_u32")
        assert "256" in restore

    assert align_start < align_end < tail_start


def test_scale_cadence_once_per_epoch(dual_src):
    body = "\n".join(_mainloop_lines(dual_src))
    # One SRD advance, one DTL load, one GR write-half swap and one LR read-half
    # swap for EACH of the two scale tensors (MXSA, MXSB) per 256-K epoch.
    assert body.count("Scale SRD update") == 2
    scale_dtl = [
        ln for ln in body.splitlines()
        if re.search(r"^\s*buffer_load_dwordx4\b", ln)
        and ("sgprSrdMXSA" in ln or "sgprSrdMXSB" in ln)
        and re.search(r"\blds\b", ln)
    ]
    assert len(scale_dtl) == 2, "scale DTL loads: %s" % scale_dtl
    assert sum("sgprSrdMXSA" in ln for ln in scale_dtl) == 1
    assert sum("sgprSrdMXSB" in ln for ln in scale_dtl) == 1
    assert len(re.findall(r"swap MXS\w+ GR m0", body)) == 2
    assert len(re.findall(r"swap MXS\w+ LR vgpr", body)) == 2


def test_vgpr_within_two_wave_slot_budget(dual_src):
    m = re.search(r"\.vgpr_count:\s*(\d+)", dual_src)
    assert m, "could not find .vgpr_count in emitted assembly"
    vgpr = int(m.group(1))
    assert vgpr <= 128, (
        "architectural VGPR %d exceeds the two-wave-slot budget (128)" % vgpr)


def test_preloop_seeds_complete_scale_e0_once(dual_src):
    """The preloop seeds every scale VGPR needed by the first C0 body."""
    preloop = []
    for ln in dual_src.splitlines():
        if ln.strip() == "label_LoopBeginL:":
            break
        preloop.append(ln)

    scale_lr = [
        ln for ln in preloop
        if re.search(r"^\s*ds_read_b32\b.*// S[AB] MT=", ln)
    ]
    sa = sorted(int(re.search(r"// SA MT=.* tile=(\d+)", ln).group(1))
                for ln in scale_lr if "// SA MT=" in ln)
    sb = sorted(int(re.search(r"// SB MT=.* tile=(\d+)", ln).group(1))
                for ln in scale_lr if "// SB MT=" in ln)
    assert sa == [0, 2, 4, 6]
    assert sb == [0, 2]

    data_lr = [
        l for l in preloop
        if re.search(r"^\s*ds_read_\w+\b.*// [AB](?: MT=)?", l)
    ]
    assert data_lr, "pre-loop lost its data operand prefetch"
