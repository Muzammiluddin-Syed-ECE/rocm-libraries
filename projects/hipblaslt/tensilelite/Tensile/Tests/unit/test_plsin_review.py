################################################################################
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
################################################################################

from copy import deepcopy
from math import log2
import re
import shutil
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from rocisa.code import Module
from rocisa.container import MUBUFModifiers, vgpr, sgpr
from rocisa.instruction import (
    BufferStoreB128,
    DSBPermuteB32,
    SNop,
    SWaitCnt,
    VAddU32,
    VCvtPkF32toBF16,
    VCvtPkF32toFP16,
    VLShiftRightB32,
    VPermlane32SwapB32,
)
from Tensile.Common import DataDirection
from Tensile.Components.GlobalWriteBatch import (
    GlobalWriteBatchWriter,
    _plsinStoreGate,
)
from Tensile.KernelWriterAssembly import KernelWriterAssembly
from Tensile.SolutionStructs.Solution import Solution

pytestmark = pytest.mark.unit


def _dtype(*, float4=False, bf16=False, half=False):
    dtype = MagicMock()
    dtype.isFloat4.return_value = float4
    dtype.isBFloat16.return_value = bf16
    dtype.isHalf.return_value = half
    return dtype


def _eligible_state(*, depth_u=256, pgr=2, dest="bf16", stream_k=0,
                    stream_k_atomic=0):
    """Smallest realistic state consumed by assignPostLoopStoreInNll."""
    dest_type = _dtype(bf16=dest == "bf16", half=dest == "half")
    return {
        "PostLoopStoreInNll": True,
        "PLSINStoreMode": "Weave",
        "ISA": [9, 5, 0],
        "DepthU": depth_u,
        "UseSubtileImpl": True,
        "EnableMatrixInstruction": True,
        "PrefetchGlobalRead": pgr,
        "BufferStore": True,
        "WavefrontSize": 64,
        "StoreRemapVectorWidth": 0,
        "StreamK": stream_k,
        "StreamKAtomic": stream_k_atomic,
        "_GlobalAccumulation": None,
        "MIWaveTile": [8, 8],
        "ProblemType": {
            "DataTypeA": _dtype(float4=True),
            "DataTypeB": _dtype(float4=True),
            "DestDataType": dest_type,
            "HighPrecisionAccumulate": True,
            "MXBlockA": 32,
            "MXBlockB": 32,
        },
    }


@pytest.mark.parametrize("dest", ["bf16", "half"])
@pytest.mark.parametrize("pgr", [1, 2])
def test_plsin_accepts_supported_destination_and_pgr(dest, pgr, monkeypatch):
    monkeypatch.delenv("TENSILE_PLSIN_DEBUG", raising=False)
    state = _eligible_state(dest=dest, pgr=pgr)

    Solution.assignPostLoopStoreInNll(state)

    assert state["PostLoopStoreInNll"] is True


@pytest.mark.parametrize("depth_u", [0, 1, 48, 96, 768])
def test_plsin_rejects_non_power_of_two_depthu(depth_u, monkeypatch):
    monkeypatch.delenv("TENSILE_PLSIN_DEBUG", raising=False)
    state = _eligible_state(depth_u=depth_u)

    Solution.assignPostLoopStoreInNll(state)

    assert state["PostLoopStoreInNll"] is False
    assert state["PLSINStoreMode"] == "Weave"


@pytest.mark.parametrize("depth_u", [2, 64, 128, 256, 1024])
def test_plsin_accepts_power_of_two_depthu(depth_u, monkeypatch):
    monkeypatch.delenv("TENSILE_PLSIN_DEBUG", raising=False)
    state = _eligible_state(depth_u=depth_u)

    Solution.assignPostLoopStoreInNll(state)

    assert state["PostLoopStoreInNll"] is True


@pytest.mark.parametrize("depth_u", [0, 1, 768])
def test_plsin_emitters_defensively_reject_non_power_of_two_depthu(depth_u):
    writer = object.__new__(KernelWriterAssembly)
    compute_type = MagicMock()
    compute_type.isSingle.return_value = True
    writer.states = SimpleNamespace()
    kernel = {
        "DepthU": depth_u,
        "PostLoopStoreInNll": True,
        "ProblemType": {"ComputeDataType": compute_type},
    }

    with pytest.raises(AssertionError, match="positive power of two"):
        writer.computePostLoopFusedStore(kernel)
    with pytest.raises(AssertionError, match="positive power of two"):
        writer.emitFusedStoreGuard(kernel, targetLabel=None)


def test_plsin_rejects_missing_structural_nll(monkeypatch):
    monkeypatch.delenv("TENSILE_PLSIN_DEBUG", raising=False)
    state = _eligible_state(pgr=0)

    Solution.assignPostLoopStoreInNll(state)

    assert state["PostLoopStoreInNll"] is False


@pytest.mark.parametrize(
    "stream_k,stream_k_atomic,expected",
    [
        (0, 0, True),
        (3, 0, True),
        (3, 1, False),
    ],
)
def test_plsin_streamk_eligibility(stream_k, stream_k_atomic, expected, monkeypatch):
    monkeypatch.delenv("TENSILE_PLSIN_DEBUG", raising=False)
    state = _eligible_state(stream_k=stream_k, stream_k_atomic=stream_k_atomic)

    Solution.assignPostLoopStoreInNll(state)

    assert state["PostLoopStoreInNll"] is expected


def test_plsin_does_not_reenable_explicitly_disabled_solution(monkeypatch):
    monkeypatch.delenv("TENSILE_PLSIN_DEBUG", raising=False)
    state = deepcopy(_eligible_state())
    state["PostLoopStoreInNll"] = False
    state["PLSINStoreMode"] = "Lend"

    Solution.assignPostLoopStoreInNll(state)

    assert state["PostLoopStoreInNll"] is False
    assert state["PLSINStoreMode"] == "Weave"


def test_fused_epilogue_resource_plan_covers_rich_epilogue():
    writer = object.__new__(KernelWriterAssembly)
    writer.states = SimpleNamespace(
        useBias=DataDirection.READ,
        useGateResidual=True,
    )
    kernel = {
        "ProblemType": {
            "UseScaleAB": "Vector",
            "UseScaleAlphaVec": 1,
            "UseE": True,
            "UseScaleCD": True,
        },
    }

    specs = writer._fusedEpilogueResourceSpecs(kernel)

    assert specs == [
        ("SrdScaleA", 4, 4),
        ("SrdScaleB", 4, 4),
        ("SrdScaleAlphaVec", 4, 4),
        ("SrdBias", 4, 4),
        ("SrdGate", 4, 4),
        ("GateNullOne", 2, 2),
        ("SrdE", 4, 4),
        ("ScaleD", 2, 2),
    ]


def test_fused_epilogue_resource_plan_is_empty_for_plain_store():
    writer = object.__new__(KernelWriterAssembly)
    writer.states = SimpleNamespace(
        useBias=DataDirection.NONE,
        useGateResidual=False,
    )
    kernel = {
        "ProblemType": {
            "UseScaleAB": "",
            "UseScaleAlphaVec": 0,
            "UseE": False,
            "UseScaleCD": False,
        },
    }

    assert writer._fusedEpilogueResourceSpecs(kernel) == []


def test_vgpr_lending_tolerates_already_released_tile():
    writer = object.__new__(KernelWriterAssembly)
    writer.vgprPool = MagicMock()
    writer.vgprPool.addFromCheckOut.side_effect = RuntimeError(
        "RegisterPool::addFromCheckOut(16) but it was never checked out"
    )

    assert writer._lendSubtileFusedVgpr(16) is False


def test_vgpr_lending_propagates_invalid_pool_state():
    writer = object.__new__(KernelWriterAssembly)
    writer.vgprPool = MagicMock()
    writer.vgprPool.addFromCheckOut.side_effect = RuntimeError(
        "RegisterPool::addFromCheckOut(16) is not in InUse state"
    )

    with pytest.raises(RuntimeError, match="not in InUse state"):
        writer._lendSubtileFusedVgpr(16)


def test_plsin_store_gates_are_evaluated_at_emit_time(monkeypatch):
    monkeypatch.delenv("TENSILE_PLSIN_DEBUG", raising=False)
    assert _plsinStoreGate("PLSIN_STORE_HOIST_ADDR", default=True) is True

    monkeypatch.setenv("TENSILE_PLSIN_DEBUG", "PLSIN_STORE_HOIST_ADDR=0")
    assert _plsinStoreGate("PLSIN_STORE_HOIST_ADDR", default=True) is False

    monkeypatch.setenv("TENSILE_PLSIN_DEBUG", "PLSIN_STORE_HOIST_ADDR=1")
    assert _plsinStoreGate("PLSIN_STORE_HOIST_ADDR", default=True) is True


def test_weave_epilogue_reads_advance_readiness_without_advancing_store_counter():
    writer = object.__new__(GlobalWriteBatchWriter)
    writer.batchElements = [
        (0, tt0, 0, 0)
        for tt0 in range(8)
    ]
    writer.parentWriter = SimpleNamespace(
        states=SimpleNamespace(subtileWeavePairCounter=5)
    )
    writer._weaveEpiloguePairCounter = None
    writer._weaveEmitReady = MagicMock()
    writer._weaveEmitGroup = MagicMock()
    writer._weaveEmitAll = MagicMock()
    writer._weaveLookahead = lambda: 2
    writer._popSubtileAccVgprReads = lambda element_idx: Module(
        f"read{element_idx}"
    )

    for element_idx in range(len(writer.batchElements)):
        writer._weaveReadForEpilogue(Module("out"), element_idx)

    ready_pairs = [
        args[1]
        for args, _kwargs in writer._weaveEmitReady.call_args_list
    ]
    assert ready_pairs == [5, 6, 7, 8]
    assert writer._weaveEpiloguePairCounter == 9
    assert writer.parentWriter.states.subtileWeavePairCounter == 5
    lookahead_pairs = [
        args[1]
        for args, _kwargs in writer._weaveEmitGroup.call_args_list
    ]
    assert lookahead_pairs == [7, 8, 9, 10]
    writer._weaveEmitAll.assert_not_called()


def _legacy_paired_store_issue(
    writer, v_pack, v_addr_scratch, addr_calc,
    sum_idx0, sum_idx1, prefix_offset, tt0,
):
    """Pre-refactor pipeline issue emitter retained only as a byte-stability oracle."""
    module = Module("pairedStoreIssue")
    is_fp16 = writer.kernel["ProblemType"]["DestDataType"].isHalf()
    cvt_pk_f32_to_16 = VCvtPkF32toFP16 if is_fp16 else VCvtPkF32toBF16
    type_str = "fp16" if is_fp16 else "bf16"
    v_perm_addr = writer.cvtVgprStruct.vgprPermAddr
    v_lg_delta = writer.cvtVgprStruct.vgprLaneGroupDelta
    addr_d_vgpr = addr_calc.addrDVgpr

    def vc(sum_idx, vi):
        return vgpr("ValuC+" + str(sum_idx + vi - prefix_offset))

    module.addComment1(
        f"[pipeline] ISSUE tt0={tt0}: pack 8 f32 -> 4 {type_str} dwords "
        f"+ ds_bpermute (buf v{v_pack})"
    )
    module.add(cvt_pk_f32_to_16(
        dst=vgpr(v_pack + 0), src0=vc(sum_idx0, 0), src1=vc(sum_idx0, 1),
        comment=f"sba=0 tt0={tt0}[0:1] -> {type_str}"))
    module.add(cvt_pk_f32_to_16(
        dst=vgpr(v_pack + 1), src0=vc(sum_idx0, 2), src1=vc(sum_idx0, 3),
        comment=f"sba=0 tt0={tt0}[2:3] -> {type_str}"))
    module.add(cvt_pk_f32_to_16(
        dst=vgpr(v_pack + 2), src0=vc(sum_idx1, 0), src1=vc(sum_idx1, 1),
        comment=f"sba=1 tt0={tt0}[0:1] -> {type_str}"))
    module.add(cvt_pk_f32_to_16(
        dst=vgpr(v_pack + 3), src0=vc(sum_idx1, 2), src1=vc(sum_idx1, 3),
        comment=f"sba=1 tt0={tt0}[2:3] -> {type_str}"))

    module.addComment1(
        "ds_bpermute in-place: gather packed dwords from partner lane-group")
    for k in range(4):
        module.add(DSBPermuteB32(
            dst=vgpr(v_pack + k), src0=vgpr(v_perm_addr),
            src1=vgpr(v_pack + k), comment=f"perm dword {k}"))

    bpe_curr = writer.parentWriter.states.bpeCexternal
    bpe_dest = writer.parentWriter.states.bpeCexternalGSU1
    global_offset = addr_calc.globalOffset * bpe_dest // bpe_curr
    addr_scale_shift = int(log2(bpe_curr // bpe_dest)) \
        if bpe_curr > bpe_dest else 0
    if addr_scale_shift:
        module.add(VLShiftRightB32(
            dst=vgpr(v_addr_scratch), shiftHex=addr_scale_shift,
            src=vgpr(addr_d_vgpr),
            comment=f"scale addrDVgpr bpe {bpe_curr}->{bpe_dest}"))
        module.add(VAddU32(
            dst=vgpr(v_addr_scratch), src0=vgpr(v_addr_scratch),
            src1=vgpr(v_lg_delta),
            comment="adjusted D addr = scaled addrDVgpr + lane_group*8"))
    else:
        module.add(VAddU32(
            dst=vgpr(v_addr_scratch), src0=vgpr(addr_d_vgpr),
            src1=vgpr(v_lg_delta),
            comment="adjusted D addr = addrDVgpr + lane_group*8"))
    return module, global_offset


def _legacy_paired_store_commit(
    writer, v_pack, v_addr_scratch, global_offset, tt0, dscnt,
):
    """Pre-refactor pipeline commit emitter retained only as a byte-stability oracle."""
    module = Module("pairedStoreCommit")
    ntd = writer.kernel["NonTemporalD"]
    module.add(SWaitCnt(
        dscnt=dscnt,
        comment=f"[pipeline] wait this group's ds_bpermute; "
                f"{dscnt} younger ds in flight (tt0={tt0})"))
    module.addComment1("v_permlane32_swap_b32: swap across lane-32 boundary")
    module.add(VPermlane32SwapB32(
        dst=vgpr(v_pack + 0), src=vgpr(v_pack + 2),
        comment="swap dwords 0<->2"))
    module.add(VPermlane32SwapB32(
        dst=vgpr(v_pack + 1), src=vgpr(v_pack + 3),
        comment="swap dwords 1<->3"))
    module.addComment1(
        "buffer_store_dwordx4: write 8 16bit values (4 dwords, 2-aligned src)")
    module.add(BufferStoreB128(
        src=vgpr(v_pack, 4),
        vaddr=vgpr(v_addr_scratch),
        saddr=sgpr("SrdD", 4),
        soffset=0,
        mubuf=MUBUFModifiers(
            offen=True,
            offset12=global_offset,
            glc=bool(ntd & 0x1),
            slc=bool(ntd & 0x2),
            nt=bool(ntd & 0x4),
        ),
        comment=f"[pipeline] 16bit paired dwordx4 store tt0={tt0},{tt0 + 1}",
    ))
    module.add(SNop(
        waitState=0,
        comment="1 wait state: WAR store src -> next same-buffer pack dst"))
    return module


@pytest.mark.parametrize("full_tile", [False, True])
def test_paired_store_pipeline_refactor_is_assembly_identical(
    monkeypatch, full_tile,
):
    """The pipeline stays legacy-identical even if its caller looks fused/full-tile."""
    monkeypatch.delenv("TENSILE_PLSIN_DEBUG", raising=False)
    writer = object.__new__(GlobalWriteBatchWriter)
    writer.kernel = {
        "ProblemType": {"DestDataType": _dtype(bf16=True)},
        "NonTemporalD": 4,
    }
    writer.parentWriter = SimpleNamespace(states=SimpleNamespace(
        bpeCexternal=4,
        bpeCexternalGSU1=2,
        storeAlign8=False,
    ))
    writer.cvtVgprStruct = SimpleNamespace(
        vgprPermAddr=40,
        vgprLaneGroupDelta=41,
        vgprBf16Temp=50,
        vgprAddrScratch=60,
    )
    writer._subtileHoistedAddrDVgpr = -1
    writer._subtileHoistedAddrBlockN = -1
    writer._fusedFullTileNoGuards = lambda: full_tile
    addr_calc = SimpleNamespace(addrDVgpr=10, globalOffset=64)

    actual_issue, actual_offset = writer._emitPairedStoreIssue(
        20, 30, addr_calc, 100, 200, 80, 2)
    expected_issue, expected_offset = _legacy_paired_store_issue(
        writer, 20, 30, addr_calc, 100, 200, 80, 2)
    actual_commit = writer._emitPairedStoreCommit(
        20, 30, actual_offset, 2, 4)
    expected_commit = _legacy_paired_store_commit(
        writer, 20, 30, expected_offset, 2, 4)

    assert actual_offset == expected_offset
    assert str(actual_issue) == str(expected_issue)
    assert str(actual_commit) == str(expected_commit)


@pytest.fixture(scope="module")
def gfx950_codegen():
    """Real gfx950 solution/codegen context, with process globals restored."""
    from Tensile.Common.Architectures import gfxToIsa
    from Tensile.Common.Capabilities import makeIsaInfoMap
    from Tensile.Common.GlobalParameters import globalParameters, assignGlobalParameters
    from Tensile.Common.ValidParameters import validParameters
    from Tensile.Toolchain.Assembly import makeAssemblyToolchain
    from Tensile.Toolchain.Validators import validateToolchain, ToolchainDefaults

    saved_gp = deepcopy(dict(globalParameters))
    saved_vp = deepcopy(dict(validParameters))
    cxx = validateToolchain("amdclang++")
    isa = gfxToIsa("gfx950")
    iim = makeIsaInfoMap([isa], cxx)
    assignGlobalParameters({}, iim)
    bundler = validateToolchain(ToolchainDefaults.OFFLOAD_BUNDLER)
    assembler = makeAssemblyToolchain(cxx, bundler, "5").assembler
    try:
        yield SimpleNamespace(isa=isa, iim=iim, assembler=assembler)
    finally:
        globalParameters.clear()
        globalParameters.update(saved_gp)
        validParameters.clear()
        validParameters.update(saved_vp)


def _real_plsin_solution(ctx, *, depth_u=256, pgr=2, stream_k=3,
                         dest="B", problem_overrides=None,
                         matrix_instruction=None, **solution_overrides):
    """Derive the production gfx950 FP4 MT256x256 PLSIN shape."""
    from Tensile.BenchmarkProblems import matrixInstructionToMIParameters

    problem_type = {
        "OperationType": "GEMM",
        "DataType": "F4",
        "DestDataType": dest,
        "ComputeDataType": "S",
        "HighPrecisionAccumulate": True,
        "MXBlockA": 32,
        "MXBlockB": 32,
        "TransposeA": True,
        "TransposeB": False,
        "UseBeta": True,
        "Batched": True,
    }
    problem_type.update(problem_overrides or {})
    matrix_instruction = matrix_instruction or [16, 16, 128, 1, 1, 8, 8, 2, 2]
    params = {
        "ProblemType": problem_type,
        "ISA": ctx.isa,
        "MatrixInstruction": matrix_instruction,
        "WorkGroup": [16, 16, 1],
        "WavefrontSize": 64,
        "DepthU": depth_u,
        "KernelLanguage": "Assembly",
        "PrefetchGlobalRead": pgr,
        "PrefetchLocalRead": 1,
        "ScheduleIterAlg": 3,
        "DirectToLds": 1,
        "StaggerU": 0,
        "StreamK": stream_k,
        "UseSubtileImpl": True,
        "LocalReadVectorWidth": -1,
        "GlobalReadVectorWidthA": 16,
        "GlobalReadVectorWidthB": 16,
        "TransposeLDS": 1,
        "LdsPadA": -1,
        "LdsPadB": -1,
        "LdsBlockSizePerPadA": -1,
        "LdsBlockSizePerPadB": -1,
        "1LDSBuffer": 0,
        "VectorWidthA": -1,
        "VectorWidthB": -1,
        "StoreVectorWidth": -1,
        "SourceSwap": True,
        "ExpandPointerSwap": True,
        "GlobalSplitU": 1,
        "InnerUnroll": 1,
        "DebugStreamK": 0,
        "NonTemporalD": 4,
    }
    params.update(matrixInstructionToMIParameters(
        matrix_instruction,
        ctx.isa,
        params["WavefrontSize"],
        problem_type,
        params["WorkGroup"],
        ctx.iim,
    ))
    params.update(solution_overrides)
    return Solution(params, False, True, False, ctx.assembler, ctx.iim)


def _emit_real_plsin(ctx, **kwargs):
    """Emit one real solution through KernelWriterAssembly."""
    import rocisa
    from Tensile.Common.Types import DebugConfig
    from Tensile.SolutionStructs.Naming import getKernelFileBase
    from Tensile.TensileCreateLibrary.Run import (
        generateKernelObjectsFromSolutions,
        processKernelSource,
    )

    solution = _real_plsin_solution(ctx, **kwargs)
    assert solution.get("Valid") is True
    kernels = generateKernelObjectsFromSolutions([solution])
    assert len(kernels) == 1
    kernel = kernels[0]
    kernel.duplicate = False
    kernel["BaseName"] = getKernelFileBase(False, kernel)
    ri = rocisa.rocIsa.getInstance()
    ri.init(tuple(kernel["ISA"]), shutil.which("amdclang++") or "/usr/bin/amdclang++")
    ri.setKernel(tuple(kernel["ISA"]), kernel["WavefrontSize"])
    result = processKernelSource(
        KernelWriterAssembly(ctx.assembler, DebugConfig()),
        ri.getData(),
        ri.getOutputOptions(),
        False,
        kernel,
    )
    source = result.src.decode(errors="replace") \
        if isinstance(result.src, (bytes, bytearray)) else result.src
    return solution, source, result.err


def _assert_gfx950_assembles(ctx, source, tmp_path):
    """Compile generated assembly so symbol/resource errors cannot hide in text checks."""
    asm_path = tmp_path / "kernel.s"
    obj_path = tmp_path / "kernel.o"
    asm_path.write_text(source)
    ctx.assembler("gfx950", 64, str(asm_path), str(obj_path))
    assert obj_path.exists()
    assert obj_path.stat().st_size > 0


@pytest.mark.parametrize("pgr", [1, 2])
@pytest.mark.parametrize("dest", ["B", "H"])
def test_real_gfx950_fp4_plsin_is_default_on_and_assembles(
    gfx950_codegen, tmp_path, pgr, dest
):
    solution, source, err = _emit_real_plsin(
        gfx950_codegen, pgr=pgr, dest=dest
    )

    assert solution.get("Valid") is True
    assert solution["PostLoopStoreInNll"] is True
    assert err == 0
    _assert_gfx950_assembles(gfx950_codegen, source, tmp_path)


def test_real_gfx950_depthu768_uses_normal_codegen_fallback(
    gfx950_codegen, tmp_path
):
    solution, source, err = _emit_real_plsin(
        gfx950_codegen,
        depth_u=768,
        # Keep the LDS footprint below gfx950's limit while retaining enough
        # output pairs that PLSIN would otherwise be profitable.
        matrix_instruction=[16, 16, 128, 1, 1, 2, 4, 2, 2],
        problem_overrides={"MXBlockA": 0, "MXBlockB": 0},
        DirectToLds=0,
    )

    assert solution.get("Valid") is True
    assert solution["PostLoopStoreInNll"] is False
    assert err == 0
    assert "PostLoopStoreInNll 4d-3a" not in source
    _assert_gfx950_assembles(gfx950_codegen, source, tmp_path)


@pytest.mark.parametrize(
    "problem_overrides,required_symbols",
    [
        ({"UseScaleAB": "Vector"}, ("sgprSrdScaleA", "sgprSrdScaleB")),
        ({"UseScaleCD": True}, ("sgprScaleD",)),
        ({"UseScaleAlphaVec": 1}, ("sgprSrdScaleAlphaVec",)),
        ({"UseBias": 1, "BiasDataTypeList": ["S"]}, ("sgprSrdBias",)),
        (
            {
                "UseE": True,
                "DataTypeE": "B",
                "Activation": True,
                "ActivationType": "hipblaslt_all",
            },
            ("sgprSrdE", "sgprStrideE"),
        ),
        (
            {"UseGateResidual": True, "GateResidualDataTypeList": ["B"]},
            (
                "sgprSrdGate",
                "sgprGateNullOne",
                "GateResidual identity",
                "ArgType==2 Gate -> not fused",
            ),
        ),
        (
            {"UseGateResidual": True, "GateResidualDataTypeList": ["B", "H"]},
            (
                "GateLoadAll",
                "GateCvtAll",
                "GateResidual identity",
                "ArgType==2 Gate -> not fused",
            ),
        ),
        (
            {
                "Activation": True,
                "ActivationType": "hipblaslt_all",
                "ActivationFuncCall": True,
            },
            ("sgprActivationType",),
        ),
    ],
    ids=[
        "scale-ab",
        "scale-cd",
        "sav",
        "bias",
        "e",
        "gate",
        "gate-multi-dtype",
        "activation",
    ],
)
def test_real_gfx950_fused_epilogues_codegen(
    gfx950_codegen, tmp_path, problem_overrides, required_symbols
):
    solution, source, err = _emit_real_plsin(
        gfx950_codegen, problem_overrides=problem_overrides
    )

    assert solution["PostLoopStoreInNll"] is True
    assert err == 0
    assert "PostLoopStoreInNll 4d-3a" in source
    for symbol in required_symbols:
        assert symbol in source
    fused_source = source[source.index("PostLoopStoreInNll 4d-3a"):]
    if problem_overrides.get("UseScaleCD"):
        assert fused_source.index(
            "MFMA C[4,0] += A[4,K=1]"
        ) < fused_source.index(
            "copy acc to vreg[16]"
        )
    if problem_overrides.get("UseGateResidual"):
        assert fused_source.index(
            "copy acc to vreg[0]"
        ) < fused_source.index(
            "GateResidual identity"
        )
    _assert_gfx950_assembles(gfx950_codegen, source, tmp_path)


def test_real_gfx950_rich_epilogue_stays_within_sgpr_budget(
    gfx950_codegen, tmp_path
):
    # This is the production-rich FP4 epilogue: scalar ScaleAB, SAV, bias and
    # runtime activation. It is intentionally at the gfx950 SGPR ceiling.
    rich = {
        "UseScaleAB": "Scalar",
        "UseScaleAlphaVec": 1,
        "UseBias": 1,
        "BiasDataTypeList": ["S"],
        "Activation": True,
        "ActivationType": "hipblaslt_all",
        "ActivationFuncCall": True,
    }
    solution, source, err = _emit_real_plsin(
        gfx950_codegen, problem_overrides=rich
    )

    assert solution["PostLoopStoreInNll"] is True
    assert err == 0
    next_free = [
        int(value)
        for value in re.findall(r"\.amdhsa_next_free_sgpr\s+(\d+)", source)
    ]
    assert next_free
    assert max(next_free) <= 102
    _assert_gfx950_assembles(gfx950_codegen, source, tmp_path)
