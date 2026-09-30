# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Correctness guards and code generation for folding D stores into NLL."""

from dataclasses import dataclass
from collections import deque
import re
from typing import Any, Iterable, List, Optional

from rocisa.code import Label, Module
from rocisa.container import (
    DSModifiers,
    MUBUFModifiers,
    VCC,
    VOP3PModifiers,
    accvgpr,
    sgpr,
    vgpr,
)
from rocisa.instruction import (
    BufferStoreB128,
    BufferStoreB64,
    DSLoadB128,
    SBranch,
    SCBranchSCC0,
    SCmpEQU32,
    SCmpEQU64,
    SNop,
    SWaitCnt,
    VAccvgprReadB32,
    VAddPKF32,
    VAndB32,
    VCmpEQU32,
    VCndMaskB32,
    VCvtPkF32toBF16,
    VAddU32,
    VAddLShiftLeftU32,
    VLShiftLeftB32,
    VLShiftRightB32,
    VMovB32,
    VMadU32U24,
    VMulPKF32,
    VMulLOU32,
    VPermlane16SwapB32,
    VSubU32,
)

_MFMA_TILE_RE = re.compile(r"MFMA C\[(\d+),(\d+)\]")


def subtileFlyStorePacketGeometryRejectionReason(state: dict) -> Optional[str]:
    """Return why the focused Fly store geometry is inadmissible."""
    if not state.get("SubtileFlyStorePacketGeometry", 0):
        return None
    checks = (
        (tuple(state.get("ISA", ())) == (9, 5, 0),
         "SubtileFlyStorePacketGeometry requires gfx950"),
        (state.get("UseSubtileImpl", False),
         "SubtileFlyStorePacketGeometry requires UseSubtileImpl=1"),
        (state.get("SourceSwap", False),
         "SubtileFlyStorePacketGeometry requires SourceSwap=1"),
        (state.get("SubtileStoreInNLL", 0),
         "SubtileFlyStorePacketGeometry requires SubtileStoreInNLL=1"),
        (state.get("SubtileRollingPeel", 0),
         "SubtileFlyStorePacketGeometry requires SubtileRollingPeel=1"),
        (state.get("SubtileFlyNMajor", 0),
         "SubtileFlyStorePacketGeometry requires SubtileFlyNMajor=1"),
        (state.get("MacroTile0") == 256 and state.get("MacroTile1") == 256,
         "SubtileFlyStorePacketGeometry requires MacroTile=256x256"),
        (state.get("DepthU", state.get("_DepthUA")) == 256,
         "SubtileFlyStorePacketGeometry requires DepthU=256"),
        (state.get("MatrixInstM") == 16 and state.get("MatrixInstK") == 128,
         "SubtileFlyStorePacketGeometry requires MI16x16x128"),
        (state.get("MIWaveGroup") == [2, 2],
         "SubtileFlyStorePacketGeometry requires MIWaveGroup=[2,2]"),
        (state.get("MIWaveTile") == [8, 8],
         "SubtileFlyStorePacketGeometry requires MIWaveTile=[8,8]"),
    )
    for accepted, reason in checks:
        if not accepted:
            return reason
    return None


def flyStorePacketGeometryEnabled(kernel: dict) -> bool:
    """Validate and return the target-only Fly store packet gate."""
    reason = subtileFlyStorePacketGeometryRejectionReason(kernel)
    if reason is not None:
        raise ValueError(reason)
    return bool(kernel.get("SubtileFlyStorePacketGeometry", 0))


def flyStorePacketCoordinates(
        wave_id: int, lane: int, tile_n: int, sl: int,
        component: int) -> list:
    """Return the four C coordinates carried by one Fly dwordx2 store.

    The two M waves own 128-row halves.  ``sl`` selects a 64-row quarter
    within each wave, while the lane's low four bits select a four-row
    packet.  The N wave owns one 128-column half; ``tile_n`` and the lane
    group select its 16- and four-column subdivisions.
    """
    values = (wave_id, lane, tile_n, sl, component)
    limits = (4, 64, 8, 2, 4)
    assert all(0 <= int(value) < limit
               for value, limit in zip(values, limits))
    lane16 = lane & 15
    lane_group = lane >> 4
    wave_m = wave_id & 1
    wave_n = wave_id >> 1
    m_base = wave_m * 128 + sl * 64 + lane16 * 4
    n = wave_n * 128 + tile_n * 16 + lane_group * 4 + component
    return [(m_base + ji, n) for ji in range(4)]


def flyNMajorProducerQ(tile_n: int, sl: int, ji: int) -> int:
    """Map a Fly N-major store packet row to its producer tile index."""
    assert 0 <= tile_n < 8
    assert 0 <= sl < 2
    assert 0 <= ji < 4
    return sl * 32 + tile_n * 4 + ji


def subtileStoreInNLLRejectionReason(state: dict) -> Optional[str]:
    """Return why ``SubtileStoreInNLL`` is inadmissible, or ``None``."""
    if not state.get("SubtileStoreInNLL", 0):
        return None

    pt = state.get("ProblemType", {})
    dest_type = pt.get("DestDataType")
    dest_is_bf16 = (dest_type == 7
                    or getattr(dest_type, "isBFloat16", lambda: False)())
    checks = (
        (tuple(state.get("ISA", ())) == (9, 5, 0), "SubtileStoreInNLL requires gfx950"),
        (state.get("UseSubtileImpl", False), "SubtileStoreInNLL requires UseSubtileImpl=1"),
        (state.get("SourceSwap", False), "SubtileStoreInNLL requires SourceSwap=1"),
        (not state.get("CompactLoopStore", False),
         "SubtileStoreInNLL is incompatible with CompactLoopStore"),
        (state.get("StreamK", 0) == 0, "SubtileStoreInNLL requires StreamK=0"),
        (state.get("GlobalSplitU", 1) == 1, "SubtileStoreInNLL requires GlobalSplitU=1"),
        (state.get("MIWaveTile") == [8, 8],
         "SubtileStoreInNLL requires MIWaveTile=[8,8]"),
        (not state.get("MIArchVgpr", False),
         "SubtileStoreInNLL requires accumulator registers"),
        (pt.get("HighPrecisionAccumulate", False),
         "SubtileStoreInNLL requires HighPrecisionAccumulate=1"),
        (dest_is_bf16,
         "SubtileStoreInNLL requires BF16 destination"),
        (not pt.get("UseE", False), "SubtileStoreInNLL does not support E output"),
        (not pt.get("UseScaleD", False) and not pt.get("UseScaleCD", False),
         "SubtileStoreInNLL does not support scaleD"),
        (not pt.get("OutputAmaxD", False), "SubtileStoreInNLL does not support amaxD"),
        (bool(state.get("NonTemporalD", 0) & 0x4),
         "SubtileStoreInNLL requires NonTemporalD bit 0x4"),
        (state.get("AssertFree0ElementMultiple", 1) % state.get("MacroTile0", 1) == 0,
         "SubtileStoreInNLL requires full MacroTile-aligned M"),
        (state.get("AssertFree1ElementMultiple", 1) % state.get("MacroTile1", 1) == 0,
         "SubtileStoreInNLL requires full MacroTile-aligned N"),
    )
    for accepted, reason in checks:
        if not accepted:
            return reason
    fly_reason = subtileFlyStorePacketGeometryRejectionReason(state)
    if fly_reason is not None:
        return fly_reason
    return None


@dataclass(frozen=True)
class NLLStoreEvent:
    """One abstract NLL issue used by :func:`planNLLStoreInterleave`."""

    kind: str
    accumulator: Optional[Any] = None
    payload: Any = None


def planNLLStoreInterleave(
    mfmaAccumulators: Iterable[Any],
    stores: Iterable[NLLStoreEvent],
    minMfmaGap: int = 4,
) -> List[NLLStoreEvent]:
    """Place each store after its accumulator's final producer.

    A store is delayed by ``minMfmaGap`` later MFMA issues when the NLL has that
    much remaining work.  Near the end, it is drained immediately after the
    final available MFMA, so every store remains inside NLL.  The routine is
    intentionally independent of rocISA instruction classes and can schedule
    either direct-AGPR stores or a conversion-plus-store payload.
    """
    if minMfmaGap < 0:
        raise ValueError("minMfmaGap must be non-negative")

    mfmas = list(mfmaAccumulators)
    storeList = list(stores)
    lastProducer = {acc: idx for idx, acc in enumerate(mfmas)}
    buckets = [[] for _ in range(len(mfmas) + 1)]

    for store in storeList:
        if store.kind != "store":
            raise ValueError("stores must contain only kind='store' events")
        if store.accumulator not in lastProducer:
            raise ValueError(f"no NLL producer for accumulator {store.accumulator!r}")
        producer = lastProducer[store.accumulator]
        issueAfter = min(producer + minMfmaGap, len(mfmas) - 1)
        buckets[issueAfter + 1].append(store)

    result: List[NLLStoreEvent] = []
    for idx, accumulator in enumerate(mfmas):
        result.append(NLLStoreEvent("mfma", accumulator))
        result.extend(buckets[idx + 1])
    return result


def nonTemporalDFlags(nonTemporalD: int) -> dict:
    """Return the MUBUF cache flags, preserving the D ``nt`` bit."""
    return {
        "glc": bool(nonTemporalD & 0x1),
        "slc": bool(nonTemporalD & 0x2),
        "nt": bool(nonTemporalD & 0x4),
    }


def emitSubtileStoreInNLLAddressInit(writer, kernel) -> Module:
    """Allocate the folded-store address/scratch VGPRs and form D's lane address."""
    module = Module("SubtileStoreInNLLAddressInit")
    if hasattr(writer.states, "subtileStoreInNLLVaddr"):
        return module
    fly_packets = bool(kernel.get("SubtileFlyStorePacketGeometry", 0))
    if fly_packets:
        flyStorePacketGeometryEnabled(kernel)

    # Keep the lane base plus two alternating four-address banks.  Eight
    # packed dwordx2 packets likewise alternate across two four-packet banks.
    vaddr = writer.vgprPool.checkOut(9, "subtileStoreInNLLVaddr")
    data = writer.vgprPool.checkOutAligned(32, 2, "subtileStoreInNLLData")
    stride = writer.vgprPool.checkOut(4, "subtileStoreInNLLStride")
    alpha = writer.vgprPool.checkOutAligned(2, 2, "subtileStoreInNLLAlpha")
    writer.states.subtileStoreInNLLVaddr = vaddr
    writer.states.subtileStoreInNLLData = data
    writer.states.subtileStoreInNLLStride = stride
    writer.states.subtileStoreInNLLStride8 = stride + 1
    writer.states.subtileStoreInNLLStride32 = stride + 2
    if kernel.get("SubtileFlyNMajor", 0):
        writer.states.subtileStoreInNLLStride16 = stride + 2
    writer.states.subtileStoreInNLLLaneGroupOdd = stride + 3
    writer.states.subtileStoreInNLLAlpha = alpha
    if fly_packets or kernel.get("SubtileFlyNMajor", 0):
        writer.states.subtileStoreInNLLSecondHalfOffset = 128
        module.addComment0(
            "Fly store packets: second accumulator quarter is +128 bytes")
    strideD1 = "StrideD%s" % writer.states.indexChars[kernel["PackedC1IndicesX"][0]]
    if kernel.get("SubtileFlyNMajor", 0):
        # Form Fly coordinates directly.  Legacy SourceSwap coord0/coord1 shift
        # their wave terms by VW=8, so they cannot be algebraically rescaled.
        lane_group = stride + 3
        mbase = data
        nbase = data + 1
        wave = data + 2
        tmp = data + 3
        module.add(VAndB32(
            dst=vgpr(mbase), src0=15, src1=vgpr("Serial"),
            comment="Fly lane16"))
        module.add(VLShiftLeftB32(
            dst=vgpr(mbase), shiftHex=hex(2), src=vgpr(mbase),
            comment="Fly lane16 four-row M base"))
        module.add(VLShiftRightB32(
            dst=vgpr(wave), shiftHex=hex(6), src=vgpr("Serial"),
            comment="Fly wave id"))
        module.add(VAndB32(
            dst=vgpr(tmp), src0=1, src1=vgpr(wave),
            comment="Fly M-wave id"))
        module.add(VLShiftLeftB32(
            dst=vgpr(tmp), shiftHex=hex(7), src=vgpr(tmp),
            comment="Fly M-wave 128-row offset"))
        module.add(VAddU32(
            dst=vgpr(mbase), src0=vgpr(mbase), src1=vgpr(tmp)))
        writer.states.subtileStoreInNLLLocalM = vaddr + 5
        module.add(VMovB32(
            dst=vgpr(writer.states.subtileStoreInNLLLocalM),
            src=vgpr(mbase),
            comment="preserve Fly local M base for vector LDS reads"))
        module.add(VMovB32(
            dst=vgpr(tmp), src=sgpr("WorkGroup0"),
            comment="copy workgroup M coordinate"))
        module.add(VLShiftLeftB32(
            dst=vgpr(tmp), shiftHex=hex(8), src=vgpr(tmp),
            comment="Fly workgroup M base"))
        module.add(VAddU32(
            dst=vgpr(mbase), src0=vgpr(mbase), src1=vgpr(tmp)))
        module.add(VLShiftRightB32(
            dst=vgpr(lane_group), shiftHex=hex(4), src=vgpr("Serial"),
            comment="Fly lane-group index"))
        module.add(VAndB32(
            dst=vgpr(lane_group), src0=3, src1=vgpr(lane_group)))
        module.add(VLShiftLeftB32(
            dst=vgpr(nbase), shiftHex=hex(2), src=vgpr(lane_group),
            comment="Fly lane-group four-column N base"))
        module.add(VLShiftRightB32(
            dst=vgpr(wave), shiftHex=hex(1), src=vgpr(wave),
            comment="Fly N-wave id"))
        module.add(VLShiftLeftB32(
            dst=vgpr(wave), shiftHex=hex(7), src=vgpr(wave),
            comment="Fly N-wave 128-column offset"))
        module.add(VAddU32(
            dst=vgpr(nbase), src0=vgpr(nbase), src1=vgpr(wave)))
        module.add(VMulLOU32(
            dst=vgpr(vaddr), src0=vgpr(nbase), src1=sgpr(strideD1),
            comment="Fly N-major tile-local D column offset"))
        module.add(VAddLShiftLeftU32(
            dst=vgpr(vaddr), src0=vgpr(vaddr), src1=vgpr(mbase),
            shiftHex=1,
            comment="Fly N-major D address = (N*strideD + M) * sizeof(bf16)"))
    else:
        module.add(VMulLOU32(
            dst=vgpr(vaddr),
            src0=vgpr(writer.vgprs.coord1),
            src1=sgpr(strideD1),
            comment="folded-store global D column offset",
        ))
        module.add(VAddLShiftLeftU32(
            dst=vgpr(vaddr),
            src0=vgpr(vaddr),
            src1=vgpr(writer.vgprs.coord0),
            shiftHex=1,
            comment="folded-store D address = (column*strideD + row) * sizeof(bf16)",
        ))
    module.add(VLShiftLeftB32(
        dst=vgpr(stride),
        shiftHex=hex(1),
        src=sgpr(strideD1),
        comment="folded-store BF16 D column stride in bytes"))
    module.add(VLShiftLeftB32(
        dst=vgpr(stride + 1),
        shiftHex=hex(3),
        src=vgpr(stride),
        comment="peeled packet-group eight-column byte stride"))
    module.add(VLShiftLeftB32(
        dst=vgpr(stride + 2),
        shiftHex=hex(4 if kernel.get("SubtileFlyNMajor", 0) else 5),
        src=vgpr(stride),
        comment=(
            "Fly packet-group 16-column byte stride"
            if kernel.get("SubtileFlyNMajor", 0)
            else "peeled packet-group 32-column byte stride")))
    if not kernel.get("SubtileFlyNMajor", 0):
        module.add(VLShiftRightB32(
            dst=vgpr(stride + 3),
            shiftHex=hex(4),
            src=vgpr("Serial"),
            comment="folded-store lane-group index"))
        module.add(VAndB32(
            dst=vgpr(stride + 3),
            src0=1,
            src1=vgpr(stride + 3),
            comment="folded-store odd lane-group flag"))
    identity_store_interleave = bool(
        kernel.get("SubtileIdentityStoreFastPath", 0)
        and kernel.get("SubtileStoreInNLL", 0)
        and kernel.get("SubtileFlyNMajor", 0)
        and kernel.get("SubtileRollingPeel", 0)
        and not kernel.get("SubtileDeferredStoreInterleave", 0)
        and not kernel.get("SubtileNativeWholeLoopScheduler", 0))
    if identity_store_interleave:
        module.add(VMovB32(
            dst=vgpr(alpha), src=sgpr("AddressD"),
            comment="preserve raw D base low across loop"))
        module.add(VMovB32(
            dst=vgpr(alpha + 1), src=sgpr("AddressD+1"),
            comment="preserve raw D base high across loop"))
    else:
        module.add(VMovB32(
            dst=vgpr(alpha), src=sgpr("Alpha"),
            comment="duplicate folded-store alpha for packed multiply"))
        module.add(VMovB32(
            dst=vgpr(alpha + 1), src=sgpr("Alpha"),
            comment="duplicate folded-store alpha for packed multiply"))
    return module


def cleanupSubtileStoreInNLL(writer) -> None:
    """Release folded-store-only VGPRs after post-loop cleanup."""
    if hasattr(writer.states, "subtileStoreInNLLVaddr"):
        writer.vgprPool.checkIn(writer.states.subtileStoreInNLLVaddr)
        del writer.states.subtileStoreInNLLVaddr
    if hasattr(writer.states, "subtileStoreInNLLData"):
        writer.vgprPool.checkIn(writer.states.subtileStoreInNLLData)
        del writer.states.subtileStoreInNLLData
    if hasattr(writer.states, "subtileStoreInNLLStride"):
        writer.vgprPool.checkIn(writer.states.subtileStoreInNLLStride)
        del writer.states.subtileStoreInNLLStride
        del writer.states.subtileStoreInNLLStride8
        del writer.states.subtileStoreInNLLStride32
        if hasattr(writer.states, "subtileStoreInNLLStride16"):
            del writer.states.subtileStoreInNLLStride16
        del writer.states.subtileStoreInNLLLaneGroupOdd
    if hasattr(writer.states, "subtileStoreInNLLAlpha"):
        writer.vgprPool.checkIn(writer.states.subtileStoreInNLLAlpha)
        del writer.states.subtileStoreInNLLAlpha
    if hasattr(writer.states, "subtileStoreInNLLSecondHalfOffset"):
        del writer.states.subtileStoreInNLLSecondHalfOffset
    if hasattr(writer.states, "subtileStoreInNLLLocalM"):
        del writer.states.subtileStoreInNLLLocalM


def _column_accumulators(writer, dtileInfo, column: int) -> list:
    """Return the eight accumulator indices owned by one output column."""
    tiles_m = dtileInfo.localMMATileGrid[0]
    tile_n = column % dtileInfo.localMMATileGrid[1]
    component = column // dtileInfo.localMMATileGrid[1]
    regs = []
    for tile_m in range(tiles_m):
        tile = dtileInfo.vgprTiles[tile_m + tile_n * tiles_m]
        if tile.regList.pool == writer.vgprPool:
            raise RuntimeError("SubtileStoreInNLL does not support accumulator spill to VGPR")
        regs.append(tile.regList.indices[component])
    if len(regs) != 8:
        raise RuntimeError(f"SubtileStoreInNLL expected 8 accumulators, got {len(regs)}")
    return regs


def _store_unit(writer, kernel, dtileInfo, column: int) -> list:
    """Return one eight-row wide BF16 store for a SourceSwap output column."""
    regs = _column_accumulators(writer, dtileInfo, column)

    flags = nonTemporalDFlags(kernel["NonTemporalD"])
    mubuf = lambda offset: MUBUFModifiers(
        offen=True, offset12=offset, glc=flags["glc"], slc=flags["slc"], nt=flags["nt"])
    vaddr = writer.states.subtileStoreInNLLVaddr
    data = writer.states.subtileStoreInNLLData
    alpha = writer.states.subtileStoreInNLLAlpha
    insts = []

    for row, reg in enumerate(regs):
        insts.append(VAccvgprReadB32(
            dst=vgpr(data + row), src=accvgpr(reg),
            comment=f"fold C column {column} row {row}: acc -> vgpr"))
    for row in range(0, 8, 2):
        insts.append(VMulPKF32(
            dst=vgpr(data + row, 2),
            src0=vgpr(alpha, 2),
            src1=vgpr(data + row, 2),
            vop3=VOP3PModifiers(op_sel_hi=[0, 1, 1]),
            comment="apply alpha to two folded-store rows"))
    for pair in range(4):
        insts.append(VCvtPkF32toBF16(
            dst=vgpr(data + pair),
            src0=vgpr(data + 2 * pair),
            src1=vgpr(data + 2 * pair + 1),
            comment=f"pack folded BF16 rows {2 * pair}:{2 * pair + 2}"))
    insts.append(BufferStoreB128(
        src=vgpr(data, 4),
        vaddr=vgpr(vaddr),
        saddr=sgpr("SrdD", 4),
        soffset=0,
        mubuf=mubuf(0),
        comment=f"folded wide non-temporal D store column {column}"))

    if column != 31:
        insts.append(VAddU32(
            dst=vgpr(vaddr), src0=vgpr(vaddr),
            src1=vgpr(writer.states.subtileStoreInNLLStride),
            comment="advance folded-store address by one BF16 D column"))
    return insts


def _store_packet(writer, kernel, dtileInfo, column: int,
                  row_block: int) -> list:
    """Return one absolute-address alpha-one four-row dwordx2 packet."""
    regs = _column_accumulators(writer, dtileInfo, column)
    row_start = row_block * 4
    regs = regs[row_start:row_start + 4]
    flags = nonTemporalDFlags(kernel["NonTemporalD"])
    mubuf = MUBUFModifiers(
        offen=True, offset12=row_start * 2, glc=flags["glc"],
        slc=flags["slc"], nt=flags["nt"])
    base_vaddr = writer.states.subtileStoreInNLLVaddr
    vaddr = base_vaddr + 1
    data = writer.states.subtileStoreInNLLData
    insts = [VMadU32U24(
        dst=vgpr(vaddr),
        src0=column,
        src1=vgpr(writer.states.subtileStoreInNLLStride),
        src2=vgpr(base_vaddr),
        comment="peeled packet absolute D address")]
    for row, reg in enumerate(regs):
        insts.append(VAccvgprReadB32(
            dst=vgpr(data + row), src=accvgpr(reg),
            comment=(f"peel C column {column} rows "
                     f"{row_start + row}: acc -> vgpr")))
    for pair in range(2):
        insts.append(VCvtPkF32toBF16(
            dst=vgpr(data + pair),
            src0=vgpr(data + 2 * pair),
            src1=vgpr(data + 2 * pair + 1),
            comment="pack alpha-one peeled BF16 rows"))
    insts.append(BufferStoreB64(
        src=vgpr(data, 2),
        vaddr=vgpr(vaddr),
        saddr=sgpr("SrdD", 4),
        soffset=0,
        mubuf=mubuf,
        comment=(f"peeled dwordx2 D store column {column} "
                 f"rows {row_start}:{row_start + 4}")))
    return insts


def _store_packet_group(writer, kernel, dtileInfo, tile_n: int,
                        row_block: int, bank: int) -> list:
    """Stage four dwordx2 packets, then issue one four-store burst."""
    columns = list(range(tile_n, 32, dtileInfo.localMMATileGrid[1]))
    if len(columns) != 4:
        raise RuntimeError("peeled store group requires four physical columns")

    base_vaddr = writer.states.subtileStoreInNLLVaddr
    addresses = base_vaddr + 1 + bank * 4
    data = writer.states.subtileStoreInNLLData + bank * 8
    scratch = writer.states.subtileStoreInNLLData + 16
    stride = writer.states.subtileStoreInNLLStride
    stride8 = writer.states.subtileStoreInNLLStride8
    row_start = row_block * 4
    flags = nonTemporalDFlags(kernel["NonTemporalD"])
    mubuf = MUBUFModifiers(
        offen=True, offset12=row_start * 2, glc=flags["glc"],
        slc=flags["slc"], nt=flags["nt"])

    insts = [VMadU32U24(
        dst=vgpr(addresses),
        src0=tile_n,
        src1=vgpr(stride),
        src2=vgpr(base_vaddr),
        comment="peeled packet-group first D address")]
    for slot in range(1, 4):
        insts.append(VAddU32(
            dst=vgpr(addresses + slot),
            src0=vgpr(addresses + slot - 1),
            src1=vgpr(stride8),
            comment="advance peeled address by eight D columns"))

    for slot, column in enumerate(columns):
        regs = _column_accumulators(writer, dtileInfo, column)
        regs = regs[row_start:row_start + 4]
        packed = data + slot * 2
        insts.extend([
            VAccvgprReadB32(
                dst=vgpr(packed), src=accvgpr(regs[0]),
                comment=f"stage peel column {column} row {row_start}"),
            VAccvgprReadB32(
                dst=vgpr(scratch), src=accvgpr(regs[1]),
                comment=f"stage peel column {column} row {row_start + 1}"),
            VCvtPkF32toBF16(
                dst=vgpr(packed), src0=vgpr(packed), src1=vgpr(scratch),
                comment="pack staged peeled BF16 pair"),
            VAccvgprReadB32(
                dst=vgpr(packed + 1), src=accvgpr(regs[2]),
                comment=f"stage peel column {column} row {row_start + 2}"),
            VAccvgprReadB32(
                dst=vgpr(scratch + 1), src=accvgpr(regs[3]),
                comment=f"stage peel column {column} row {row_start + 3}"),
            VCvtPkF32toBF16(
                dst=vgpr(packed + 1), src0=vgpr(packed + 1),
                src1=vgpr(scratch + 1),
                comment="pack staged peeled BF16 pair"),
        ])

    for slot, column in enumerate(columns):
        insts.append(BufferStoreB64(
            src=vgpr(data + slot * 2, 2),
            vaddr=vgpr(addresses + slot),
            saddr=sgpr("SrdD", 4),
            soffset=0,
            mubuf=mubuf,
            comment=(f"staged peeled dwordx2 D store column {column} "
                     f"rows {row_start}:{row_start + 4}")))
    return insts


def _store_recoalesced_packet_group(writer, kernel, dtileInfo,
                                    tile_n: int) -> list:
    """Transpose SourceSwap lane groups into contiguous four-row dwordx2 stores.

    A SourceSwap lane owns eight adjacent M rows, so splitting its output into
    two dwordx2 stores leaves an eight-byte hole between neighboring lanes.
    Pair adjacent 16-lane groups instead: even groups carry the lower halves
    and odd groups carry the upper halves of one 32-column band.  The second
    store reverses that ownership for the adjacent band.  Both stores then
    cover complete cache-line runs, matching FlyDSL's packet coalescing while
    preserving SS1's accumulator mapping.
    """
    columns = list(range(tile_n, 32, dtileInfo.localMMATileGrid[1]))
    if len(columns) != 4:
        raise RuntimeError("recoalesced store group requires four physical columns")

    base_vaddr = writer.states.subtileStoreInNLLVaddr
    addresses = base_vaddr + 1
    data = writer.states.subtileStoreInNLLData
    scratch = data + 16
    stride = writer.states.subtileStoreInNLLStride
    stride8 = writer.states.subtileStoreInNLLStride8
    stride32 = writer.states.subtileStoreInNLLStride32
    odd = writer.states.subtileStoreInNLLLaneGroupOdd
    flags = nonTemporalDFlags(kernel["NonTemporalD"])
    mubuf = MUBUFModifiers(
        offen=True, offset12=0, glc=flags["glc"],
        slc=flags["slc"], nt=flags["nt"])

    # Start from this lane's normal SourceSwap address.  Address A merges lane
    # groups 0/1 and 2/3 into the even 32-column bands; address B merges them
    # into the odd bands.  Odd groups contribute rows +4 (eight bytes).
    insts = [
        VMadU32U24(
            dst=vgpr(addresses),
            src0=tile_n,
            src1=vgpr(stride),
            src2=vgpr(base_vaddr),
            comment="recoalesced packet original D address"),
        VSubU32(
            dst=vgpr(scratch),
            src0=vgpr(addresses),
            src1=vgpr(stride32),
            comment="odd-group even-band D address"),
        VAddU32(
            dst=vgpr(scratch),
            src0=vgpr(scratch),
            src1=8,
            comment="odd-group upper four-row D address"),
        VAddU32(
            dst=vgpr(scratch + 1),
            src0=vgpr(addresses),
            src1=vgpr(stride32),
            comment="even-group odd-band D address"),
        VCmpEQU32(
            dst=VCC(), src0=vgpr(odd), src1=1,
            comment="select odd SourceSwap lane groups"),
        VCndMaskB32(
            dst=vgpr(addresses),
            src0=vgpr(addresses),
            src1=vgpr(scratch),
            src2=VCC(),
            comment="recoalesced even-band D address"),
        VAddU32(
            dst=vgpr(addresses + 1),
            src0=vgpr(base_vaddr),
            src1=8,
            comment="odd-group odd-band upper-row candidate"),
        VMadU32U24(
            dst=vgpr(addresses + 1),
            src0=tile_n,
            src1=vgpr(stride),
            src2=vgpr(addresses + 1),
            comment="odd-group odd-band D address"),
        VCndMaskB32(
            dst=vgpr(addresses + 1),
            src0=vgpr(scratch + 1),
            src1=vgpr(addresses + 1),
            src2=VCC(),
            comment="recoalesced odd-band D address"),
    ]
    for slot in range(1, 4):
        insts.extend([
            VAddU32(
                dst=vgpr(addresses + slot * 2),
                src0=vgpr(addresses + (slot - 1) * 2),
                src1=vgpr(stride8),
                comment="advance recoalesced even-band address by eight columns"),
            VAddU32(
                dst=vgpr(addresses + slot * 2 + 1),
                src0=vgpr(addresses + (slot - 1) * 2 + 1),
                src1=vgpr(stride8),
                comment="advance recoalesced odd-band address by eight columns"),
        ])

    for slot, column in enumerate(columns):
        regs = _column_accumulators(writer, dtileInfo, column)
        packed = data + slot * 4
        for pair in range(4):
            insts.extend([
                VAccvgprReadB32(
                    dst=vgpr(scratch), src=accvgpr(regs[pair * 2]),
                    comment=f"stage recoalesced column {column} row {pair * 2}"),
                VAccvgprReadB32(
                    dst=vgpr(scratch + 1), src=accvgpr(regs[pair * 2 + 1]),
                    comment=f"stage recoalesced column {column} row {pair * 2 + 1}"),
                VCvtPkF32toBF16(
                    dst=vgpr(packed + pair),
                    src0=vgpr(scratch),
                    src1=vgpr(scratch + 1),
                    comment="pack recoalesced peeled BF16 pair"),
            ])
        for dword in range(2):
            lower = packed + dword
            upper = packed + 2 + dword
            insts.append(VPermlane16SwapB32(
                dst=vgpr(lower),
                src=vgpr(upper),
                comment="transpose lower/upper packets across lane groups"))
        if slot & 1:
            for store_slot in (slot - 1, slot):
                store_column = columns[store_slot]
                insts.extend([
                    BufferStoreB64(
                        src=vgpr(data + store_slot * 4, 2),
                        vaddr=vgpr(addresses + store_slot * 2),
                        saddr=sgpr("SrdD", 4),
                        soffset=0,
                        mubuf=mubuf,
                        comment=(
                            "recoalesced even-band dwordx2 "
                            f"column {store_column}")),
                    BufferStoreB64(
                        src=vgpr(data + store_slot * 4 + 2, 2),
                        vaddr=vgpr(addresses + store_slot * 2 + 1),
                        saddr=sgpr("SrdD", 4),
                        soffset=0,
                        mubuf=mubuf,
                        comment=(
                            "recoalesced odd-band dwordx2 "
                            f"column {store_column}")),
                ])
    return insts


def _store_final_wide_group(writer, kernel, dtileInfo, tile_n: int) -> list:
    """Drain the final SourceSwap N group as four coalesced dwordx4 stores."""
    columns = list(range(tile_n, 32, dtileInfo.localMMATileGrid[1]))
    if len(columns) != 4:
        raise RuntimeError("final wide store group requires four physical columns")

    base_vaddr = writer.states.subtileStoreInNLLVaddr
    addresses = base_vaddr + 1
    data = writer.states.subtileStoreInNLLData
    scratch = data + 16
    stride = writer.states.subtileStoreInNLLStride
    stride8 = writer.states.subtileStoreInNLLStride8
    flags = nonTemporalDFlags(kernel["NonTemporalD"])
    mubuf = MUBUFModifiers(
        offen=True, offset12=0, glc=flags["glc"],
        slc=flags["slc"], nt=flags["nt"])

    insts = [VMadU32U24(
        dst=vgpr(addresses),
        src0=tile_n,
        src1=vgpr(stride),
        src2=vgpr(base_vaddr),
        comment="final wide-group first D address")]
    for slot in range(1, 4):
        insts.append(VAddU32(
            dst=vgpr(addresses + slot),
            src0=vgpr(addresses + slot - 1),
            src1=vgpr(stride8),
            comment="advance final wide address by eight columns"))

    for slot, column in enumerate(columns):
        regs = _column_accumulators(writer, dtileInfo, column)
        packed = data + slot * 4
        for pair in range(4):
            insts.extend([
                VAccvgprReadB32(
                    dst=vgpr(scratch), src=accvgpr(regs[pair * 2]),
                    comment=f"stage final wide column {column} row {pair * 2}"),
                VAccvgprReadB32(
                    dst=vgpr(scratch + 1),
                    src=accvgpr(regs[pair * 2 + 1]),
                    comment=f"stage final wide column {column} row {pair * 2 + 1}"),
                VCvtPkF32toBF16(
                    dst=vgpr(packed + pair),
                    src0=vgpr(scratch),
                    src1=vgpr(scratch + 1),
                    comment="pack final wide BF16 pair"),
            ])

    for slot, column in enumerate(columns):
        insts.append(BufferStoreB128(
            src=vgpr(data + slot * 4, 4),
            vaddr=vgpr(addresses + slot),
            saddr=sgpr("SrdD", 4),
            soffset=0,
            mubuf=mubuf,
            comment=f"final wide dwordx4 column {column}"))
    return insts


def _store_fly_nmajor_group(writer, kernel, dtileInfo, tile_n: int) -> list:
    """Emit FlyDSL's direct ji-adjacent stores for one original-N tile."""
    if dtileInfo.localMMATileGrid != [8, 8]:
        raise RuntimeError("Fly N-major stores require an 8x8 MFMA tile")

    base_vaddr = writer.states.subtileStoreInNLLVaddr
    addresses = base_vaddr + 1
    data = writer.states.subtileStoreInNLLData
    lds_addr = base_vaddr + 6
    alpha = writer.states.subtileStoreInNLLAlpha
    stride = writer.states.subtileStoreInNLLStride
    stride16 = writer.states.subtileStoreInNLLStride16
    flags = nonTemporalDFlags(kernel["NonTemporalD"])
    insts = [VMulLOU32(
        dst=vgpr(addresses),
        src0=tile_n,
        src1=vgpr(stride16),
        comment=f"Fly N-major tile {tile_n} column-byte offset"),
        VAddU32(
            dst=vgpr(addresses),
            src0=vgpr(addresses),
            src1=vgpr(base_vaddr),
            comment=f"Fly N-major tile {tile_n} first D address")]
    for component in range(1, 4):
        insts.append(VAddU32(
            dst=vgpr(addresses + component),
            src0=vgpr(addresses + component - 1),
                src1=vgpr(stride),
                comment="advance Fly N-major address by one D column"))
    insts.append(SNop(
        waitState=7,
        comment="wait for final MFMA accumulator writes before folded stores"))

    for sl in range(2):
        insts.extend([
            VLShiftLeftB32(
                dst=vgpr(lds_addr),
                shiftHex=hex(2),
                src=vgpr(writer.states.subtileStoreInNLLLocalM),
                comment="Fly bias/scale-alpha LDS byte address"),
            DSLoadB128(
                dst=vgpr(bias, 4),
                src=vgpr(lds_addr),
                ds=DSModifiers(offset=sl * 256),
                comment=f"Fly bias rows sl={sl}"),
            DSLoadB128(
                dst=vgpr(scale_alpha, 4),
                src=vgpr(lds_addr),
                ds=DSModifiers(offset=1024 + sl * 256),
                comment=f"Fly scale-alpha rows sl={sl}"),
        ])
        raw_regs = []
        for component in range(4):
            regs = []
            for ji in range(4):
                q = flyNMajorProducerQ(tile_n, sl, ji)
                tile = dtileInfo.vgprTiles[q]
                if tile.regList.pool == writer.vgprPool:
                    raise RuntimeError(
                        "Fly N-major stores do not support accumulator spill to VGPR")
                regs.append(tile.regList.indices[component])
            raw = data + component * 4
            raw_regs.append(raw)
            for ji, reg in enumerate(regs):
                insts.append(VAccvgprReadB32(
                    dst=vgpr(raw + ji), src=accvgpr(reg),
                    comment=(
                        f"Fly N-major sl={sl} e={component} ji={ji}")))
        insts.append(SWaitCnt(
            dscnt=0, vlcnt=-1, vscnt=-1,
            comment="wait for Fly bias/scale-alpha rows"))
        for component, raw in enumerate(raw_regs):
            packed = data + component * 2
            for pair in range(2):
                row_pair = raw + pair * 2
                insts.extend([
                    VMulPKF32(
                        dst=vgpr(row_pair, 2),
                        src0=vgpr(alpha, 2),
                        src1=vgpr(row_pair, 2),
                        vop3=VOP3PModifiers(op_sel_hi=[0, 1, 1]),
                        comment="apply alpha to Fly rows"),
                    VMulPKF32(
                        dst=vgpr(row_pair, 2),
                        src0=vgpr(scale_alpha + pair * 2, 2),
                        src1=vgpr(row_pair, 2),
                        comment="apply scale-alpha to Fly rows"),
                    VAddPKF32(
                        dst=vgpr(row_pair, 2),
                        src0=vgpr(bias + pair * 2, 2),
                        src1=vgpr(row_pair, 2),
                        comment="add bias to Fly rows"),
                ])
            insts.extend([
                VCvtPkF32toBF16(
                    dst=vgpr(packed),
                    src0=vgpr(raw),
                    src1=vgpr(raw + 1),
                    comment="pack Fly N-major ji 0:2"),
                VCvtPkF32toBF16(
                    dst=vgpr(packed + 1),
                    src0=vgpr(raw + 2),
                    src1=vgpr(raw + 3),
                    comment="pack Fly N-major ji 2:4"),
            ])
        for component in range(4):
            offset = sl * writer.states.subtileStoreInNLLSecondHalfOffset
            mubuf = MUBUFModifiers(
                offen=True, offset12=offset, glc=flags["glc"],
                slc=flags["slc"], nt=flags["nt"])
            insts.append(BufferStoreB64(
                src=vgpr(data + component * 2, 2),
                vaddr=vgpr(addresses + component),
                saddr=sgpr("SrdD", 4),
                soffset=0,
                mubuf=mubuf,
                comment=(
                    f"Fly N-major dwordx2 tile={tile_n} "
                    f"sl={sl} e={component}")))
    return insts


def _store_fly_nmajor_staged_drain(
        writer, kernel, dtileInfo, stage: int) -> list:
    """Stage a canonical 64-read component pass before N-major stores."""
    base_vaddr = writer.states.subtileStoreInNLLVaddr
    address = base_vaddr + 1
    data = writer.states.subtileStoreInNLLData
    bias = data + 16
    scale_alpha = data + 20
    lds_addr = base_vaddr + 6
    alpha = writer.states.subtileStoreInNLLAlpha
    stride = writer.states.subtileStoreInNLLStride
    stride16 = writer.states.subtileStoreInNLLStride16
    flags = nonTemporalDFlags(kernel["NonTemporalD"])
    insts = [VLShiftLeftB32(
        dst=vgpr(lds_addr),
        shiftHex=hex(2),
        src=vgpr(writer.states.subtileStoreInNLLLocalM),
        comment="Fly bias/scale-alpha LDS byte address")]
    for sl in range(2):
        bias = data + 16 + sl * 8
        scale_alpha = bias + 4
        insts.extend([
            DSLoadB128(
                dst=vgpr(bias, 4),
                src=vgpr(lds_addr),
                ds=DSModifiers(offset=sl * 256),
                comment=f"preload Fly bias rows sl={sl}"),
            DSLoadB128(
                dst=vgpr(scale_alpha, 4),
                src=vgpr(lds_addr),
                ds=DSModifiers(offset=1024 + sl * 256),
                comment=f"preload Fly scale-alpha rows sl={sl}"),
        ])
    insts.append(SWaitCnt(
        dscnt=0, vlcnt=-1, vscnt=-1,
        comment="wait for all Fly bias/scale-alpha rows"))

    for component in range(4):
        for q, tile in enumerate(dtileInfo.vgprTiles):
            if tile.regList.pool == writer.vgprPool:
                raise RuntimeError(
                    "Fly N-major stores do not support accumulator spill to VGPR")
            insts.append(VAccvgprReadB32(
                dst=vgpr(stage + q),
                src=accvgpr(tile.regList.indices[component]),
                comment=f"stage Fly component={component} q={q}"))

        for sl in range(2):
            bias = data + 16 + sl * 8
            scale_alpha = bias + 4
            for tile_n in range(8):
                q = flyNMajorProducerQ(tile_n, sl, 0)
                for pair in range(2):
                    raw = stage + q + pair * 2
                    insts.extend([
                        VMulPKF32(
                            dst=vgpr(raw, 2),
                            src0=vgpr(alpha, 2),
                            src1=vgpr(raw, 2),
                            vop3=VOP3PModifiers(op_sel_hi=[0, 1, 1]),
                            comment="apply alpha to staged Fly rows"),
                        VMulPKF32(
                            dst=vgpr(raw, 2),
                            src0=vgpr(scale_alpha + pair * 2, 2),
                            src1=vgpr(raw, 2),
                            comment="apply scale-alpha to staged Fly rows"),
                        VAddPKF32(
                            dst=vgpr(raw, 2),
                            src0=vgpr(bias + pair * 2, 2),
                            src1=vgpr(raw, 2),
                            comment="add bias to staged Fly rows"),
                        VCvtPkF32toBF16(
                            dst=vgpr(data + pair),
                            src0=vgpr(raw),
                            src1=vgpr(raw + 1),
                            comment="pack staged Fly BF16 rows"),
                    ])

                insts.extend([
                    VMulLOU32(
                        dst=vgpr(address),
                        src0=tile_n,
                        src1=vgpr(stride16),
                        comment=f"Fly tile {tile_n} column-byte offset"),
                    VAddU32(
                        dst=vgpr(address),
                        src0=vgpr(address),
                        src1=vgpr(base_vaddr),
                        comment="add Fly D lane base"),
                ])
                for _ in range(component):
                    insts.append(VAddU32(
                        dst=vgpr(address),
                        src0=vgpr(address),
                        src1=vgpr(stride),
                        comment="advance Fly address by one D column"))
                insts.append(BufferStoreB64(
                    src=vgpr(data, 2),
                    vaddr=vgpr(address),
                    saddr=sgpr("SrdD", 4),
                    soffset=0,
                    mubuf=MUBUFModifiers(
                        offen=True,
                        offset12=(
                            sl * writer.states.subtileStoreInNLLSecondHalfOffset),
                        glc=flags["glc"], slc=flags["slc"], nt=flags["nt"]),
                    comment=(
                        f"staged Fly dwordx2 tile={tile_n} "
                        f"sl={sl} e={component}")))
    return insts


def _store_fly_nmajor_identity_staged_drain(
        writer, kernel, dtileInfo, stage: int) -> list:
    """Store alpha-one output directly when optional vectors are absent."""
    base_vaddr = writer.states.subtileStoreInNLLVaddr
    address = base_vaddr + 1
    data = writer.states.subtileStoreInNLLData
    stride = writer.states.subtileStoreInNLLStride
    stride16 = writer.states.subtileStoreInNLLStride16
    flags = nonTemporalDFlags(kernel["NonTemporalD"])
    insts = []

    for component in range(4):
        for q, tile in enumerate(dtileInfo.vgprTiles):
            if tile.regList.pool == writer.vgprPool:
                raise RuntimeError(
                    "Fly N-major stores do not support accumulator spill to VGPR")
            insts.append(VAccvgprReadB32(
                dst=vgpr(stage + q),
                src=accvgpr(tile.regList.indices[component]),
                comment=f"stage identity Fly component={component} q={q}"))

        for sl in range(2):
            for tile_n in range(8):
                raw = stage + flyNMajorProducerQ(tile_n, sl, 0)
                insts.extend([
                    VCvtPkF32toBF16(
                        dst=vgpr(data),
                        src0=vgpr(raw),
                        src1=vgpr(raw + 1),
                        comment="pack identity Fly BF16 rows 0:2"),
                    VCvtPkF32toBF16(
                        dst=vgpr(data + 1),
                        src0=vgpr(raw + 2),
                        src1=vgpr(raw + 3),
                        comment="pack identity Fly BF16 rows 2:4"),
                    VMulLOU32(
                        dst=vgpr(address),
                        src0=tile_n,
                        src1=vgpr(stride16),
                        comment=f"Fly tile {tile_n} column-byte offset"),
                    VAddU32(
                        dst=vgpr(address),
                        src0=vgpr(address),
                        src1=vgpr(base_vaddr),
                        comment="add Fly D lane base"),
                ])
                for _ in range(component):
                    insts.append(VAddU32(
                        dst=vgpr(address),
                        src0=vgpr(address),
                        src1=vgpr(stride),
                        comment="advance Fly address by one D column"))
                insts.append(BufferStoreB64(
                    src=vgpr(data, 2),
                    vaddr=vgpr(address),
                    saddr=sgpr("SrdD", 4),
                    soffset=0,
                    mubuf=MUBUFModifiers(
                        offen=True,
                        offset12=(
                            sl * writer.states.subtileStoreInNLLSecondHalfOffset),
                        glc=flags["glc"], slc=flags["slc"], nt=flags["nt"]),
                    comment=(
                        f"identity Fly dwordx2 tile={tile_n} "
                        f"sl={sl} e={component}")))
    return insts


def emitFlyNMajorVectorPreload(writer) -> Module:
    """Load both folded-store vector halves after their LDS initialization."""
    module = Module("FlyNMajorVectorPreload")
    base_vaddr = writer.states.subtileStoreInNLLVaddr
    data = writer.states.subtileStoreInNLLData
    lds_addr = base_vaddr + 6
    module.add(VLShiftLeftB32(
        dst=vgpr(lds_addr),
        shiftHex=hex(2),
        src=vgpr(writer.states.subtileStoreInNLLLocalM),
        comment="Fly bias/scale-alpha LDS byte address"))
    for sl in range(2):
        bias = data + 16 + sl * 8
        scale_alpha = bias + 4
        module.add(DSLoadB128(
            dst=vgpr(bias, 4),
            src=vgpr(lds_addr),
            ds=DSModifiers(offset=sl * 256),
            comment=f"preload Fly bias rows sl={sl}"))
        module.add(DSLoadB128(
            dst=vgpr(scale_alpha, 4),
            src=vgpr(lds_addr),
            ds=DSModifiers(offset=1024 + sl * 256),
            comment=f"preload Fly scale-alpha rows sl={sl}"))
    module.add(SWaitCnt(
        dscnt=0, vlcnt=-1, vscnt=-1,
        comment="wait for all Fly bias/scale-alpha rows"))
    return module


def emitFlyNMajorTileStage(
        writer, dtileInfo, tile_n: int, stage: int) -> Module:
    """Stage one N-major tile's 32 accumulator values into a VGPR bank."""
    module = Module(f"FlyNMajorTile{tile_n}Stage")
    module.add(SNop(
        waitState=7,
        comment=f"wait for tile {tile_n} final MFMA accumulator writes"))
    for component in range(4):
        for sl in range(2):
            for ji in range(4):
                q = flyNMajorProducerQ(tile_n, sl, ji)
                tile = dtileInfo.vgprTiles[q]
                if tile.regList.pool == writer.vgprPool:
                    raise RuntimeError(
                        "Fly N-major stores do not support accumulator spill to VGPR")
                dst = stage + component * 8 + sl * 4 + ji
                module.add(VAccvgprReadB32(
                    dst=vgpr(dst),
                    src=accvgpr(tile.regList.indices[component]),
                    comment=(
                        f"stage Fly tile={tile_n} sl={sl} "
                        f"e={component} ji={ji}")))
    return module


def emitFlyNMajorIdentityCellRetire(
        writer, kernel, dtileInfo, tile_n: int, tile_m: int,
        stage: int) -> Module:
    """Retire one final producer and flush its four-cell packet when complete."""
    module = Module(f"FlyNMajorIdentityCell{tile_m}_{tile_n}")
    sl, ji = divmod(tile_m, 4)
    bank = tile_n & 1
    raw = stage + bank * 16
    q = flyNMajorProducerQ(tile_n, sl, ji)
    tile = dtileInfo.vgprTiles[q]
    if tile.regList.pool == writer.vgprPool:
        raise RuntimeError(
            "Fly N-major stores do not support accumulator spill to VGPR")
    for component in range(4):
        module.add(VAccvgprReadB32(
            dst=vgpr(raw + component * 4 + ji),
            src=accvgpr(tile.regList.indices[component]),
            comment=(
                f"retire Fly tile={tile_n} sl={sl} "
                f"e={component} ji={ji}")))
    if ji != 3:
        return module

    data = writer.states.subtileStoreInNLLData
    for component in range(4):
        for pair in range(2):
            module.add(VCvtPkF32toBF16(
                dst=vgpr(data + component * 2 + pair),
                src0=vgpr(raw + component * 4 + pair * 2),
                src1=vgpr(raw + component * 4 + pair * 2 + 1),
                comment="pack retired Fly BF16 pair"))

    base_vaddr = writer.states.subtileStoreInNLLVaddr
    addresses = base_vaddr + 1
    stride = writer.states.subtileStoreInNLLStride
    stride16 = writer.states.subtileStoreInNLLStride16
    if sl == 0:
        module.add(VMulLOU32(
            dst=vgpr(addresses),
            src0=tile_n,
            src1=vgpr(stride16),
            comment=f"Fly tile {tile_n} column-byte offset"))
        module.add(VAddU32(
            dst=vgpr(addresses),
            src0=vgpr(addresses),
            src1=vgpr(base_vaddr),
            comment=f"Fly tile {tile_n} first D address"))
        for component in range(1, 4):
            module.add(VAddU32(
                dst=vgpr(addresses + component),
                src0=vgpr(addresses + component - 1),
                src1=vgpr(stride),
                comment="advance Fly address by one D column"))

    flags = nonTemporalDFlags(kernel["NonTemporalD"])
    for component in range(4):
        module.add(BufferStoreB64(
            src=vgpr(data + component * 2, 2),
            vaddr=vgpr(addresses + component),
            saddr=sgpr("SrdD", 4),
            soffset=0,
            mubuf=MUBUFModifiers(
                offen=True,
                offset12=(
                    sl * writer.states.subtileStoreInNLLSecondHalfOffset),
                glc=flags["glc"], slc=flags["slc"], nt=flags["nt"]),
            comment=(
                f"retired Fly dwordx2 tile={tile_n} "
                f"sl={sl} e={component}")))
    return module


def emitFlyNMajorTileConsume(
        writer, kernel, tile_n: int, stage: int) -> Module:
    """Convert and store one staged N-major tile."""
    module = Module(f"FlyNMajorTile{tile_n}Consume")
    base_vaddr = writer.states.subtileStoreInNLLVaddr
    addresses = base_vaddr + 1
    data = writer.states.subtileStoreInNLLData
    alpha = writer.states.subtileStoreInNLLAlpha
    stride = writer.states.subtileStoreInNLLStride
    stride16 = writer.states.subtileStoreInNLLStride16
    flags = nonTemporalDFlags(kernel["NonTemporalD"])

    module.add(VMulLOU32(
        dst=vgpr(addresses),
        src0=tile_n,
        src1=vgpr(stride16),
        comment=f"Fly tile {tile_n} column-byte offset"))
    module.add(VAddU32(
        dst=vgpr(addresses),
        src0=vgpr(addresses),
        src1=vgpr(base_vaddr),
        comment=f"Fly tile {tile_n} first D address"))
    for component in range(1, 4):
        module.add(VAddU32(
            dst=vgpr(addresses + component),
            src0=vgpr(addresses + component - 1),
            src1=vgpr(stride),
            comment="advance Fly address by one D column"))

    for sl in range(2):
        bias = data + 16 + sl * 8
        scale_alpha = bias + 4
        for component in range(4):
            raw = stage + component * 8 + sl * 4
            packed = data + component * 2
            for pair in range(2):
                row_pair = raw + pair * 2
                module.add(VMulPKF32(
                    dst=vgpr(row_pair, 2),
                    src0=vgpr(alpha, 2),
                    src1=vgpr(row_pair, 2),
                    vop3=VOP3PModifiers(op_sel_hi=[0, 1, 1]),
                    comment="apply alpha to staged Fly rows"))
                module.add(VMulPKF32(
                    dst=vgpr(row_pair, 2),
                    src0=vgpr(scale_alpha + pair * 2, 2),
                    src1=vgpr(row_pair, 2),
                    comment="apply scale-alpha to staged Fly rows"))
                module.add(VAddPKF32(
                    dst=vgpr(row_pair, 2),
                    src0=vgpr(bias + pair * 2, 2),
                    src1=vgpr(row_pair, 2),
                    comment="add bias to staged Fly rows"))
                module.add(VCvtPkF32toBF16(
                    dst=vgpr(packed + pair),
                    src0=vgpr(row_pair),
                    src1=vgpr(row_pair + 1),
                    comment="pack staged Fly BF16 rows"))
        for component in range(4):
            module.add(BufferStoreB64(
                src=vgpr(data + component * 2, 2),
                vaddr=vgpr(addresses + component),
                saddr=sgpr("SrdD", 4),
                soffset=0,
                mubuf=MUBUFModifiers(
                    offen=True,
                    offset12=(
                        sl * writer.states.subtileStoreInNLLSecondHalfOffset),
                    glc=flags["glc"], slc=flags["slc"], nt=flags["nt"]),
                comment=(
                    f"interleaved Fly dwordx2 tile={tile_n} "
                    f"sl={sl} e={component}")))
    return module


def _alpha_one_store_pairs(writer, kernel, dtileInfo, columns: list) -> list:
    """Pipeline no-multiply stores in pairs after the final MFMA."""
    flags = nonTemporalDFlags(kernel["NonTemporalD"])
    mubuf = lambda: MUBUFModifiers(
        offen=True, offset12=0, glc=flags["glc"], slc=flags["slc"], nt=flags["nt"])
    data_base = writer.states.subtileStoreInNLLData
    vaddr = writer.states.subtileStoreInNLLVaddr
    insts = []

    for pair_start in range(0, len(columns), 2):
        pair = columns[pair_start:pair_start + 2]
        for bank, column in enumerate(pair):
            data = data_base + bank * 8
            for row, reg in enumerate(
                    _column_accumulators(writer, dtileInfo, column)):
                insts.append(VAccvgprReadB32(
                    dst=vgpr(data + row), src=accvgpr(reg),
                    comment=f"fast fold C column {column} row {row}: acc -> vgpr"))
        for bank, column in enumerate(pair):
            data = data_base + bank * 8
            for packed_pair in range(4):
                insts.append(VCvtPkF32toBF16(
                    dst=vgpr(data + packed_pair),
                    src0=vgpr(data + 2 * packed_pair),
                    src1=vgpr(data + 2 * packed_pair + 1),
                    comment="pack alpha-one folded BF16 rows"))
            insts.append(BufferStoreB128(
                src=vgpr(data, 4),
                vaddr=vgpr(vaddr),
                saddr=sgpr("SrdD", 4),
                soffset=0,
                mubuf=mubuf(),
                comment=f"fast folded D store column {column}"))
            if column != 31:
                insts.append(VAddU32(
                    dst=vgpr(vaddr), src0=vgpr(vaddr),
                    src1=vgpr(writer.states.subtileStoreInNLLStride),
                    comment="advance folded-store address by one BF16 D column"))
    return insts


def _emit_deferred_store_paths(writer, kernel, dtileInfo,
                               columns: list) -> Module:
    """Emit one runtime alpha branch for the post-MFMA store drain."""
    module = Module("SubtileStoreInNLLDeferred")
    if not columns:
        return module
    general = Label(
        writer.labels.getNameInc("SubtileStoreInNLLGeneralAlpha"), "")
    done = Label(
        writer.labels.getNameInc("SubtileStoreInNLLDeferredDone"), "")
    module.add(SCmpEQU32(
        src0=sgpr("Alpha"), src1=1.0,
        comment="use pipelined store drain when alpha is one"))
    module.add(SCBranchSCC0(
        labelName=general.getLabelName(),
        comment="alpha != 1.0"))
    module.addItems(_alpha_one_store_pairs(
        writer, kernel, dtileInfo, columns))
    module.add(SBranch(
        labelName=done.getLabelName(),
        comment="skip general-alpha store drain"))
    module.add(general)
    for column in columns:
        module.addItems(_store_unit(writer, kernel, dtileInfo, column))
    module.add(done)
    return module


def emitSubtileStoreInNLLGroupedDrain(writer, kernel, dtileInfo) -> Module:
    """Emit a Fly N-major drain outside the scheduled scaled-MFMA block."""
    module = Module("SubtileStoreInNLLGroupedDrain")
    if not kernel.get("SubtileStoreInNLL", 0):
        return module
    if not kernel.get("SubtileFlyNMajor", 0):
        raise RuntimeError(
            "deferred grouped drain requires SubtileFlyNMajor")
    stage = writer.vgprPool.checkOutAligned(
        64, 2, "subtileStoreInNLLStage")
    try:
        if kernel.get("SubtileIdentityStoreFastPath", 0):
            general = Label(
                writer.labels.getNameInc("SubtileStoreGeneralEpilogue"), "")
            done = Label(
                writer.labels.getNameInc("SubtileStoreEpilogueDone"), "")
            for register, comment in (
                    ("Alpha", "alpha must be one"),
                    ("AddressBias", "bias must be absent"),
                    ("AddressScaleAlphaVec", "scale-alpha must be absent")):
                if register == "Alpha":
                    module.add(SCmpEQU32(
                        src0=sgpr(register), src1=1.0, comment=comment))
                else:
                    module.add(SCmpEQU64(
                        src0=sgpr(writer.sgprs[register], 2),
                        src1=0, comment=comment))
                module.add(SCBranchSCC0(
                    labelName=general.getLabelName(),
                    comment="use general folded-store epilogue"))
            module.addItems(_store_fly_nmajor_identity_staged_drain(
                writer, kernel, dtileInfo, stage))
            module.add(SBranch(
                labelName=done.getLabelName(),
                comment="identity folded-store epilogue complete"))
            module.add(general)
            module.addItems(_store_fly_nmajor_staged_drain(
                writer, kernel, dtileInfo, stage))
            module.add(done)
        else:
            module.addItems(_store_fly_nmajor_staged_drain(
                writer, kernel, dtileInfo, stage))
    finally:
        writer.vgprPool.checkIn(stage)
    return module


def interleaveSubtileStoreInNLL(
    module: Module,
    writer,
    kernel,
    dtileInfo,
    minMfmaGap: int = 4,
    issueRate: int = 8,
    fourRowPackets: bool = False,
) -> Module:
    """Pace completed output-column stores through a scheduled NLL MFMA stream."""
    if not kernel.get("SubtileStoreInNLL", 0):
        return module

    items = list(module.flatitems())
    mfmas = []
    for item_index, item in enumerate(items):
        match = _MFMA_TILE_RE.search(str(item))
        if match:
            mfmas.append((item_index, int(match.group(1)), int(match.group(2))))
    if not mfmas:
        raise RuntimeError("SubtileStoreInNLL found no NLL MFMA producers")

    last_producer = {}
    for seq, (_, tile_m, tile_n) in enumerate(mfmas):
        last_producer[(tile_m, tile_n)] = seq
    last_seq = len(mfmas) - 1

    units = []
    groupedPackets = fourRowPackets or kernel.get("SubtileFlyNMajor", 0)
    if groupedPackets:
        for tile_n in range(dtileInfo.localMMATileGrid[1]):
            producer = max(last_producer[(tile_m, tile_n)]
                           for tile_m in range(
                               dtileInfo.localMMATileGrid[0]))
            release = min(producer + minMfmaGap, last_seq)
            units.append({
                "release": release,
                "column": tile_n,
                "row_block": 0,
                "instructions": deque(
                    _store_fly_nmajor_group(
                        writer, kernel, dtileInfo, tile_n)
                    if kernel.get("SubtileFlyNMajor", 0)
                    else (
                        _store_final_wide_group(
                            writer, kernel, dtileInfo, tile_n)
                        if tile_n == dtileInfo.localMMATileGrid[1] - 1
                        else _store_recoalesced_packet_group(
                            writer, kernel, dtileInfo, tile_n))),
                "started": False,
            })
    else:
        for column in range(32):
            tile_n = column % dtileInfo.localMMATileGrid[1]
            producer = max(last_producer[(tile_m, tile_n)]
                           for tile_m in range(
                               dtileInfo.localMMATileGrid[0]))
            release = min(producer + minMfmaGap, last_seq)
            units.append({
                "release": release,
                "column": column,
                "row_block": 0,
                "instructions": deque(
                    _store_unit(writer, kernel, dtileInfo, column)),
                "started": False,
            })
    if groupedPackets:
        units.sort(key=lambda unit: (unit["release"], unit["row_block"],
                                     unit["column"]))

    out = Module(module.name)
    pending = deque()
    next_unit = 0
    mfma_seq = -1
    for item in items:
        out.add(item)
        if not _MFMA_TILE_RE.search(str(item)):
            continue
        mfma_seq += 1
        while (next_unit < len(units)
               and units[next_unit]["release"] <= mfma_seq):
            pending.append(units[next_unit])
            next_unit += 1

        budget = issueRate
        while pending and budget:
            unit = pending[0]
            unit["started"] = True
            out.add(unit["instructions"].popleft())
            budget -= 1
            if not unit["instructions"]:
                pending.popleft()

        if mfma_seq == last_seq:
            while next_unit < len(units):
                pending.append(units[next_unit])
                next_unit += 1
            while pending and pending[0]["started"]:
                unit = pending.popleft()
                while unit["instructions"]:
                    out.add(unit["instructions"].popleft())
            if groupedPackets:
                while pending:
                    unit = pending.popleft()
                    while unit["instructions"]:
                        out.add(unit["instructions"].popleft())
            else:
                out.add(_emit_deferred_store_paths(
                    writer, kernel, dtileInfo,
                    [unit["column"] for unit in pending]))
            pending.clear()
    return out
