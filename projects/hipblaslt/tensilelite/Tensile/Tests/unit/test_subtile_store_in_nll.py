# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from unittest.mock import MagicMock

import pytest

from Tensile.Common.GlobalParameters import defaultSolution
from Tensile.Common.RequiredParameters import getRequiredParametersMin
from Tensile.Common.ValidParameters import validParameters
from Tensile.Components.Subtile.SubtileGeometry import (
    flyAParityLdsEnabled,
    flyAParityReadAddress,
    flyAParityReadBanks,
    flyAParityWriteMapping,
    subtileInterleaveVW,
    subtileLdsBlockPadBytes,
)
from Tensile.Components.Subtile.SubtileScaleEmit import (
    scaleDirectDeferredWaitEnabled,
    scaleDirectToVgprEnabled,
    scaleGroupBlockedOffset,
    mxsaDirectCanonicalByte,
    mxsaDirectCanonicalByteFromConsumerQ,
)
from Tensile.Components.Subtile.StoreInNLL import (
    NLLStoreEvent,
    flyStorePacketGeometryEnabled,
    flyStorePacketCoordinates,
    nonTemporalDFlags,
    planNLLStoreInterleave,
    subtileFlyStorePacketGeometryRejectionReason,
    subtileStoreInNLLRejectionReason,
)


def test_blocked_scale_groups_advance_across_lds():
    tile_info = MagicMock()
    tile_info.lrSubtileSize = 256
    assert [scaleGroupBlockedOffset(tile_info, group) for group in range(4)] == [
        0, 256, 512, 768,
    ]


def test_fly_map_keeps_exact_block_padding_for_sub_block_lane_strides(monkeypatch):
    monkeypatch.delenv("TENSILE_SUBTILE_LDS_BLOCK_PAD", raising=False)
    kernel = {"SourceSwap": True, "SubtileFlyNMajor": 1}
    assert subtileLdsBlockPadBytes(kernel, 128) == 32


def test_fly_a_parity_write_read_mapping_and_bank_spread():
    seen = set()
    for wave_id in range(4):
        for tile in range(8):
            for lane_row in range(8):
                logical_row, block_base = flyAParityWriteMapping(
                    wave_id, tile, lane_row)
                lane16 = (wave_id & 1) * 8 + lane_row
                for physical_chunk in range(8):
                    logical_chunk = (physical_chunk - lane16) & 7
                    write_address = (
                        block_base + lane_row * 128
                        + physical_chunk * 16
                    )
                    read_address = (
                        (wave_id >> 1) * 16384
                        + flyAParityReadAddress(
                            lane16, tile, logical_chunk)
                    )
                    assert write_address == read_address
                    seen.add((logical_row, logical_chunk))
    assert len(seen) == 256 * 8

    # A4's unswizzled 512-byte lane stride aliases every lane on one
    # start bank. The parity map rotates by region-row and reaches 8.
    assert len(set(flyAParityReadBanks(tile=0, chunk=0))) == 8


def test_fly_a_parity_gate_is_default_off_and_fail_closed():
    assert not flyAParityLdsEnabled({})
    with pytest.raises(ValueError, match="requires the gfx950 SS1"):
        flyAParityLdsEnabled({"SubtileFlyAParityLds": 1})


def _bf16():
    dtype = MagicMock()
    dtype.isBFloat16.return_value = True
    return dtype


def _admissible_state():
    return {
        "SubtileStoreInNLL": 1,
        "SubtileRollingPeel": 1,
        "SubtileFlyNMajor": 1,
        "ISA": (9, 5, 0),
        "UseSubtileImpl": True,
        "SourceSwap": True,
        "CompactLoopStore": False,
        "StreamK": 0,
        "GlobalSplitU": 1,
        "MIWaveTile": [8, 8],
        "MIArchVgpr": False,
        "NonTemporalD": 4,
        "MacroTile0": 256,
        "MacroTile1": 256,
        "DepthU": 256,
        "MatrixInstM": 16,
        "MatrixInstK": 128,
        "MIWaveGroup": [2, 2],
        "AssertFree0ElementMultiple": 256,
        "AssertFree1ElementMultiple": 256,
        "ProblemType": {
            "UseBeta": False,
            "HighPrecisionAccumulate": True,
            "DestDataType": _bf16(),
            "UseBias": False,
            "UseE": False,
            "UseScaleD": False,
            "UseScaleCD": False,
            "UseScaleAlphaVec": False,
            "OutputAmaxD": False,
            "ActivationType": "none",
        },
    }


def test_parameter_is_explicit_serialized_and_disabled_by_default():
    assert validParameters["SubtileStoreInNLL"] == [0, 1]
    assert defaultSolution["SubtileStoreInNLL"] == 0
    assert "SubtileStoreInNLL" in getRequiredParametersMin()
    assert validParameters["SubtileRollingPeel"] == [0, 1]
    assert defaultSolution["SubtileRollingPeel"] == 0
    assert "SubtileRollingPeel" in getRequiredParametersMin()
    assert validParameters["SubtileFlyNMajor"] == [0, 1]
    assert defaultSolution["SubtileFlyNMajor"] == 0
    assert "SubtileFlyNMajor" in getRequiredParametersMin()
    assert validParameters["SubtileFlyAParityLds"] == [0, 1]
    assert defaultSolution["SubtileFlyAParityLds"] == 0
    assert "SubtileFlyAParityLds" in getRequiredParametersMin()
    assert validParameters["SubtileFlyStorePacketGeometry"] == [0, 1]
    assert defaultSolution["SubtileFlyStorePacketGeometry"] == 0
    assert "SubtileFlyStorePacketGeometry" in getRequiredParametersMin()
    assert validParameters["SubtileDirectToVgprScale"] == [0, 1, 2, 3]
    assert defaultSolution["SubtileDirectToVgprScale"] == 0
    assert "SubtileDirectToVgprScale" in getRequiredParametersMin()
    assert validParameters["SubtileDirectScaleDeferredWait"] == [0, 1]
    assert defaultSolution["SubtileDirectScaleDeferredWait"] == 0
    assert "SubtileDirectScaleDeferredWait" in getRequiredParametersMin()
    assert validParameters["SubtileNativeWholeLoopScheduler"] == [0, 1]
    assert defaultSolution["SubtileNativeWholeLoopScheduler"] == 0
    assert "SubtileNativeWholeLoopScheduler" in getRequiredParametersMin()
    assert "LaneContiguousGFX950" in validParameters["MXScaleFormat"]


@pytest.mark.parametrize(
    "mask,direct_a,direct_b,expected_a,expected_b",
    [
        (0, True, True, False, False),
        (1, True, False, True, False),
        (2, False, True, False, True),
        (3, True, True, True, True),
        (1, False, True, False, False),
        (2, True, False, False, False),
    ],
)
def test_direct_scale_bitmask_requires_selected_dtv_metadata(
        mask, direct_a, direct_b, expected_a, expected_b):
    kernel = {
        "SubtileDirectToVgprScale": mask,
        "DirectToVgprMXSA": direct_a,
        "DirectToVgprMXSB": direct_b,
    }
    assert scaleDirectToVgprEnabled(kernel) == (expected_a or expected_b)
    assert scaleDirectToVgprEnabled(kernel, "MXSA") == expected_a
    assert scaleDirectToVgprEnabled(kernel, "MXSB") == expected_b
    assert scaleDirectToVgprEnabled(kernel, "A") == expected_a
    assert scaleDirectToVgprEnabled(kernel, "B") == expected_b


def test_direct_scale_deferred_wait_is_blocked_b_only():
    kernel = {
        "SubtileDirectToVgprScale": 2,
        "SubtileDirectScaleDeferredWait": 1,
        "DirectToVgprMXSA": False,
        "DirectToVgprMXSB": True,
        "SubtileFlyNMajor": 1,
        "SourceSwap": True,
    }
    assert scaleDirectDeferredWaitEnabled(kernel, "MXSB")
    assert not scaleDirectDeferredWaitEnabled(kernel, "MXSA")
    kernel["SubtileDirectScaleDeferredWait"] = 0
    assert not scaleDirectDeferredWaitEnabled(kernel, "MXSB")


def test_lane_contiguous_direct_loads_defer_both_operand_waits():
    kernel = {
        "MXScaleFormat": "LaneContiguousGFX950",
        "SubtileDirectToVgprScale": 3,
        "DirectToVgprMXSA": True,
        "DirectToVgprMXSB": True,
        "SourceSwap": True,
    }
    assert scaleDirectDeferredWaitEnabled(kernel, "MXSA")
    assert scaleDirectDeferredWaitEnabled(kernel, "MXSB")


def test_mxsa_direct_canonical_bytes_match_interleaved_lane_map():
    for wave_m in range(2):
        for lane in range(64):
            lane_row = lane % 16
            lane_k_pair = 2 * (lane // 16)
            for tile in range(8):
                expected_row = (
                    128 * wave_m + 4 * lane_row
                    + tile % 4 + 64 * (tile // 4)
                )
                q = (
                    1024 * wave_m
                    + 256 * (lane_row // 8)
                    + 64 * (lane // 16)
                    + 16 * (lane_row % 4)
                    + 4 * (tile % 4)
                    + 512 * (tile // 4)
                )
                for sub_iter_k in range(2):
                    expected = mxsaDirectCanonicalByte(
                        lane, wave_m, tile, sub_iter_k
                    )
                    assert expected == (
                        expected_row, lane_k_pair + sub_iter_k
                    )
                    assert mxsaDirectCanonicalByteFromConsumerQ(
                        q, lane, sub_iter_k
                    ) == expected


def test_fly_nmajor_uses_original_a_four_way_interleave():
    kernel = {
        "SubtileFlyNMajor": 1,
        "SourceSwap": True,
        "VectorWidthA": 1,
        "VectorWidthB": 4,
    }
    assert subtileInterleaveVW(kernel, "A") == 4
    assert subtileInterleaveVW(kernel, "B") == 1


def test_disabled_path_has_no_rejection():
    state = _admissible_state()
    state["SubtileStoreInNLL"] = 0
    assert subtileStoreInNLLRejectionReason(state) is None


@pytest.mark.parametrize(
    "mutate,reason",
    [
        (lambda s: s.update(ISA=(9, 4, 2)), "requires gfx950"),
        (lambda s: s.update(SourceSwap=False), "requires SourceSwap=1"),
        (lambda s: s.update(StreamK=3), "requires StreamK=0"),
        (lambda s: s.update(GlobalSplitU=0), "requires GlobalSplitU=1"),
        (lambda s: s.update(NonTemporalD=0), "requires NonTemporalD bit 0x4"),
        (lambda s: s.update(AssertFree0ElementMultiple=8), "MacroTile-aligned M"),
        (lambda s: s["ProblemType"].update(UseScaleD=True), "does not support scaleD"),
        (lambda s: s["ProblemType"].update(OutputAmaxD=True), "does not support amaxD"),
    ],
)
def test_narrow_scope_validation(mutate, reason):
    state = _admissible_state()
    mutate(state)
    assert reason in subtileStoreInNLLRejectionReason(state)


def test_admissible_scope_is_enabled():
    assert subtileStoreInNLLRejectionReason(_admissible_state()) is None


def test_fly_store_packet_gate_is_default_off_and_fail_closed():
    assert not flyStorePacketGeometryEnabled({})
    state = _admissible_state()
    state["SubtileFlyStorePacketGeometry"] = 1
    assert flyStorePacketGeometryEnabled(state)
    state["DepthU"] = 512
    assert "DepthU=256" in subtileFlyStorePacketGeometryRejectionReason(state)
    with pytest.raises(ValueError, match="DepthU=256"):
        flyStorePacketGeometryEnabled(state)


def test_fly_store_packets_cover_c_tile_once_without_overlap():
    """All 64 stores per wave/lane map onto one exact 256x256 C tile."""
    from Tensile.Components.Subtile.InstructionEmitter import (
        flyStoreAccumulatorIndex)

    owners = {}
    for wave_id in range(4):
        for lane in range(64):
            packet_bases = set()
            accumulator_owners = {}
            for tile_n in range(8):
                for sl in range(2):
                    for component in range(4):
                        coords = flyStorePacketCoordinates(
                            wave_id, lane, tile_n, sl, component)
                        assert len(coords) == 4
                        assert all(coords[ji][0] + 1 == coords[ji + 1][0]
                                   for ji in range(3))
                        packet_bases.add(coords[0])
                        for ji, coord in enumerate(coords):
                            accumulator = (
                                flyStoreAccumulatorIndex(
                                    sl * 4 + ji, tile_n),
                                component,
                            )
                            assert accumulator not in accumulator_owners
                            accumulator_owners[accumulator] = coord
                            assert coord not in owners
                            owners[coord] = (
                                wave_id, lane, tile_n, sl, component, ji)
            assert len(packet_bases) == 64
            assert len(accumulator_owners) == 256

    assert set(owners) == {
        (m, n) for m in range(256) for n in range(256)
    }
    # Regression for the traced defect: M-wave 1 starts at row 64, while
    # sl=1 starts at row 128. They must neither overlap nor exceed row 255.
    assert flyStorePacketCoordinates(1, 0, 0, 0, 0)[0] == (64, 0)
    assert flyStorePacketCoordinates(0, 0, 0, 1, 0)[0] == (128, 0)
    assert flyStorePacketCoordinates(1, 15, 0, 1, 0)[-1] == (255, 0)


def test_universal_problem_type_is_allowed_under_explicit_runtime_contract():
    state = _admissible_state()
    state["ProblemType"].update(
        UseBeta=True, UseBias=True, UseScaleAlphaVec=True,
        ActivationType="hipblaslt_all")
    assert subtileStoreInNLLRejectionReason(state) is None


def test_interleave_places_stores_after_final_producer_with_four_issue_gap():
    mfmas = ["a0", "a1", "a0", "a2", "a3", "a4", "a5", "a6"]
    stores = [
        NLLStoreEvent("store", "a0", "store-a0"),
        NLLStoreEvent("store", "a1", "store-a1"),
    ]
    events = planNLLStoreInterleave(mfmas, stores)

    for store in stores:
        store_idx = events.index(store)
        producer_indices = [
            idx for idx, event in enumerate(events)
            if event.kind == "mfma" and event.accumulator == store.accumulator
        ]
        assert store_idx > max(producer_indices)
        intervening = sum(
            event.kind == "mfma"
            for event in events[max(producer_indices) + 1:store_idx]
        )
        assert intervening == 4


def test_interleave_drains_late_store_before_nll_exit():
    store = NLLStoreEvent("store", "late", "convert-and-store")
    events = planNLLStoreInterleave(["early", "late", "other"], [store])
    assert events[-1] == store
    assert events.index(store) > max(
        idx for idx, event in enumerate(events)
        if event.kind == "mfma" and event.accumulator == "late"
    )


def test_non_temporal_d_bit_is_preserved_for_future_mubuf_store():
    assert nonTemporalDFlags(4) == {"glc": False, "slc": False, "nt": True}
    assert nonTemporalDFlags(7) == {"glc": True, "slc": True, "nt": True}
