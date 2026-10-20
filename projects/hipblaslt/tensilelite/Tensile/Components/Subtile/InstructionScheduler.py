# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Instruction scheduler for subtile-based mainloop.

Interleaves non-MFMA instructions between MFMAs using a slot-based placer
with pluggable scheduling rules.
"""

from typing import List, Tuple, Optional
from rocisa.code import Module
from rocisa.instruction import SWaitCnt, SBarrier, MFMAInstruction, MXMFMAInstruction, \
    LocalReadInstruction, GlobalReadInstruction, CommonInstruction


class _SlotPlacer:
    """Generic slot placement engine for interleaving instructions between MFMAs.

    Each interval (pair of adjacent MFMAs) has a configurable number of
    4-cycle placement slots.
    Rules are injected via callbacks:
      - validators: (placer, pos, inst) -> bool — reject invalid slots
      - adjusters:  (placer, limit, inst) -> limit — shift search start
      - onPlace:    (placer, pos, inst) -> None — update rule state after placement
    """

    def __init__(self, intervals: int, numModules: int,
                 pathOrders: List[List[int]],
                 validators=None, adjusters=None, onPlace=None,
                 slotsPerInterval: int = 2):
        self.slotsPerInterval = slotsPerInterval
        self.totalSlots = intervals * slotsPerInterval
        self._n = numModules
        self._prevInPath: List[int] = [-1] * numModules
        self._nextInPath: List[int] = [-1] * numModules
        for order in pathOrders:
            for a, b in zip(order, order[1:]):
                self._prevInPath[b] = a
                self._nextInPath[a] = b
        self._validators = validators or []
        self._adjusters = adjusters or []
        self._onPlace = onPlace

        self._placed: List[List[Tuple[int, object]]] = [[] for _ in range(self.totalSlots)]
        self._firstPos: List[Optional[int]] = [None] * numModules
        self._lastPos: List[Optional[int]] = [None] * numModules
        self.leftovers: List[Tuple[int, object]] = []

    # ── Placement ──

    def _canPlace(self, pos: int, inst) -> bool:
        if pos < 0 or pos >= self.totalSlots or len(self._placed[pos]) >= 2:
            return False
        return all(v(self, pos, inst) for v in self._validators)

    def adjustLimit(self, limit: int, inst) -> int:
        for adj in self._adjusters:
            limit = adj(self, limit, inst)
        return limit

    def bounds(self, mid: int) -> Tuple[int, int]:
        lo = 0
        pred = self._prevInPath[mid]
        if 0 <= pred < self._n and self._lastPos[pred] is not None:
            lo = self._lastPos[pred] + 1
        hi = self.totalSlots - 1
        succ = self._nextInPath[mid]
        if 0 <= succ < self._n and self._firstPos[succ] is not None:
            hi = self._firstPos[succ] - 1
        return lo, hi

    def findSlot(self, mid: int, inst, limit: int, reverse: bool = False) -> Optional[int]:
        lo, hi = self.bounds(mid)
        if reverse:
            hi = min(hi, limit)
        else:
            lo = max(lo, limit)
        if hi < lo:
            return None
        for pos in (range(hi, lo - 1, -1) if reverse else range(lo, hi + 1)):
            if self._canPlace(pos, inst):
                return pos
        return None

    def _forceSlot(self, mid: int, limit: int, reverse: bool) -> int:
        """Find the closest valid slot respecting dependencies, allowing >2 items per slot."""
        lo, hi = self.bounds(mid)
        if reverse:
            hi = min(hi, limit)
            lo = max(lo, 0)
            if hi < lo:
                hi = lo
            return hi
        else:
            lo = max(lo, limit)
            hi = min(hi, self.totalSlots - 1)
            if lo > hi:
                lo = hi
            return lo

    def place(self, pos: int, item: Tuple[int, object], reverse: bool = False):
        mid = item[0]
        if reverse:
            self._placed[pos].insert(0, item)
        else:
            self._placed[pos].append(item)
        if self._firstPos[mid] is None or pos < self._firstPos[mid]:
            self._firstPos[mid] = pos
        if self._lastPos[mid] is None or pos > self._lastPos[mid]:
            self._lastPos[mid] = pos
        if self._onPlace:
            self._onPlace(self, pos, item[1])

    def placePath(self, pathInsts: List[Tuple[int, object]], reverse: bool = False,
                  multiDU: bool = False, startLimit: Optional[int] = None,
                  overflowToTail: bool = False):
        """Place a sequence of (moduleId, instruction) items into slots.

        Walks pathInsts in order, applying adjusters (forward only) and
        finding valid slots. When no empty slot is found, force-places at
        the closest valid position respecting dependencies (allowing >2
        items per slot).

        When multiDU is True, wait_gr immediately followed by barrier is
        placed as an atomic pair in the same slot so other paths cannot
        insert m0 setup between them.

        startLimit can reserve leading slots for an already-placed prefix.
        overflowToTail preserves forward path order by placing the first
        unplaceable item and the rest of its path after the final MFMA,
        rather than collapsing them into one overfull slot.
        """
        limit = ((self.totalSlots - 1) if reverse else 0) \
            if startLimit is None else startLimit
        if not multiDU:
            for idx, item in enumerate(pathInsts):
                mid, inst = item
                if not reverse:
                    limit = self.adjustLimit(limit, inst)
                pos = self.findSlot(mid, inst, limit, reverse=reverse)
                if pos is None:
                    if overflowToTail and not reverse:
                        self.leftovers.extend(pathInsts[idx:])
                        return
                    pos = self._forceSlot(mid, limit, reverse)
                self.place(pos, item, reverse=reverse)
                limit = (pos - 1) if reverse else (pos + 1)
            return

        idx = 0
        while idx < len(pathInsts):
            item = pathInsts[idx]
            mid, inst = item
            if (not reverse and idx + 1 < len(pathInsts)
                    and _isWaitGr(inst) and _isBarrier(pathInsts[idx + 1][1])):
                if not reverse:
                    limit = self.adjustLimit(limit, inst)
                pos = self.findSlot(mid, inst, limit, reverse=reverse)
                if pos is None:
                    if overflowToTail:
                        self.leftovers.extend(pathInsts[idx:])
                        return
                    pos = self._forceSlot(mid, limit, reverse)
                self.place(pos, item, reverse=reverse)
                sync_item = pathInsts[idx + 1]
                if self._canPlace(pos, sync_item[1]):
                    self.place(pos, sync_item, reverse=reverse)
                else:
                    sync_pos = self.findSlot(sync_item[0], sync_item[1],
                                             pos + 1, reverse=reverse)
                    if sync_pos is None:
                        if overflowToTail:
                            self.leftovers.extend(pathInsts[idx + 1:])
                            return
                        sync_pos = self._forceSlot(sync_item[0], pos + 1, reverse)
                    self.place(sync_pos, sync_item, reverse=reverse)
                    pos = sync_pos
                limit = (pos - 1) if reverse else (pos + 1)
                idx += 2
                continue
            if not reverse:
                limit = self.adjustLimit(limit, inst)
            pos = self.findSlot(mid, inst, limit, reverse=reverse)
            if pos is None:
                if overflowToTail and not reverse:
                    self.leftovers.extend(pathInsts[idx:])
                    return
                pos = self._forceSlot(mid, limit, reverse)
            self.place(pos, item, reverse=reverse)
            limit = (pos - 1) if reverse else (pos + 1)
            idx += 1

    # ── Assembly ──

    def assemble(self, mfmas) -> Module:
        intervals = len(mfmas) - 1
        result = Module()
        result.add(mfmas[0])
        for i in range(intervals):
            slotBegin = self.slotsPerInterval * i
            slotEnd = slotBegin + self.slotsPerInterval
            for slot in range(slotBegin, slotEnd):
                for item in self._placed[slot]:
                    result.add(item[1])
            result.add(mfmas[i + 1])
        for _, inst in self.leftovers:
            result.add(inst)
        return result


# ── Scheduling rules ──

# Hardcoded gap to hide ds_read latency. TODO: compute this more accurately.
# gfx1250 needs a larger gap (8); other archs use 4.
_MIN_MFMA_GAP_DS_READ_TO_WAIT_DEFAULT = 4
_MIN_MFMA_GAP_DS_READ_TO_WAIT_GFX1250 = 8

_isDsRead = lambda x: isinstance(x, LocalReadInstruction)
_isBufferLoad = lambda x: isinstance(x, GlobalReadInstruction)
_isWaitCnt = lambda x: isinstance(x, SWaitCnt)
_isM0Update = lambda x: isinstance(x, CommonInstruction) and hasattr(x, 'dst') and hasattr(x.dst, 'regType') and x.dst.regType == 'm'
_isVALU = lambda x: isinstance(x, CommonInstruction) \
    and not isinstance(x, (GlobalReadInstruction, LocalReadInstruction)) \
    and hasattr(x, 'dst') and hasattr(x.dst, 'regType') \
    and x.dst.regType == 'v'
# Typed op detection (see SWaitCntEx.isWaitGr / SBarrier) rather than matching
# substrings in the instruction comment, which is brittle to comment edits.
_isWaitGr = lambda x: _isWaitCnt(x) and getattr(x, "isWaitGr", False)
_isBarrier = lambda x: isinstance(x, SBarrier)


class _SchedulingRules:
    """Scheduling rules for slot placement: validators, adjusters, and placement hooks.

    Owns all rule state (ds_read/waitcnt tracking, buffer-load spreading).
    Bound methods are passed as callbacks to _SlotPlacer.
    """

    def __init__(self, totalSlots: int, minGapDsReadToWait: int = _MIN_MFMA_GAP_DS_READ_TO_WAIT_DEFAULT):
        # Cross-path state
        self.lastDsReadPos = -1
        self.earliestWaitCntPos = totalSlots
        self.waitGrBarrierPos: Optional[int] = None
        self.maxDsReadAfterWaitGr = -1
        self.minGap = minGapDsReadToWait
        # Per-path state
        self._resetPath()

    def _resetPath(self):
        self.firstBufLoadPos: Optional[int] = None
        self.bufLoadIdx = 0
        self.bufLoadMaxSlot = 0
        self.numBufLoads = 0
        self.localReadIdx = 0
        self.numLocalReads = 0

    # ── Validators: (placer, pos, inst) -> bool ──

    def oneDsReadPerInterval(self, placer, pos, inst):
        """At most one ds_read per two-slot SIMD issue pair."""
        if not _isDsRead(inst):
            return True
        peer = pos ^ 1
        slots = [pos]
        if 0 <= peer < placer.totalSlots:
            slots.append(peer)
        return not any(_isDsRead(item[1])
                       for slot in slots for item in placer._placed[slot])

    def noDsReadOrValuInFirstSlot(self, placer, pos, inst):
        """Reserve each p0 interval's first slot from DS reads and VALU."""
        if pos % placer.slotsPerInterval != 0:
            return True
        return not (_isDsRead(inst) or _isVALU(inst))

    def minGapDsReadBeforeWait(self, placer, pos, inst):
        """Reject ds_read too close to an already-placed waitcnt ahead."""
        if not _isDsRead(inst):
            return True
        if pos >= self.earliestWaitCntPos:
            return True
        gap = self.minGap * placer.slotsPerInterval
        return self.earliestWaitCntPos - pos >= gap

    def minGapDsReadToWait(self, placer, pos, inst):
        """Reject waitcnt too close to the last placed ds_read."""
        if not _isWaitCnt(inst) or self.lastDsReadPos < 0:
            return True
        if pos <= self.lastDsReadPos:
            return True
        gap = self.minGap * placer.slotsPerInterval
        return pos - self.lastDsReadPos >= gap

    def noM0WithBufferLoad(self, placer, pos, inst):
        """Avoid placing M0 updates and buffer_loads in the same MFMA interval."""
        if not _isM0Update(inst) and not _isBufferLoad(inst):
            return True
        peer = pos ^ 1
        slots = [pos]
        if 0 <= peer < placer.totalSlots:
            slots.append(peer)
        if _isM0Update(inst):
            return not any(_isBufferLoad(item[1]) for s in slots for item in placer._placed[s])
        return not any(_isM0Update(item[1]) for s in slots for item in placer._placed[s])

    def grAfterWaitGrDsReads(self, placer, pos, inst):
        """Keep GR (m0 / buffer_load) out of the window after wait_gr+barrier until ds_reads finish."""
        if not (_isBufferLoad(inst) or _isM0Update(inst)):
            return True
        if self.waitGrBarrierPos is None or self.maxDsReadAfterWaitGr < 0:
            return True
        if pos <= self.waitGrBarrierPos:
            return True
        return pos > self.maxDsReadAfterWaitGr

    # ── Adjusters: (placer, limit, inst) -> limit ──

    def deferGrAfterWaitGrDsReads(self, placer, limit, inst):
        """Once past wait_gr, start searching for GR only after post-barrier ds_reads."""
        if not (_isBufferLoad(inst) or _isM0Update(inst)):
            return limit
        if self.waitGrBarrierPos is None or self.maxDsReadAfterWaitGr < 0:
            return limit
        if limit <= self.waitGrBarrierPos:
            return limit
        return max(limit, self.maxDsReadAfterWaitGr + 1)

    def spreadBufferLoads(self, placer, limit, inst):
        """Spread buffer_load instructions evenly across available range."""
        if not _isBufferLoad(inst) or self.bufLoadMaxSlot <= 0:
            return limit
        if self.firstBufLoadPos is not None:
            stride = max(1, (self.bufLoadMaxSlot - self.firstBufLoadPos)
                         // self.numBufLoads)
            limit = max(limit, self.firstBufLoadPos
                        + self.bufLoadIdx * stride)
        self.bufLoadIdx += 1
        return limit

    def spreadLocalReads(self, placer, limit, inst):
        """Spread a selected LR path across all available 4-cycle slots."""
        if not _isDsRead(inst) or self.numLocalReads <= 1:
            return limit
        target = self.localReadIdx * (placer.totalSlots - 1) \
            // (self.numLocalReads - 1)
        self.localReadIdx += 1
        return max(limit, target)

    # ── Placement hook: (placer, pos, inst) -> None ──

    def trackPlacement(self, placer, pos, inst):
        """Update rule state after a successful placement."""
        if _isDsRead(inst):
            self.lastDsReadPos = max(self.lastDsReadPos, pos)
        if _isWaitCnt(inst):
            self.earliestWaitCntPos = min(self.earliestWaitCntPos, pos)
        if _isBufferLoad(inst) and self.firstBufLoadPos is None:
            self.firstBufLoadPos = pos

    # ── Per-path setup ──

    def resetPath(self):
        self._resetPath()

    def setupBufLoadSpreading(self, placer, pathInsts, order):
        """Compute buffer-load spreading bounds for a forward path.

        Reserves tail slots for non-buffer-load instructions in modules that
        follow the last GR module (e.g. GR_INC SRD updates, LDS buffer swaps).
        """
        self.numBufLoads = sum(1 for _, inst in pathInsts if _isBufferLoad(inst))
        if self.numBufLoads > 1:
            _, rawMax = placer.bounds(pathInsts[-1][0])
            grModuleIds = {mid for mid, inst in pathInsts if _isBufferLoad(inst)}
            lastGrIdx = max(order.index(m) for m in grModuleIds if m in order)
            tailModuleIds = set(order[lastGrIdx + 1:])
            numTailInsts = sum(1 for mid, _ in pathInsts if mid in tailModuleIds)
            # this is an approximation as we don't know exactly how many slots will be use by modules after the GR yet (in this codepath)
            self.bufLoadMaxSlot = max(0, rawMax - numTailInsts)

    def setupLocalReadSpreading(self, pathInsts):
        """Enable full-range spreading for the current local-read path."""
        self.numLocalReads = sum(
            1 for _, inst in pathInsts if _isDsRead(inst)
        )

    def setPostWaitGrDsReadFence(self, placer) -> None:
        """After the wait_gr path is placed, record ds_read coverage for GR deferral."""
        wg_pos = None
        for pos, slot in enumerate(placer._placed):
            for _, inst in slot:
                if _isWaitGr(inst):
                    wg_pos = pos if wg_pos is None else min(wg_pos, pos)
        if wg_pos is None:
            self.waitGrBarrierPos = None
            self.maxDsReadAfterWaitGr = -1
            return
        max_ds = -1
        for pos, slot in enumerate(placer._placed):
            if pos <= wg_pos:
                continue
            for _, inst in slot:
                if _isDsRead(inst):
                    max_ds = max(max_ds, pos)
        self.waitGrBarrierPos = wg_pos
        self.maxDsReadAfterWaitGr = max_ds


def _classifyPaths(pathOrders, emittedModules):
    """Classify paths by wait_gr presence, sorted: wait_gr first, then by index."""
    paths = []
    for order in pathOrders:
        hasWaitGR = any(emittedModules[i].opType == "wait_gr" for i in order)
        paths.append((order, hasWaitGR))
    paths.sort(key=lambda p: (0 if p[1] else 1, p[0][0] if p[0] else 10**9))
    return paths


def _flattenPath(order, emittedModules, reverse=False):
    """Flatten a path of module indices into (moduleId, instruction) pairs."""
    pathInsts = [(mid, inst) for mid in order for inst in emittedModules[mid].instructions]
    if reverse:
        pathInsts.reverse()
    return pathInsts


def _isWrapPrefetchPath(order, emittedModules):
    """Return True for a wait path that reads operands for the next MT.

    Partition-directed PLR marks the final partition's wrap local reads with
    mtIteration > 0. Those reads feed the next partition-0, not the current
    partition's MFMAs, so their payload can be scheduled after an early MFMA.
    """
    return any(
        emittedModules[mid].opType == "lr"
        and getattr(emittedModules[mid].source, "mtIteration", 0) > 0
        for mid in order
    )


def _findPartitionZeroPrefetchOrder(preMfmaOrders, emittedModules):
    """Find the exact 16-read p0 dual-period prefetch path."""
    for order in preMfmaOrders:
        modules = [emittedModules[mid] for mid in order]
        if not (any(em.opType == "wait_gr" for em in modules)
                and any(em.opType == "sync" for em in modules)):
            continue

        readModules = [em for em in modules if em.opType == "lr"]
        if not readModules:
            continue
        partitions = {
            getattr(em.source, "partition", None) for em in readModules
        }
        if partitions != {0}:
            continue

        hasNextPartitionData = any(
            getattr(em.source, "tensor", None) in ("A", "B")
            and getattr(em.source, "targetPartition", 0) == 1
            and getattr(em.source, "mtIteration", 0) == 0
            for em in readModules
        )
        scaleTensors = {
            getattr(em.source, "tensor", None)
            for em in readModules
            if any(_isDsRead(inst) for inst in em.instructions)
        } & {"SA", "SB"}
        readCount = sum(
            _isDsRead(inst)
            for em in readModules
            for inst in em.instructions
        )
        if not hasNextPartitionData or scaleTensors != {"SA", "SB"} \
                or readCount != 16:
            continue
        return order
    return None


def _setPartitionZeroPrefetchWait(preMfmaOrders, emittedModules):
    """Leave up to 15 current-partition reads outstanding at p0's wait."""
    order = _findPartitionZeroPrefetchOrder(
        preMfmaOrders, emittedModules)
    if order is None:
        return
    for mid in order:
        em = emittedModules[mid]
        if em.opType != "wait_lr":
            continue
        for inst in em.instructions:
            if _isWaitCnt(inst) and getattr(inst, "dscnt", -1) == 0:
                inst.dscnt = 15
                inst.comment = \
                    "Wait for prior LR; leave p0 prefetch reads in flight"


def _splitPartitionZeroPrefetch(preMfmaOrders, emittedModules):
    """Move p0's full LR drain before its 16 independent younger reads.

    Returns the matched order, a pre-MFMA wait/sync prefix, and the payload
    path to spread through four 4-cycle scheduling slots per MXMFMA.
    """
    order = _findPartitionZeroPrefetchOrder(
        preMfmaOrders, emittedModules)
    if order is None:
        return None

    prefixInsts = []
    for opType in ("wait_gr", "sync", "wait_lr"):
        for mid in order:
            em = emittedModules[mid]
            if em.opType != opType:
                continue
            for inst in em.instructions:
                if opType == "wait_lr" and _isWaitCnt(inst):
                    inst.dscnt = 0
                    inst.comment = "Wait for preloop LR before p0 MFMA"
                prefixInsts.append((mid, inst))

    payloadOrder = [
        mid for mid in order
        if emittedModules[mid].opType not in ("wait_gr", "sync", "wait_lr")
    ]
    payloadInsts = _flattenPath(payloadOrder, emittedModules)
    return order, prefixInsts, payloadOrder, payloadInsts


def _splitItemsAfterHalf(pathInsts, predicate):
    """Split an instruction path after half its matching instructions."""
    if not pathInsts:
        return [], []

    total = sum(predicate(inst) for _, inst in pathInsts)
    if total == 0:
        split = (len(pathInsts) + 1) // 2
        return pathInsts[:split], pathInsts[split:]

    target = (total + 1) // 2
    count = 0
    for index, (_, inst) in enumerate(pathInsts):
        count += predicate(inst)
        if count >= target:
            split = index + 1
            return pathInsts[:split], pathInsts[split:]
    return pathInsts, []


def _nextPlacementSlot(placer):
    """Return the first slot after all instructions placed so far."""
    for pos in range(placer.totalSlots - 1, -1, -1):
        if placer._placed[pos]:
            return pos + 1
    return 0


def _cmsMxfp8PartitionIndex(mfmaModule, mfmas):
    """Identify one quadrant of the integrated DU128 MXFP8 CMS profile."""
    if getattr(mfmaModule, "scheduleProfile", "") != "cms-mxfp8-du128":
        return None
    if len(mfmas) != 16 or not all(
            isinstance(inst, MXMFMAInstruction)
            and inst.getIssueLatency() == 8 for inst in mfmas):
        return None

    source = mfmaModule.source
    tileMaps = getattr(source, "vgpr_tile_maps", {})
    if not {"A", "B", "SA", "SB"}.issubset(tileMaps):
        return None

    tileA = tuple(getattr(getattr(source, "tileA", None),
                          "tileId_list", ()))
    tileB = tuple(getattr(getattr(source, "tileB", None),
                          "tileId_list", ()))
    quadrants = {
        ((0, 1, 2, 3), (0, 1, 2, 3)): 0,
        ((4, 5, 6, 7), (0, 1, 2, 3)): 1,
        ((0, 1, 2, 3), (4, 5, 6, 7)): 2,
        ((4, 5, 6, 7), (4, 5, 6, 7)): 3,
    }
    return quadrants.get((tileA, tileB))


def _groupLocalReadItems(emittedModule):
    """Map one data LR module's emitted instructions to its logical tiles."""
    tiles = emittedModule.source.tiles
    tileIds = list(range(tiles.tileId_start, tiles.tileId_end))
    if not tileIds or len(emittedModule.instructions) % len(tileIds):
        return {}
    readsPerTile = len(emittedModule.instructions) // len(tileIds)
    return {
        tile: emittedModule.instructions[
            index * readsPerTile:(index + 1) * readsPerTile]
        for index, tile in enumerate(tileIds)
    }


def _adjustVmcntForScheduledLoads(scheduled):
    """Preserve a wait's original request set after moving younger VMEM."""
    bufLoadCount = 0
    for inst in scheduled.flatitems():
        if _isBufferLoad(inst):
            bufLoadCount += 1
        elif _isWaitCnt(inst) and inst.vlcnt >= 0:
            if getattr(inst, 'adjustVmcnt', True):
                inst.vlcnt += bufLoadCount


def _scheduleCmsMxfp8Partition(
        emittedModules, mfmaIdx, pathOrders, mfmas):
    """Apply the integrated FP8-CMS structure to one MXFP8 quadrant.

    This is intentionally a complete profile rather than a stack of the
    isolated experiments.  It co-schedules the M-major MFMA traversal with:

      * first-use data/scale LDS reads,
      * wait/MFMA/barrier separation,
      * early SRD increments,
      * deadline-clustered global traffic, and
      * the reference schedule's compute-only middle and tail windows.

    Returns None for every non-matching loop variant so the generic scheduler
    remains unchanged.
    """
    partition = _cmsMxfp8PartitionIndex(
        emittedModules[mfmaIdx], mfmas)
    if partition is None:
        return None

    nonMfma = [
        (mid, em) for mid, em in enumerate(emittedModules)
        if mid != mfmaIdx
    ]

    def modules(opType, tensor=None):
        return [
            (mid, em) for mid, em in nonMfma
            if em.opType == opType
            and (tensor is None
                 or getattr(em.source, "tensor", None) == tensor)
        ]

    def exactlyOne(opType, tensor=None):
        matches = modules(opType, tensor)
        return matches[0] if len(matches) == 1 else None

    waitLr = exactlyOne("wait_lr")
    grByTensor = {
        tensor: exactlyOne("gr", tensor)
        for tensor in ("A", "B", "SA", "SB")
    }
    grIncByTensor = {
        tensor: exactlyOne("gr_inc", tensor)
        for tensor in ("A", "B", "SA", "SB")
    }
    lrByTensor = {
        tensor: modules("lr", tensor)
        for tensor in ("A", "B", "SA", "SB")
    }
    lrIncByTensor = {
        tensor: exactlyOne("lr_inc", tensor)
        for tensor in ("A", "B", "SA", "SB")
    }

    # Match only the steady-state profile. NGLL/NLL and unrelated scaled
    # kernels continue through the generic scheduler.
    signatureOk = waitLr is not None
    if partition == 0:
        signatureOk = signatureOk and exactlyOne("wait_gr") is not None \
            and exactlyOne("sync") is not None \
            and grByTensor["A"] is not None \
            and grByTensor["B"] is not None \
            and len(lrByTensor["A"]) == 1 \
            and (
                (len(lrByTensor["SA"]) == 4
                 and len(lrByTensor["SB"]) == 4)
                or (not lrByTensor["SA"] and not lrByTensor["SB"])
            )
    elif partition == 1:
        signatureOk = signatureOk and exactlyOne("wait_gr") is not None \
            and exactlyOne("sync") is not None \
            and grByTensor["A"] is not None \
            and grByTensor["B"] is not None \
            and grIncByTensor["A"] is not None \
            and len(lrByTensor["B"]) == 1
    elif partition == 2:
        signatureOk = signatureOk \
            and grByTensor["A"] is not None \
            and grByTensor["B"] is not None \
            and grIncByTensor["B"] is not None \
            and not modules("lr") \
            and not modules("wait_gr") \
            and not modules("sync")
    else:
        signatureOk = signatureOk and exactlyOne("wait_gr") is not None \
            and exactlyOne("sync") is not None \
            and grByTensor["B"] is not None \
            and grByTensor["SA"] is not None \
            and grByTensor["SB"] is not None \
            and grIncByTensor["SA"] is not None \
            and grIncByTensor["SB"] is not None \
            and len(lrByTensor["A"]) == 1 \
            and len(lrByTensor["B"]) == 1 \
            and lrIncByTensor["A"] is not None \
            and lrIncByTensor["B"] is not None
    if not signatureOk:
        return None

    prefix = []
    after = [[] for _ in mfmas]
    consumed = set()

    def putItems(items, destination):
        for inst in items:
            assert id(inst) not in consumed
            consumed.add(id(inst))
            destination.append(inst)

    def putModule(modulePair, destination):
        assert modulePair is not None
        putItems(modulePair[1].instructions, destination)

    def putChunks(items, positions, width):
        chunks = [
            items[index:index + width]
            for index in range(0, len(items), width)
        ]
        assert len(chunks) <= len(positions)
        for chunk, position in zip(chunks, positions):
            putItems(chunk, after[position])

    unrollId = getattr(emittedModules[mfmaIdx], "unrollId", 0)
    waitLrInsts = waitLr[1].instructions

    if partition == 0:
        # Current operands were read in the previous loop tail. Drain them
        # before M0, but delay the DTL wait to M6 and put M7 between it and
        # the barrier, exactly as in the CMS synchronization pattern.
        for inst in waitLrInsts:
            if _isWaitCnt(inst):
                inst.dscnt = 0
                inst.comment = "CMS: wait for prior-loop A/B reads"
        putModule(waitLr, prefix)
        putModule(exactlyOne("wait_gr"), after[6])
        putModule(exactlyOne("sync"), after[7])

        dataReads = _groupLocalReadItems(lrByTensor["A"][0][1])
        if set(dataReads) != {4, 5, 6, 7}:
            return None

        if lrByTensor["SA"]:
            scaleReads = {}
            for tensor in ("SA", "SB"):
                for _, em in lrByTensor[tensor]:
                    scaleReads[(tensor, em.source.tiles.tileId_start)] = \
                        list(em.instructions)
            requiredScales = {
                ("SA", 0), ("SA", 2), ("SA", 4), ("SA", 6),
                ("SB", 0), ("SB", 2), ("SB", 4), ("SB", 6),
            }
            if set(scaleReads) != requiredScales:
                return None
            localReads = [
                *scaleReads[("SA", 4)],
                *scaleReads[("SB", 0)],
                *dataReads[4],
                *dataReads[5],
                *scaleReads[("SA", 6)],
                *dataReads[6],
                *dataReads[7],
                *scaleReads[("SB", 2)],
                # These four scale groups are first consumed in p2.
                *scaleReads[("SA", 0)],
                *scaleReads[("SA", 2)],
                *scaleReads[("SB", 4)],
                *scaleReads[("SB", 6)],
            ]
        else:
            localReads = [
                inst for tile in (4, 5, 6, 7)
                for inst in dataReads[tile]
            ]
        # Five DTL requests occupy the CMS GRA window after M8.
        globalItems = [
            *grByTensor["A"][1].instructions,
            *grByTensor["B"][1].instructions,
        ]
        putChunks(globalItems, list(range(8, 13)), 2)
        # Where the streams overlap, SALU/VMEM precedes LDS issue. This keeps
        # DS/VALU out of the first post-MFMA slot, as required by the retained
        # four-slot safety rule.
        putChunks(localReads, list(range(7, 15)), 2)
        for tensor in ("SA", "SB"):
            if lrIncByTensor[tensor] is not None:
                putModule(lrIncByTensor[tensor], after[14])

    elif partition == 1:
        # Only the last four C0 reads are p2-only scale groups.
        for inst in waitLrInsts:
            if _isWaitCnt(inst):
                inst.dscnt = 4 if unrollId == 0 else 0
                inst.comment = (
                    "CMS: wait for p1 data/scales; leave 4 p2 scales"
                    if unrollId == 0
                    else "CMS: wait for p1 A reads")
        putModule(waitLr, prefix)

        # Finish the old-address B requests and expose A's SRD update early.
        putChunks(grByTensor["B"][1].instructions, [0, 1, 2], 2)
        putChunks(grIncByTensor["A"][1].instructions, [2, 3], 2)

        # CMS places a matrix issue between the p1 wait and barrier.
        putModule(exactlyOne("wait_gr"), after[4])
        putModule(exactlyOne("sync"), after[5])

        bReads = _groupLocalReadItems(lrByTensor["B"][0][1])
        if set(bReads) != {4, 5, 6, 7}:
            return None
        localReads = [
            inst for tile in (4, 5, 6, 7)
            for inst in bReads[tile]
        ]
        putChunks(localReads, [5, 6, 7, 8], 2)

        # Launch the first new-address A request at the reference deadline.
        putModule(grByTensor["A"], after[10])

    elif partition == 2:
        for inst in waitLrInsts:
            if _isWaitCnt(inst):
                inst.dscnt = 0
                inst.comment = "CMS: wait for p2 B reads"
        putModule(waitLr, prefix)

        # Preserve M28-M38 as a compute-only run globally, then cluster the
        # remaining VMEM/address path at M39-M43 and leave M44-M47 clean.
        putChunks(grByTensor["A"][1].instructions, [7, 8, 9], 2)
        putChunks(grIncByTensor["B"][1].instructions, [9, 10], 2)
        putChunks(grByTensor["B"][1].instructions, [10, 11], 2)

    else:
        # Current p3 operands are already resident. The DTL wait can cross M48;
        # the barrier still precedes every next-buffer LDS read.
        for inst in waitLrInsts:
            if _isWaitCnt(inst):
                inst.dscnt = 0
                inst.comment = "CMS: final current-operand LDS drain"
        putModule(waitLr, prefix)
        putModule(exactlyOne("wait_gr"), prefix)
        putModule(exactlyOne("sync"), after[0])

        aReads = _groupLocalReadItems(lrByTensor["A"][0][1])
        bReads = _groupLocalReadItems(lrByTensor["B"][0][1])
        if set(aReads) != {0, 1, 2, 3} \
                or set(bReads) != {0, 1, 2, 3}:
            return None
        localReads = [
            inst
            for tensorReads in (aReads, bReads)
            for tile in (0, 1, 2, 3)
            for inst in tensorReads[tile]
        ]
        putChunks(localReads, list(range(0, 8)), 2)
        putModule(lrIncByTensor["A"], after[3])
        putModule(lrIncByTensor["B"], after[7])

        # B traffic starts at M52. Scale address work and requests follow in
        # dependency order and finish by M62, leaving the final MFMA clean.
        globalItems = [
            *grByTensor["B"][1].instructions,
            *grIncByTensor["SA"][1].instructions,
            *grByTensor["SA"][1].instructions,
            *grIncByTensor["SB"][1].instructions,
            *grByTensor["SB"][1].instructions,
        ]
        putChunks(globalItems, list(range(4, 15)), 2)

    expected = [
        inst
        for mid, em in nonMfma
        for inst in em.instructions
    ]
    if len(consumed) != len(expected) \
            or any(id(inst) not in consumed for inst in expected):
        missing = [
            f"{em.opType}:{type(inst).__name__}"
            for _, em in nonMfma
            for inst in em.instructions
            if id(inst) not in consumed
        ]
        raise AssertionError(
            "integrated CMS profile left instructions unscheduled: "
            + ", ".join(missing))

    scheduled = Module()
    for inst in prefix:
        scheduled.add(inst)
    for index, mfma in enumerate(mfmas):
        scheduled.add(mfma)
        for inst in after[index]:
            scheduled.add(inst)
    _adjustVmcntForScheduledLoads(scheduled)
    return scheduled


def _splitCmsGlobalStream(scheduledGroups):
    """Return one 64-MFMA stream as (mfmas, prefix, post-MFMA buckets)."""
    items = [
        inst
        for scheduled in scheduledGroups
        for inst in scheduled.flatitems()
    ]
    mfmas = []
    prefix = []
    after = []
    current = -1
    for inst in items:
        if isinstance(inst, (MFMAInstruction, MXMFMAInstruction)):
            mfmas.append(inst)
            after.append([])
            current += 1
        elif current < 0:
            prefix.append(inst)
        else:
            after[current].append(inst)
    if len(mfmas) != 64:
        raise AssertionError(
            f"CMS MXFP8 global stream has {len(mfmas)} MFMAs, expected 64")
    return mfmas, prefix, after


def _scheduleCmsMxfp8LocalSync(
        groups, scheduledGroups):
    """Overlay CMS local-read, synchronization, DTL, and loop-tail phasing.

    A/B reads occupy M0-M3/M13-M16 and drain at M6/M20. Data DTL requests
    follow the CMS A-first M8-M25 and B M26-M53 phases. A single M46 VMEM
    drain makes the two early readiness waits redundant and permits barriers
    at M7/M21/M47. Loop-carried reads occupy M48-M55 and drain at M61.
    C0-only scale reads remain split by first-use deadline.
    """
    mfmas, prefix, after = _splitCmsGlobalStream(scheduledGroups)

    def modules(partition, opType, tensor=None):
        return [
            em for em in groups[partition]
            if em.opType == opType
            and (tensor is None
                 or getattr(em.source, "tensor", None) == tensor)
        ]

    def exactlyOne(partition, opType, tensor=None):
        matches = modules(partition, opType, tensor)
        if len(matches) != 1:
            raise AssertionError(
                f"CMS MXFP8 P{partition} expected one {opType}/{tensor}, "
                f"found {len(matches)}")
        return matches[0]

    p0A = exactlyOne(0, "lr", "A")
    p1B = exactlyOne(1, "lr", "B")
    p0WaitLr = exactlyOne(0, "wait_lr")
    p0WaitGr = exactlyOne(0, "wait_gr")
    p0Barrier = exactlyOne(0, "sync")
    p1WaitLr = exactlyOne(1, "wait_lr")
    p1WaitGr = exactlyOne(1, "wait_gr")
    p1Barrier = exactlyOne(1, "sync")
    p2WaitLr = exactlyOne(2, "wait_lr")
    p3WaitLr = exactlyOne(3, "wait_lr")
    p3WaitGr = exactlyOne(3, "wait_gr")
    p3Barrier = exactlyOne(3, "sync")

    dataGr = {
        tensor: [
            em
            for partition in range(4)
            for em in modules(partition, "gr", tensor)
        ]
        for tensor in ("A", "B")
    }
    grIncA = exactlyOne(1, "gr_inc", "A")
    grIncB = exactlyOne(2, "gr_inc", "B")
    scaleGr = {
        tensor: exactlyOne(3, "gr", tensor)
        for tensor in ("SA", "SB")
    }
    scaleGrInc = {
        tensor: exactlyOne(3, "gr_inc", tensor)
        for tensor in ("SA", "SB")
    }

    scaleReads = {}
    for tensor in ("SA", "SB"):
        for em in modules(0, "lr", tensor):
            scaleReads[(tensor, em.source.tiles.tileId_start)] = em
    hasScaleReads = any(em.instructions for em in scaleReads.values())
    if hasScaleReads and set(scaleReads) != {
            ("SA", 0), ("SA", 2), ("SA", 4), ("SA", 6),
            ("SB", 0), ("SB", 2), ("SB", 4), ("SB", 6)}:
        raise AssertionError("CMS MXFP8 scale-read signature changed")

    scaleIncs = [
        em for tensor in ("SA", "SB")
        for em in modules(0, "lr_inc", tensor)
    ]
    dataLrIncs = [
        exactlyOne(3, "lr_inc", tensor)
        for tensor in ("A", "B")
    ]
    relocatedModules = [
        p0A, p1B,
        p0WaitLr,
        p0WaitGr, p0Barrier,
        p1WaitLr, p1WaitGr, p1Barrier,
        p2WaitLr, p3WaitLr, p3WaitGr, p3Barrier,
        grIncA, grIncB,
        *dataGr["A"], *dataGr["B"],
        *scaleGr.values(), *scaleGrInc.values(),
        *scaleReads.values(), *scaleIncs,
    ]
    relocatedIds = {
        id(inst)
        for em in relocatedModules
        for inst in em.instructions
    }
    prefix = [inst for inst in prefix if id(inst) not in relocatedIds]
    after = [
        [inst for inst in bucket if id(inst) not in relocatedIds]
        for bucket in after
    ]

    def appendModule(em, destination):
        destination.extend(em.instructions)

    def setLocalWait(em, comment):
        for inst in em.instructions:
            if _isWaitCnt(inst):
                inst.dscnt = 0
                inst.comment = comment

    aReads = _groupLocalReadItems(p0A)
    bReads = _groupLocalReadItems(p1B)
    if set(aReads) != {4, 5, 6, 7} \
            or set(bReads) != {4, 5, 6, 7}:
        raise AssertionError("CMS MXFP8 data-read signature changed")
    for position, tile in zip(range(0, 4), range(4, 8)):
        after[position].extend(aReads[tile])

    if hasScaleReads:
        # These four scale groups feed p1 (M16) and share its M6 drain.
        appendModule(scaleReads[("SA", 4)], after[4])
        appendModule(scaleReads[("SB", 0)], after[4])
        appendModule(scaleReads[("SA", 6)], after[5])
        appendModule(scaleReads[("SB", 2)], after[5])

    setLocalWait(p1WaitLr, "CMS: drain A and p1 scales before M7")
    appendModule(p1WaitLr, after[6])
    appendModule(p0Barrier, after[7])

    for position, tile in zip(range(13, 17), range(4, 8)):
        after[position].extend(bReads[tile])

    setLocalWait(p2WaitLr, "CMS: drain B reads before M21")
    appendModule(p2WaitLr, after[20])
    appendModule(p1Barrier, after[21])

    # The initial preloop instance is drained once outside LoopBeginL. Every
    # subsequent iteration consumes the M48-M55 wrap reads here, under the
    # final three MFMAs just like the FP8 CMS tail.
    setLocalWait(p0WaitLr, "CMS: drain loop-carried A/B reads at M61")
    appendModule(p0WaitLr, after[61])

    if hasScaleReads:
        # The remaining scales are first consumed by p2 (M32).
        for position, key in zip(
                range(24, 28),
                (("SA", 0), ("SA", 2), ("SB", 4), ("SB", 6))):
            appendModule(scaleReads[key], after[position])
        for em in scaleIncs:
            appendModule(em, after[28])
        setLocalWait(p3WaitLr, "CMS: drain p2 scales before M32")
        appendModule(p3WaitLr, after[31])
        retiredIds = set()
    else:
        # C1 has no scale payload, so the former p3-entry LDS drain is empty
        # work after the A/B M6/M20 drains and is deliberately retired.
        retiredIds = {id(inst) for inst in p3WaitLr.instructions}

    def requestPairs(emitted):
        insts = [
            inst
            for em in emitted
            for inst in em.instructions
        ]
        loads = [
            index for index, inst in enumerate(insts)
            if _isBufferLoad(inst)
        ]
        if len(loads) != 8:
            raise AssertionError(
                f"CMS MXFP8 expected 8 DTL requests, found {len(loads)}")
        pairs = []
        begin = 0
        for loadIndex in loads:
            pairs.append(insts[begin:loadIndex + 1])
            begin = loadIndex + 1
        if begin != len(insts) or any(len(pair) != 2 for pair in pairs):
            raise AssertionError("CMS MXFP8 DTL request is not an m0/load pair")
        return pairs

    aRequests = requestPairs(dataGr["A"])
    bRequests = requestPairs(dataGr["B"])
    for position, request in zip(
            (8, 9, 10, 11, 12, 23, 24, 25), aRequests):
        after[position][0:0] = request
    for position, request in zip(
            (26, 27, 39, 40, 41, 42, 43, 53), bRequests):
        after[position][0:0] = request

    def appendMetadata(inst, text):
        if not hasattr(inst, "comment"):
            return
        old = getattr(inst, "comment", "") or ""
        inst.comment = f"{old}; {text}" if old else text

    def splitWriteSwap(em, tensor, blockedThrough):
        swaps = [
            inst for inst in em.instructions
            if type(inst).__name__ == "SXorB32"
        ]
        if len(swaps) != 1:
            raise AssertionError(
                f"CMS MXFP8 {tensor} expected one LDS write swap")
        pointer = [inst for inst in em.instructions if inst not in swaps]
        for inst in pointer:
            appendMetadata(
                inst,
                f"CMS earliest legal {tensor} SRD update; "
                f"old-address requests end at M{blockedThrough}")
        appendMetadata(swaps[0], f"CMS {tensor} LDS write swap at M57")
        return pointer, swaps

    # The second A/B epochs use updated SRDs. Their pointer updates are already
    # at the earliest legal boundary after the last old-address request. Split
    # the independent LDS write swaps out to the CMS M57 deadline.
    aPointer, aWriteSwap = splitWriteSwap(grIncA, "A", 11)
    bPointer, bWriteSwap = splitWriteSwap(grIncB, "B", 40)
    after[11].extend(aPointer)
    after[40].extend(bPointer)
    after[57][0:0] = [*aWriteSwap, *bWriteSwap]

    def placeVisibleChunks(em, positions, metadata):
        """Split one module by real instructions while retaining TextBlocks."""
        chunks = [[] for _ in positions]
        chunk = 0
        visible = 0
        for inst in em.instructions:
            isText = type(inst).__name__ == "TextBlock"
            if not isText and visible == 2 and chunk + 1 < len(chunks):
                chunk += 1
                visible = 0
            chunks[chunk].append(inst)
            if not isText:
                appendMetadata(inst, metadata)
                visible += 1
        for position, items in zip(positions, chunks):
            # SALU/VMEM should lead the post-MFMA interval when it shares one
            # with LDS traffic.
            after[position][0:0] = items

    # Overlay C0-only scale traffic into C1's otherwise empty/slack windows.
    # Both scale requests stay younger than the M46 data drain. Their address
    # updates finish before their request, before M60 control, and no bucket
    # grows beyond the four-instruction target.
    placeVisibleChunks(
        scaleGrInc["SA"], (44, 45),
        "CMS C0 scale overlay: SA address before M48 request")
    after[48][0:0] = scaleGr["SA"].instructions
    placeVisibleChunks(
        scaleGrInc["SB"], (49, 50),
        "CMS C0 scale overlay: SB address before M52 request")
    after[52][0:0] = scaleGr["SB"].instructions

    for em in scaleIncs:
        tensor = getattr(em.source, "tensor", "scale")
        for inst in em.instructions:
            appendMetadata(
                inst,
                f"CMS nearest legal {tensor} LR swap; "
                "p2 scale reads end at M27")
    for em, blockedThrough in zip(dataLrIncs, (51, 55)):
        tensor = getattr(em.source, "tensor", "data")
        for inst in em.instructions:
            appendMetadata(
                inst,
                f"CMS nearest legal {tensor} LR swap; "
                f"loop-carried reads end at M{blockedThrough}")

    for inst in p3WaitGr.instructions:
        if _isWaitCnt(inst):
            inst.vlcnt = 0
            inst.comment = "CMS: drain loop-carried data DTL requests"
    appendModule(p3WaitGr, after[46])
    appendModule(p3Barrier, after[47])
    retiredIds.update(
        id(inst)
        for em in (p0WaitGr, p1WaitGr)
        for inst in em.instructions
    )

    result = Module("CMS MXFP8 local-read/synchronization schedule")
    for inst in prefix:
        result.add(inst)
    for index, mfma in enumerate(mfmas):
        result.add(mfma)
        for inst in after[index]:
            result.add(inst)

    expectedIds = {
        id(inst)
        for group in groups
        for em in group
        if em.opType != "mfma"
        for inst in em.instructions
    }
    actualIds = [
        id(inst) for inst in result.flatitems()
        if not isinstance(inst, (MFMAInstruction, MXMFMAInstruction))
    ]
    if len(actualIds) != len(set(actualIds)):
        raise AssertionError("CMS MXFP8 global schedule duplicated instructions")
    missing = expectedIds - set(actualIds) - retiredIds
    extra = set(actualIds) - expectedIds
    if missing or extra:
        raise AssertionError(
            "CMS MXFP8 global schedule instruction mismatch: "
            f"missing={len(missing)} extra={len(extra)}")
    return result


def instructionScheduleCmsMxfp8Loop(emitted3d):
    """Schedule the exact CMS-MXFP8 target through one 64-MFMA entry point.

    Returns None for every non-matching loop so generic and exit-loop emission
    continue through the existing per-group scheduler.
    """
    if len(emitted3d) != 4 \
            or any(len(partition) != 1 for partition in emitted3d):
        return None

    groups = [partition[0] for partition in emitted3d]
    scheduledGroups = []
    unrollIds = set()
    for expectedPartition, emittedModules in enumerate(groups):
        mfmaModules = [
            (index, module)
            for index, module in enumerate(emittedModules)
            if module.opType == "mfma"
        ]
        if len(mfmaModules) != 1:
            return None
        mfmaIdx, mfmaModule = mfmaModules[0]
        mfmas = [
            inst for inst in mfmaModule.instructions
            if isinstance(inst, (MFMAInstruction, MXMFMAInstruction))
        ]
        if _cmsMxfp8PartitionIndex(mfmaModule, mfmas) != expectedPartition:
            return None
        unrollIds.add(getattr(mfmaModule, "unrollId", None))
        _, pathOrders, _ = extractPathsFromBeforeDeps(emittedModules)
        scheduled = _scheduleCmsMxfp8Partition(
            emittedModules, mfmaIdx, pathOrders, mfmas)
        if scheduled is None:
            return None
        scheduledGroups.append(scheduled)

    if len(unrollIds) != 1:
        return None

    return _scheduleCmsMxfp8LocalSync(groups, scheduledGroups)


def extractPathsFromBeforeDeps(emittedModules) -> Tuple[int, List[List[int]], List[List[int]]]:
    """Extract non-MFMA dependency paths using only EmittedModule.before links.

    Returns:
      (mfmaIdx, paths, preMfmaPaths)
      - mfmaIdx: index of the MFMA emitted module in emittedModules
      - paths: list of non-MFMA module-index paths to interleave between MFMAs
      - preMfmaPaths: paths that must be emitted before the first MFMA
        (reachable from the MFMA's before link)
    """
    idToIdx = {em.moduleId: i for i, em in enumerate(emittedModules)}
    n = len(emittedModules)

    mfmaModuleIds = [i for i, em in enumerate(emittedModules) if em.opType == "mfma"]
    assert len(mfmaModuleIds) == 1, "extractPathsFromBeforeDeps expects exactly one MFMA emitted module"
    mfmaIdx = mfmaModuleIds[0]
    nonMfmaIds = [i for i in range(n) if i != mfmaIdx]
    nonMfmaSet = set(nonMfmaIds)

    # Identify the non-MFMA module the MFMA depends on (if any).
    mfmaBefore = emittedModules[mfmaIdx].before
    preMfmaTarget = None
    if mfmaBefore is not None:
        bi = idToIdx.get(mfmaBefore)
        if bi is not None and bi in nonMfmaSet:
            preMfmaTarget = bi

    # Each non-MFMA module has at most one predecessor (its `before` link), but a
    # predecessor may be referenced by more than one child (e.g. single-partition
    # PGR2+PLR0, where the MFMA's wait_lr and the GR-prefetch LDS-swap chain both
    # depend on the same local read). The downstream slot placer requires simple
    # linear chains, so at a branch point only ONE child may continue the chain;
    # the others must start their own paths.
    #
    # Crucially, keep the branch that leads to the MFMA's dependency
    # (preMfmaTarget) attached, so that branch — and only that branch — is
    # classified pre-MFMA. The other branches (notably the GR-prefetch
    # buffer_loads, which the MFMA does NOT depend on) are detached into their own
    # regular paths so `spreadBufferLoads` can interleave them across the MFMA
    # stream. The real wait/barrier instructions inside each branch still enforce
    # the LDS read-before-write ordering at runtime, so detaching only drops a
    # scheduling hint, not a correctness guarantee.
    raw_pred: List[int] = [-1 for _ in range(n)]
    children_of: List[List[int]] = [[] for _ in range(n)]
    for i in nonMfmaIds:
        b = emittedModules[i].before
        if b is not None:
            bi = idToIdx.get(b)
            if bi is not None and bi != i and bi in nonMfmaSet:
                raw_pred[i] = bi
                children_of[bi].append(i)

    # For each node on preMfmaTarget's ancestor chain, record which child
    # continues toward the target so we keep that branch attached.
    preferredChild = {}
    if preMfmaTarget is not None:
        c = preMfmaTarget
        guard = 0
        while raw_pred[c] != -1 and guard < n:
            preferredChild[raw_pred[c]] = c
            c = raw_pred[c]
            guard += 1

    pred: List[int] = [-1 for _ in range(n)]
    child: List[int] = [-1 for _ in range(n)]
    for p in nonMfmaIds:
        kids = children_of[p]
        if not kids:
            continue
        primary = preferredChild.get(p, kids[0])
        if primary not in kids:
            primary = kids[0]
        # Only the primary child continues p's chain; others become path heads
        # (their pred stays -1), detaching those subtrees into separate paths.
        child[p] = primary
        pred[primary] = p

    def _findHead(mid: int) -> int:
        cur = mid
        seen = [False for _ in range(n)]
        while pred[cur] != -1 and not seen[cur]:
            seen[cur] = True
            cur = pred[cur]
        return cur

    def _walkFromHead(head: int, used: List[bool]) -> List[int]:
        order: List[int] = []
        localSeen = [False for _ in range(n)]
        cur = head
        while cur != -1 and not used[cur] and not localSeen[cur]:
            order.append(cur)
            localSeen[cur] = True
            cur = child[cur]
        return order

    used = [False for _ in range(n)]
    paths: List[List[int]] = []
    for mid in nonMfmaIds:
        if used[mid]:
            continue
        head = _findHead(mid)
        order = _walkFromHead(head, used)
        assert order, f"extractPathsFromBeforeDeps produced empty path for module {mid}"
        for i in order:
            used[i] = True
        paths.append(order)

    # Separate paths that the MFMA depends on (must go before first MFMA).
    preMfmaPaths: List[List[int]] = []
    regularPaths: List[List[int]] = []
    for path in paths:
        if preMfmaTarget is not None and preMfmaTarget in path:
            preMfmaPaths.append(path)
        else:
            regularPaths.append(path)

    return mfmaIdx, regularPaths, preMfmaPaths


def instructionSchedule(emittedModules, multiDU: bool = False,
                      minGapDsReadToWait=_MIN_MFMA_GAP_DS_READ_TO_WAIT_DEFAULT):
    """Interleave non-MFMA instructions between MFMAs.

    Rules (shared):
      - MFMA order is preserved.
      - The legacy schedule uses 2 placement slots between adjacent MFMAs.
      - The selected MXFP8 p0 experiment uses 4 4-cycle scheduling slots.
      - At most one ds_read is placed per two-slot SIMD issue pair.
      - Before dependencies are respected at module order level.
      - Minimm distance between ds_read and it waitcnt (hardcoded for now)
      - Module-internal instruction order is preserved.
      - GR path is spread as much as possible across remaining valid slots.

    When multiDU is False (single-DU / legacy path):
      - LR path containing a WAIT_GR is packed from the end backwards.

    When multiDU is True (MX multi-DU path):
      - LR path containing a WAIT_GR is placed forward (wait_gr → sync → LR insts).
      - wait_gr+barrier stay contiguous; GR m0/buffer_load defer until that path's
        post-barrier ds_reads complete.

      TODO : To be tested on multi-partition setup.
    """
    if not emittedModules:
        return Module()

    isMFMA = lambda x: isinstance(x, (MFMAInstruction, MXMFMAInstruction))
    n = len(emittedModules)

    mfmaIdx, pathOrders, preMfmaOrders = extractPathsFromBeforeDeps(emittedModules)
    mfmas = [x for x in emittedModules[mfmaIdx].instructions if isMFMA(x)]
    cmsMxfp8 = _scheduleCmsMxfp8Partition(
        emittedModules, mfmaIdx, pathOrders, mfmas)
    if cmsMxfp8 is not None:
        return cmsMxfp8

    p0Split = None
    if mfmas and all(
            isinstance(inst, MXMFMAInstruction)
            and inst.getIssueLatency() == 8 for inst in mfmas):
        p0Split = _splitPartitionZeroPrefetch(
            preMfmaOrders, emittedModules)
    if p0Split is None:
        _setPartitionZeroPrefetchWait(preMfmaOrders, emittedModules)

    slotsPerInterval = 4 if p0Split is not None else 2
    placementPathOrders = list(pathOrders)
    if p0Split is not None:
        placementPathOrders.append(p0Split[2])

    def _emitPreMfma(result):
        for order in preMfmaOrders:
            if p0Split is not None and order == p0Split[0]:
                for _, inst in p0Split[1]:
                    result.add(inst)
                continue
            for mid in order:
                for inst in emittedModules[mid].instructions:
                    result.add(inst)

    # Single MFMA: no slots to interleave into — emit preMfma, MFMA, then paths.
    if len(mfmas) < 2:
        result = Module()
        _emitPreMfma(result)
        for m in mfmas:
            result.add(m)
        if p0Split is not None:
            for _, inst in p0Split[3]:
                result.add(inst)
        for order in pathOrders:
            for mid in order:
                for inst in emittedModules[mid].instructions:
                    result.add(inst)
        return result

    paths = _classifyPaths(pathOrders, emittedModules)
    rules = _SchedulingRules(
        totalSlots=(len(mfmas) - 1) * slotsPerInterval,
        minGapDsReadToWait=minGapDsReadToWait)
    base_validators = [rules.oneDsReadPerInterval, rules.minGapDsReadBeforeWait,
                       rules.minGapDsReadToWait, rules.noM0WithBufferLoad]
    base_adjusters = [rules.spreadLocalReads, rules.spreadBufferLoads]
    if p0Split is not None:
        base_validators.append(rules.noDsReadOrValuInFirstSlot)
    if multiDU:
        base_validators.append(rules.grAfterWaitGrDsReads)
        base_adjusters.insert(0, rules.deferGrAfterWaitGrDsReads)
    placer = _SlotPlacer(
        len(mfmas) - 1, n, placementPathOrders,
        validators=base_validators,
        adjusters=base_adjusters,
        onPlace=rules.trackPlacement,
        slotsPerInterval=slotsPerInterval)

    def _placeRegularPaths():
        for order, hasWaitGR in paths:
            if not order:
                continue
            wrapPrefetch = (not multiDU) and hasWaitGR \
                and _isWrapPrefetchPath(order, emittedModules)
            if wrapPrefetch:
                # The final partition wraps its local-read prefetch to partition 0
                # of the next MT. Legacy reverse packing runs out of valid slots and
                # force-stacks wait+barrier, LR swaps, and several ds_reads before
                # MFMA #1. Keep only the synchronization in the first interval;
                # current-partition MFMAs do not consume the wrapped LR payload.
                pathInsts = _flattenPath(order, emittedModules)
                syncEnd = next(
                    (idx + 1 for idx, (_, inst) in enumerate(pathInsts)
                     if _isBarrier(inst)),
                    0)
                if syncEnd:
                    rules.resetPath()
                    placer.placePath(
                        pathInsts[:syncEnd],
                        multiDU=True)
                    placer.placePath(
                        pathInsts[syncEnd:],
                        startLimit=2,
                        overflowToTail=True)
                    continue

            reverse = (not multiDU) and hasWaitGR
            pathInsts = _flattenPath(order, emittedModules, reverse=reverse)
            rules.resetPath()
            if not hasWaitGR:
                rules.setupBufLoadSpreading(placer, pathInsts, order)
            placer.placePath(pathInsts, reverse=reverse, multiDU=multiDU)
            if multiDU and hasWaitGR:
                rules.setPostWaitGrDsReadFence(placer)

    if p0Split is not None and all(not hasWaitGR for _, hasWaitGR in paths):
        regularHalves = [
            _splitItemsAfterHalf(
                _flattenPath(order, emittedModules), _isBufferLoad)
            for order, _ in paths
        ]
        p0Halves = _splitItemsAfterHalf(p0Split[3], _isDsRead)

        def _placePacked(pathInsts):
            if not pathInsts:
                return
            rules.resetPath()
            placer.placePath(
                pathInsts,
                startLimit=_nextPlacementSlot(placer))

        # Split after complete m0/buffer-load pairs: first half of each
        # SALU/global stream, first eight DS reads, then the remaining halves.
        for first, _ in regularHalves:
            _placePacked(first)
        _placePacked(p0Halves[0])
        for _, second in regularHalves:
            _placePacked(second)
        _placePacked(p0Halves[1])
    else:
        _placeRegularPaths()
        if p0Split is not None:
            rules.resetPath()
            placer.placePath(p0Split[3])

    scheduled = Module()
    _emitPreMfma(scheduled)
    scheduled.add(placer.assemble(mfmas))

    # Post-pass: adjust waits for buffer_loads moved before them.
    _adjustVmcntForScheduledLoads(scheduled)

    return scheduled
