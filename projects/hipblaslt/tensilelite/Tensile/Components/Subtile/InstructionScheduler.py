# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Instruction scheduler for subtile-based mainloop.

Interleaves non-MFMA instructions between MFMAs using a slot-based placer
with pluggable scheduling rules.
"""

from typing import List, Tuple, Optional
from rocisa.code import Module, TextBlock
from rocisa.instruction import SWaitCnt, SBarrier, MFMAInstruction, MXMFMAInstruction, \
    LocalReadInstruction, GlobalReadInstruction, CommonInstruction


class _SlotPlacer:
    """Generic slot placement engine for interleaving instructions between MFMAs.

    Each interval (pair of adjacent MFMAs) has 2 placement slots.
    Rules are injected via callbacks:
      - validators: (placer, pos, inst) -> bool — reject invalid slots
      - adjusters:  (placer, limit, inst) -> limit — shift search start
      - onPlace:    (placer, pos, inst) -> None — update rule state after placement
    """

    def __init__(self, intervals: int, numModules: int,
                 pathOrders: List[List[int]],
                 validators=None, adjusters=None, onPlace=None,
                 maxInstructionsPerInterval: Optional[int] = None):
        self.totalSlots = intervals * 2
        self._n = numModules
        self.maxInstructionsPerInterval = maxInstructionsPerInterval
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
        if pos < 0 or pos >= self.totalSlots:
            return False
        slotCount = sum(
            not _isZeroWidth(item[1]) for item in self._placed[pos])
        if not _isZeroWidth(inst) and slotCount >= 2:
            return False
        if self.maxInstructionsPerInterval is not None:
            interval = pos // 2
            count = sum(
                not _isZeroWidth(item[1])
                for p in (2 * interval, 2 * interval + 1)
                if p < self.totalSlots
                for item in self._placed[p])
            if (not _isZeroWidth(inst)
                    and count >= self.maxInstructionsPerInterval):
                return False
        return all(v(self, pos, inst) for v in self._validators)

    def _canPlaceBundle(self, pos: int, items: List[Tuple[int, object]]) -> bool:
        """Return whether an ordered instruction bundle fits one physical slot."""
        if pos < 0 or pos >= self.totalSlots:
            return False
        slotCount = sum(
            not _isZeroWidth(item[1]) for item in self._placed[pos])
        bundleCount = sum(not _isZeroWidth(item[1]) for item in items)
        if slotCount + bundleCount > 2:
            return False
        if self.maxInstructionsPerInterval is not None:
            interval = pos // 2
            count = sum(
                not _isZeroWidth(item[1])
                for p in (2 * interval, 2 * interval + 1)
                if p < self.totalSlots
                for item in self._placed[p])
            if count + bundleCount > self.maxInstructionsPerInterval:
                return False
        return all(
            all(v(self, pos, inst) for v in self._validators)
            for _, inst in items)

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

    def findBundleSlot(self, mid: int, items: List[Tuple[int, object]],
                       limit: int) -> Optional[int]:
        """Find one slot for an ordered forward-only instruction bundle."""
        lo, hi = self.bounds(mid)
        lo = max(lo, limit)
        if hi < lo:
            return None
        firstInterval = lo // 2
        lastInterval = hi // 2
        for interval in range(firstInterval, lastInterval + 1):
            for pos in (2 * interval, 2 * interval + 1):
                if pos < lo or pos > hi:
                    continue
                if self._canPlaceBundle(pos, items):
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
                  multiDU: bool = False,
                  bundleM0BufferLoad: bool = False):
        """Place a sequence of (moduleId, instruction) items into slots.

        Walks pathInsts in order, applying adjusters (forward only) and
        finding valid slots. When no empty slot is found, force-places at
        the closest valid position respecting dependencies (allowing >2
        items per slot).

        When multiDU is True, wait_gr immediately followed by barrier is
        placed as an atomic pair in the same slot so other paths cannot
        insert m0 setup between them.
        """
        limit = (self.totalSlots - 1) if reverse else 0
        idx = 0
        while idx < len(pathInsts):
            item = pathInsts[idx]
            mid, inst = item
            if _isZeroWidth(inst):
                # Keep comments/directives in the emitted assembly without
                # charging an issue slot or advancing the placement cursor.
                prior = self._firstPos[mid] if reverse else self._lastPos[mid]
                pos = prior if prior is not None else max(
                    0, min(limit, self.totalSlots - 1))
                self.place(pos, item, reverse=reverse)
                idx += 1
                continue
            if (bundleM0BufferLoad and not reverse
                    and idx + 1 < len(pathInsts)
                    and pathInsts[idx + 1][0] == mid
                    and _isM0Update(inst)
                    and _isBufferLoad(pathInsts[idx + 1][1])):
                bundle = pathInsts[idx:idx + 2]
                limit = self.adjustLimit(limit, bundle[0][1])
                limit = self.adjustLimit(limit, bundle[1][1])
                pos = self.findBundleSlot(mid, bundle, limit)
                if pos is None:
                    # Capacity fallback: retain ordering and hard lower bounds
                    # without overfilling one physical slot.
                    for bundleItem in bundle:
                        bundleMid, bundleInst = bundleItem
                        pos = self.findSlot(
                            bundleMid, bundleInst, limit, reverse=False)
                        if pos is None:
                            pos = self._forceSlot(
                                bundleMid, limit, reverse=False)
                        self.place(pos, bundleItem, reverse=False)
                        limit = pos + 1
                else:
                    self.place(pos, bundle[0], reverse=False)
                    self.place(pos, bundle[1], reverse=False)
                    limit = pos + 1
                idx += 2
                continue
            if (multiDU and not reverse and idx + 1 < len(pathInsts)
                    and _isWaitGr(inst) and _isBarrier(pathInsts[idx + 1][1])):
                if not reverse:
                    limit = self.adjustLimit(limit, inst)
                pos = self.findSlot(mid, inst, limit, reverse=reverse)
                if pos is None:
                    pos = self._forceSlot(mid, limit, reverse)
                self.place(pos, item, reverse=reverse)
                sync_item = pathInsts[idx + 1]
                if self._canPlace(pos, sync_item[1]):
                    self.place(pos, sync_item, reverse=reverse)
                else:
                    sync_pos = self.findSlot(sync_item[0], sync_item[1],
                                             pos + 1, reverse=reverse)
                    if sync_pos is None:
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
                pos = self._forceSlot(mid, limit, reverse)
            self.place(pos, item, reverse=reverse)
            limit = (pos - 1) if reverse else (pos + 1)
            idx += 1

    # ── Assembly ──

    def assemble(self, mfmas) -> Module:
        result = Module()
        for i, mfma in enumerate(mfmas):
            result.add(mfma)
            for slot in (2 * i, 2 * i + 1):
                if slot >= self.totalSlots:
                    continue
                for item in self._placed[slot]:
                    result.add(item[1])
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
_isZeroWidth = lambda x: isinstance(x, TextBlock)
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
        self.remainingDsReads = 0
        self.allowSecondDsRead = False

    # ── Validators: (placer, pos, inst) -> bool ──

    def oneDsReadPerInterval(self, placer, pos, inst):
        """Use one ds_read per interval, permitting a second only when needed."""
        if not _isDsRead(inst):
            return True
        interval = pos // 2
        count = sum(
            _isDsRead(item[1])
            for p in (2 * interval, 2 * interval + 1)
            if p < placer.totalSlots
            for item in placer._placed[p])
        if count == 0:
            return True
        return self.allowSecondDsRead and count < 2

    def minGapDsReadBeforeWait(self, placer, pos, inst):
        """Reject ds_read too close to an already-placed waitcnt ahead."""
        if not _isDsRead(inst):
            return True
        # A wait already behind this read guards older traffic, not this read.
        # Forward producer-frontier paths intentionally have
        # wait_gr -> barrier -> ds_read, so only enforce the gap when the
        # already-placed wait is actually ahead of the candidate read.
        if pos >= self.earliestWaitCntPos:
            return True
        gap = self.minGap * 2
        return self.earliestWaitCntPos - pos >= gap

    def minGapDsReadToWait(self, placer, pos, inst):
        """Reject waitcnt too close to the last placed ds_read."""
        if not _isWaitCnt(inst) or self.lastDsReadPos < 0:
            return True
        gap = self.minGap * 2
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

    def grAfterWaitGrBarrier(self, placer, pos, inst):
        """Keep aliasing future GRs strictly after the producer frontier.

        Unlike the multi-DU fence, this allows GRs to interleave with the
        post-barrier local reads.  The barrier is the correctness boundary;
        finishing every local read first is unnecessarily restrictive.
        """
        if not (_isBufferLoad(inst) or _isM0Update(inst)):
            return True
        if self.waitGrBarrierPos is None:
            return True
        return pos > self.waitGrBarrierPos

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

    def deferGrAfterWaitGrBarrier(self, placer, limit, inst):
        """Start future GR placement immediately after the frontier barrier."""
        if not (_isBufferLoad(inst) or _isM0Update(inst)):
            return limit
        if self.waitGrBarrierPos is None:
            return limit
        return max(limit, self.waitGrBarrierPos + 1)

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

    def allowNecessarySecondDsReads(self, placer, limit, inst):
        """Permit two DS reads only when one-per-remaining-interval cannot fit."""
        if not _isDsRead(inst):
            return limit
        zeroDsIntervals = 0
        for interval in range(limit // 2, (placer.totalSlots + 1) // 2):
            positions = [
                p for p in (2 * interval, 2 * interval + 1)
                if limit <= p < placer.totalSlots
            ]
            if not positions:
                continue
            instructions = [
                item[1]
                for p in (2 * interval, 2 * interval + 1)
                if p < placer.totalSlots
                for item in placer._placed[p]
            ]
            if any(_isDsRead(item) for item in instructions):
                continue
            if (placer.maxInstructionsPerInterval is not None
                    and len(instructions) >= placer.maxInstructionsPerInterval):
                continue
            zeroDsIntervals += 1
        self.allowSecondDsRead = self.remainingDsReads > zeroDsIntervals
        return limit

    # ── Placement hook: (placer, pos, inst) -> None ──

    def trackPlacement(self, placer, pos, inst):
        """Update rule state after a successful placement."""
        if _isDsRead(inst):
            self.lastDsReadPos = max(self.lastDsReadPos, pos)
            self.remainingDsReads = max(0, self.remainingDsReads - 1)
        if _isWaitCnt(inst):
            self.earliestWaitCntPos = min(self.earliestWaitCntPos, pos)
        if _isBufferLoad(inst) and self.firstBufLoadPos is None:
            self.firstBufLoadPos = pos

    # ── Per-path setup ──

    def resetPath(self):
        self._resetPath()

    def setupPath(self, pathInsts):
        self.remainingDsReads = sum(
            1 for _, inst in pathInsts if _isDsRead(inst))

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

    def setWaitGrBarrierFence(self, placer) -> None:
        """Record the barrier that closes the current wait_gr frontier."""
        wg_pos = None
        barrier_pos = None
        for pos, slot in enumerate(placer._placed):
            for _, inst in slot:
                if _isWaitGr(inst) and wg_pos is None:
                    wg_pos = pos
                elif (_isBarrier(inst) and wg_pos is not None
                      and pos >= wg_pos and barrier_pos is None):
                    barrier_pos = pos
        self.waitGrBarrierPos = barrier_pos
        self.maxDsReadAfterWaitGr = -1


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
    pathInsts = []
    for mid in order:
        for inst in emittedModules[mid].instructions:
            # Some emitters return a nested Module (notably scale SRD/base
            # updates). Slot capacity must account for the physical
            # instructions, not the Python wrapper as one synthetic item.
            physical = list(inst.flatitems()) if isinstance(inst, Module) else [inst]
            pathInsts.extend((mid, item) for item in physical)
    if reverse:
        pathInsts.reverse()
    return pathInsts


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
                      stagedProducerFrontiers: bool = False,
                      minGapDsReadToWait=_MIN_MFMA_GAP_DS_READ_TO_WAIT_DEFAULT):
    """Interleave non-MFMA instructions between MFMAs using 2 slots/interval.

    Rules (shared):
      - MFMA order is preserved.
      - Between two adjacent MFMAs there are 2 placement slots.
      - At most one ds_read (LocalReadInstruction) per interval.
      - Before dependencies are respected at module order level.
      - Minimm distance between ds_read and it waitcnt (hardcoded for now)
      - Module-internal instruction order is preserved.
      - Paths containing a lifetime-delayed LR are emitted after the final
        MFMA instead of consuming an inter-MFMA slot.
      - GR path is spread as much as possible across remaining valid slots.

    When multiDU is False (single-DU / legacy path):
      - LR path containing a WAIT_GR is packed from the end backwards.

    When multiDU is True (MX multi-DU path):
      - LR path containing a WAIT_GR is placed forward (wait_gr → sync → LR insts).
      - wait_gr+barrier stay contiguous; GR m0/buffer_load defer until that path's
        post-barrier ds_reads complete.

    When stagedProducerFrontiers is True:
      - WAIT_GR paths are also placed forward and keep wait_gr+barrier atomic.
      - Future GRs are hard-bounded after the barrier, but may interleave with
        post-barrier local reads.

      TODO : To be tested on multi-partition setup.
    """
    if not emittedModules:
        return Module()

    isMFMA = lambda x: isinstance(x, (MFMAInstruction, MXMFMAInstruction))
    n = len(emittedModules)

    mfmaIdx, pathOrders, preMfmaOrders = extractPathsFromBeforeDeps(emittedModules)
    mfmas = [x for x in emittedModules[mfmaIdx].instructions if isMFMA(x)]

    postMfmaOrders = []
    regularOrders = []
    for order in pathOrders:
        if any(getattr(emittedModules[mid].source,
                       "scheduleAfterMfma", False)
               for mid in order):
            postMfmaOrders.append(order)
        else:
            regularOrders.append(order)
    pathOrders = regularOrders

    def _emitPreMfma(result):
        for order in preMfmaOrders:
            for mid in order:
                for inst in emittedModules[mid].instructions:
                    result.add(inst)

    def _emitPostMfma(result):
        for order in postMfmaOrders:
            for mid in order:
                for inst in emittedModules[mid].instructions:
                    result.add(inst)

    # Single MFMA: no slots to interleave into — emit preMfma, MFMA, then paths.
    if len(mfmas) < 2:
        result = Module()
        _emitPreMfma(result)
        for m in mfmas:
            result.add(m)
        for order in pathOrders:
            for mid in order:
                for inst in emittedModules[mid].instructions:
                    result.add(inst)
        _emitPostMfma(result)
        return result

    paths = _classifyPaths(pathOrders, emittedModules)
    # Staged frontiers use the interval after the partition's final MFMA too.
    # This is still inside the partition boundary and is needed to distribute
    # a full wrap-read cohort over all of its matrix issue windows.
    numIntervals = len(mfmas) if stagedProducerFrontiers else len(mfmas) - 1
    rules = _SchedulingRules(totalSlots=numIntervals * 2,
                             minGapDsReadToWait=minGapDsReadToWait)
    base_validators = [rules.oneDsReadPerInterval, rules.minGapDsReadBeforeWait,
                       rules.minGapDsReadToWait]
    if not stagedProducerFrontiers:
        base_validators.append(rules.noM0WithBufferLoad)
    base_adjusters = [rules.spreadBufferLoads]
    if multiDU:
        base_validators.append(rules.grAfterWaitGrDsReads)
        base_adjusters.insert(0, rules.deferGrAfterWaitGrDsReads)
    elif stagedProducerFrontiers:
        base_validators.append(rules.grAfterWaitGrBarrier)
        base_adjusters.insert(0, rules.deferGrAfterWaitGrBarrier)
        base_adjusters.insert(1, rules.allowNecessarySecondDsReads)
    placer = _SlotPlacer(
        numIntervals, n, pathOrders,
        validators=base_validators,
        adjusters=base_adjusters,
        onPlace=rules.trackPlacement,
        maxInstructionsPerInterval=(3 if stagedProducerFrontiers else None))

    for order, hasWaitGR in paths:
        if not order:
            continue
        forwardWaitPath = multiDU or stagedProducerFrontiers
        reverse = (not forwardWaitPath) and hasWaitGR
        pathInsts = _flattenPath(order, emittedModules, reverse=reverse)
        rules.resetPath()
        rules.setupPath(pathInsts)
        if not hasWaitGR:
            rules.setupBufLoadSpreading(placer, pathInsts, order)
        placer.placePath(
            pathInsts,
            reverse=reverse,
            multiDU=forwardWaitPath,
            bundleM0BufferLoad=stagedProducerFrontiers)
        if multiDU and hasWaitGR:
            rules.setPostWaitGrDsReadFence(placer)
        elif stagedProducerFrontiers and hasWaitGR:
            rules.setWaitGrBarrierFence(placer)

    scheduled = Module()
    _emitPreMfma(scheduled)
    scheduled.add(placer.assemble(mfmas))
    _emitPostMfma(scheduled)

    # Post-pass: adjust vmcnt of any SWaitCnt to account for buffer_loads
    # that the scheduler placed before it within this subIterK.
    bufLoadCount = 0
    for inst in scheduled.flatitems():
        if _isBufferLoad(inst):
            bufLoadCount += 1
        elif _isWaitCnt(inst) and inst.vlcnt >= 0:
            if getattr(inst, 'adjustVmcnt', True):
                inst.vlcnt += bufLoadCount

    return scheduled
