# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Ring placement targets for one half of a dual-period MXFP8 half-disjoint body.

TENSILE_MXFP8_RING=1. The two bodies of the K loop form one ring of MFMA gaps with
four halves, each opened by today's barrier. The load plan has already given every
direct-to-LDS load and every ds_read its half; this module picks the gap of each:

- Loads go on a uniform grid over the half's load gaps, one per gap, so their spacing
  is the half's gap count over its load count.
- Every ds_read has a window: from its floor (the half's barrier, the last MFMA that
  reads its destination's old contents, the scale-source retain distance) to its
  deadline (RING_READ_LEAD_MFMAS ahead of its first consumer, and ahead of the fence
  that drains its cell). Reads take a uniform grid of their own, kept off the load
  gaps, at most RING_READS_PER_GAP to a gap, earliest deadline first.
- One ds_read always goes in the barrier gap.

Reads of one chain keep their order. A read that fits no gap in its window raises
RingInfeasible, naming it, instead of being forced.
"""

import heapq
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set

RING_READ_LEAD_MFMAS = 6
RING_READS_PER_GAP = 2


class RingInfeasible(Exception):
    """A load or read has no gap left in its window."""


@dataclass
class RingRead:
    floor: int
    deadline: int
    chain: int
    name: str = ""
    target: Optional[int] = None
    onGrid: bool = False


@dataclass
class RingHalf:
    """One half's gaps. Gap g follows the half's MFMA g.

    barrierGap opens the half. loadGaps lists, in order, the gaps a load may take.
    readBlocked holds gaps no ds_read may take, and reserved the ds_reads a gap
    already holds (scale refreshes placed by the shadow plan).
    """
    gaps: int
    barrierGap: int
    loadGaps: List[int]
    loadCount: int
    reads: List[RingRead]
    readBlocked: Set[int] = field(default_factory=set)
    reserved: Dict[int, int] = field(default_factory=dict)
    barrierRead: bool = True


def ringLoadTargets(loadGaps: Sequence[int], count: int) -> List[int]:
    """count gaps of loadGaps, one at the middle of each of count equal shares."""
    if count <= 0:
        return []
    if len(loadGaps) < count:
        raise RingInfeasible(f"{count} loads over {len(loadGaps)} load gaps")
    n = len(loadGaps)
    return [loadGaps[(2 * k + 1) * n // (2 * count)] for k in range(count)]


def _chainOrdered(reads: List[RingRead]):
    """Floors rise and deadlines fall back along each chain, so EDF keeps chain order."""
    lastFloor = {}
    for read in reads:
        read.floor = max(read.floor, lastFloor.get(read.chain, read.floor))
        lastFloor[read.chain] = read.floor
    nextDeadline = {}
    for read in reversed(reads):
        read.deadline = min(read.deadline, nextDeadline.get(read.chain, read.deadline))
        nextDeadline[read.chain] = read.deadline


def _readGrid(half: RingHalf, loadSet: Set[int], last: int) -> Dict[int, int]:
    """Uniform read targets from the barrier gap to last, moved off load gaps."""
    grid: Dict[int, int] = {}
    n = len(half.reads)
    if not n:
        return grid
    first = half.barrierGap
    span = max(1, last - first + 1)

    def room(g):
        return (first <= g <= last and g not in loadSet and g not in half.readBlocked
                and grid.get(g, 0) + half.reserved.get(g, 0) < RING_READS_PER_GAP)

    for k in range(n):
        target = first + k * span // n
        for d in range(span + 1):
            spot = next((g for g in (target - d, target + d) if room(g)), None)
            if spot is not None:
                grid[spot] = grid.get(spot, 0) + 1
                break
    return grid


def ringReadTargets(half: RingHalf, loadTargets: Sequence[int]) -> List[RingRead]:
    """Give every read of half a target gap (RingRead.target), or raise RingInfeasible."""
    reads = half.reads
    _chainOrdered(reads)
    for read in reads:
        if read.floor > read.deadline:
            raise RingInfeasible(
                f"{read.name or 'ds_read'}: empty window [{read.floor}, {read.deadline}]")
    loadSet = set(loadTargets)
    last = max((read.deadline for read in reads), default=half.barrierGap)
    grid = _readGrid(half, loadSet, last)
    count = dict(half.reserved)
    used: Dict[int, int] = {}

    def take(g, read, onGrid):
        read.target = g
        read.onGrid = onGrid
        count[g] = count.get(g, 0) + 1
        if onGrid:
            used[g] = used.get(g, 0) + 1

    def free(g):
        return 0 if g in half.readBlocked else RING_READS_PER_GAP - count.get(g, 0)

    if half.barrierRead and reads:
        heads = {}
        for read in reads:
            heads.setdefault(read.chain, read)
        candidates = [r for r in heads.values()
                      if r.floor <= half.barrierGap <= r.deadline]
        if not candidates or half.barrierGap in half.readBlocked:
            raise RingInfeasible("no ds_read window covers the barrier gap")
        take(half.barrierGap, min(candidates, key=lambda r: (r.deadline, reads.index(r))),
             True)

    passes = (
        lambda g: min(grid.get(g, 0) - used.get(g, 0), free(g)),
        lambda g: 0 if g in loadSet else free(g),
        free,
    )
    pending = [read for read in reads if read.target is None]
    for capacity in passes:
        pending = _edf(pending, capacity, half.gaps,
                       lambda g, read, onGrid=capacity is passes[0]: take(g, read, onGrid))
        if not pending:
            break
    if pending:
        raise RingInfeasible(
            f"{pending[0].name or 'ds_read'}: window [{pending[0].floor}, "
            f"{pending[0].deadline}] full")
    return reads


def _edf(reads, capacity, gaps, take):
    """Earliest deadline first along the half; returns the reads left unplaced."""
    order = sorted(range(len(reads)), key=lambda i: reads[i].floor)
    heap, k, unplaced = [], 0, []
    for g in range(gaps):
        while k < len(order) and reads[order[k]].floor <= g:
            i = order[k]
            heapq.heappush(heap, (reads[i].deadline, i))
            k += 1
        while heap and heap[0][0] < g:
            unplaced.append(reads[heapq.heappop(heap)[1]])
        room = capacity(g)
        while heap and room > 0:
            take(g, reads[heapq.heappop(heap)[1]])
            room -= 1
    unplaced += [reads[i] for _, i in heap]
    unplaced += [reads[i] for i in order[k:]]
    return sorted(unplaced, key=reads.index)
