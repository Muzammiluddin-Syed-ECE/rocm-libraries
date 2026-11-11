# Plan: K=0 confinement — force the fold's data into retired K=0 holes

Status: design, not yet implemented. Base commit `728eaec169e` on
`users/muzasyed/coalesce-weave` (off PR #14644). gfx950 F4BS MXF4→bf16 TN subtile store.
Implemented on branch `users/muzasyed/k0-confinement`.

## TL;DR

Keep today's architecture — the DPP fold fires on **every** store — and make the MFMA weave fire
too by **placing the fold's data registers in the retired K=0 operand holes** instead of letting
them land on the live last-K operands the moved terminal MFMAs still read. This removes the clobber
that currently caps the weave at ~16 moves, and it removes the `+12` fresh-register cost that forces
today's overflow/lend — without giving up any folds.

## The problem, as the dry-run (TENSILE_DRYRUN_REGS) proved it

On the tight tile **MT256x256 MIWT8_8**:

- 170 operand registers: **68 retired K=0 holes** (dead before the terminal MFMAs) +
  **102 last-K-live** registers (v56..157) the terminal MFMAs still read.
- The fold's **scratch** (9-reg cvt block, `vgprBf16Temp`+perm/laneGroup/addr/voff/blend) already
  lands entirely in the retired K=0 holes (v36..44). Not the problem.
- The fold's **data** is the clobber: the DPP blend writes `vCoalesceStoreData` (= `sumIdx1`, a
  ValuC slot physically in the last-K range). That is 88 `v_cndmask` writes onto registers the
  moved MFMAs read → `_planCapturedTerminalMfmas`'s store-clobber guard (already in the base)
  correctly rejects those moves → only ~16 of N store-gaps reach weave quota.
- **~55 K=0 holes are free** after the cvt block — more than enough room for the fold's data.

A prior attempt that *dedicated* `vCoalesceStoreData`/`vPack2` via a generic
`vgprPool.checkOutAligned` **failed** because the generic checkout grabbed v100..136 — *inside* the
last-K range — not the free K=0 holes. Right idea (dedicate), wrong placement. This plan fixes the
placement.

## The fix

Confine the fold's **data** registers (the blend's store-data quad, and the batchB pack quad if it
is given its own regs) to the retired K=0 holes, never to live last-K.

The machinery already exists:
- `_operandLendVgprs(dead_ids)` (LogicalScheduler) returns the exact `(base, size)` register ranges
  of the dead K=0 tiles. `_deadOperandTileIds(unroll_iter)` computes `dead_ids`.
- The pool exposes `addFromCheckOut(start)` / `removeFromCheckOut(start)` (register.py) to mark a
  specific range Available / restore it. KWA already drives this for the store lend via
  `self.states.subtileFusedLendVgprs` around `globalWriteElements` (~line 16189–16220).
- The last-K-live tiles are already kept reserved (InUse) across the loop; only the K=0 holes are
  lent. The failure mode is that the fold's *data* checkout can still pull a register that is not
  explicitly reserved. The fix makes the fold-data checkout draw **only** from the K=0-hole set.

## Implementation sketch

In `KernelWriterAssembly.globalWriteElements` (the subtile cvt block, ~19255–19345):

1. **Get the K=0-hole register set available at store time.** It is already materialized as
   `self.states.subtileFusedLendVgprs` (the lend list the policy produced). Partition it: the cvt
   block already consumes the low part; reserve a disjoint sub-range of the *remaining* K=0 holes
   for the fold's data.
2. **Allocate `vCoalesceStoreData` (and `vPack2` if dedicating the batchB pack) from that K=0
   sub-range**, not via a generic `checkOutAligned`. Options, simplest first:
   - Carve the fold-data quads out of the lend list directly (indices from `subtileFusedLendVgprs`),
     bypassing the generic pool checkout; or
   - Temporarily `removeFromCheckOut` / mark-unavailable everything *except* the chosen K=0 sub-range
     so a `checkOutAligned(4,2)` is forced to land there, then restore.
3. **Wire the struct** (`BF16CVTVgprStruct.vgprCoalesceStoreData`, optionally `vgprCoalescePack2`)
   and check them in at store end (the prior bug: a missing `checkIn` tripped
   `RegisterPool::checkFinalState`; keep the check-in).
4. **Emitter** (`_emit16bitSubtilePairedStoreCoalesce`, GlobalWriteBatch.py): use the dedicated
   K=0 `vCoalesceStoreData` on the weave path; keep the ValuC reuse (`sumIdx1`) off the weave path
   (so monolithic/guarded stores are byte-identical). The store-clobber guard stays unchanged.

The 8 fold-data regs are *shared* across all stores (the cvt block is allocated once, reused
sequentially), so confining them once covers every store — this is not per-store.

## Why it should fit the budget

Today's overflow is `base + (last-K operands live) + ~12 fold-data` ≈ 268. Sourcing the fold data
from the already-counted K=0 holes removes the `+12` (it is recycled dead registers, not fresh),
dropping to `base + last-K-live ≈ 256`. It is **right at the line** — confirm `vgpr_count ≤ 256`
with `TENSILE_DRYRUN_REGS` and the `.vgpr_count` directive. If a tile is over by a hair, that tile
cannot weave every store even with confinement (that is the signal that the weave-baseline
per-store deferral is needed for it) — degrade gracefully (fall back to ValuC reuse / monolithic
for that store), never overflow.

## Metric & verification

- Dry-run (`TENSILE_DRYRUN_REGS=1`): confirm the fold-data registers now land **in the K=0 holes**
  (print them; they must be disjoint from the last-K range), before running the GPU.
- Woven coverage: count store-gaps reaching quota in the MT256x256 `.s` (expect ≫16), plus
  `.vgpr_count ≤ 256`.
- `--verify` norm_error at the MXFP4 floor across STA shapes: 4096³, 8192³, 2048×2048×8192,
  4096²×1024, 8192×2048×4096. Silent-wrong-data is the risk class already hit once — full sweep
  every build. Build cache is content-hashed: delete `build_tmp/.../assembly/*.s`,
  `library/gfx950/*F4BS*.co`, `device-library/*.stamp` before a source-only rebuild.
- Perf A/B vs the base (all-fold, ~16 weave) on MT256x256 shapes (8192×6144×8192, 6144×8192×8192)
  and a good tile (8192×5120×8192 → 256x320), auto clocks (NEVER pin), median-of-7.

## Risks

- **Budget is right at 256** — may not fit every tile; needs the graceful fallback, not an overflow.
- **Checkout-ordering correctness** — forcing the checkout into a specific range via add/remove is
  fiddly; get it wrong and you either overflow or silently reuse a live register (corruption).
  `--verify` is the gate.
- **Keep the store-clobber guard** — it is the correctness backstop; confinement only *widens* the
  safe set, never bypasses the check.
