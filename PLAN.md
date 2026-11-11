# Plan: weave-as-baseline, fold-when-affordable (minimize the unfolded-weaved prefix)

Status: design, not yet implemented. Branch context: `users/muzasyed/coalesce-weave`
(off `users/muzasyed/dpp-dppsf-singular`, PR #14644). gfx950 F4BS MXF4→bf16 TN subtile store.

## TL;DR

Invert the current store architecture. Today the **DPP fold** (store coalescing) is the
baseline — it fires on every store via a per-kernel gate — and the **MFMA weave** (store/MFMA
overlap) is bolted on and starved. Instead, make the **weave the baseline** (every store gets its
overlap, exactly as develop's paired store does) and add the **fold incrementally**, per store,
only once enough registers have retired to pay for it. The early stores run weave-only (unfolded);
the goal is to make that unfolded prefix as short as possible.

## Background: two orthogonal optimizations on the subtile store

- **DPP fold / coalesce** (`_emit16bitSubtilePairedStoreCoalesce`): merges two M-adjacent
  half-cacheline bf16 `buffer_store_dwordx4` into all-lanes DPP-blended full-line stores. Kills
  write amplification (WA 1.30→1.00). **The big win: +8.5–21%.**
- **MFMA weave** (the convert-interleave): the block scheduler moves terminal (last-K) MFMAs into
  the store's convert-gap so the store's `v_cvt_pk` converts hide in the MFMA issue shadows.
  Develop's un-folded paired store (`_emit16bitSubtilePairedStoreWoven`) does this at ~29/32.
  **The small win: ~+0.6–1%.**

Both want registers. The fold needs its coalesce batch + DPP scratch; the weave needs the terminal
MFMAs' operands to stay **live** into the store so the moved MFMAs can read them.

## The register conflict (what the dry-run proved)

On the tight tile **MT256x256 MIWT8_8** the dry-run instrumentation showed:

- 170 operand registers total: **68 retired K=0 holes** (dead before the terminal MFMAs) +
  **102 last-K-live** registers (v56..157) that the terminal MFMAs still read.
- The fold's **scratch** (9-reg cvt block) already lands entirely in the retired K=0 holes
  (v36..44) — it is *not* the problem.
- The fold's **data** is the problem: the DPP blend writes `vCoalesceStoreData` (= `sumIdx1`, a
  ValuC slot that physically lands in the last-K range), i.e. 88 `v_cndmask` writes + address math
  land on exactly the registers the moved MFMAs need. That is the clobber that today caps the
  weave at ~16 moves (~4 of N stores covered).

Key point: **an un-folded weave-only store does not have that data.** It is develop's plain paired
store — small footprint, no last-K writes — so it weaves cleanly. Folding a store is what injects
the clobbering data.

## The model

1. **Weave is the floor.** Every store emits `_emit16bitSubtilePairedStoreWoven` and gets its
   MFMA overlap. This is the proven-correct develop path; no fold data, no last-K clobber.
2. **Fold is layered on where affordable.** As the store code descends, each `buffer_store`
   retires its ValuC → registers free up. Once the running free count covers the fold's need
   (coalesce batch + DPP scratch), switch that store (and the rest) to
   `_emit16bitSubtilePairedStoreCoalesce` (fold + weave).
3. **Minimize the unfolded prefix.** The number of stores that are weave-only (before the fold
   becomes affordable) is the metric to drive down. Every store above the switch point keeps its
   WA (lost on that store) but keeps its overlap.

Why this is better than today's "fold-all, weave-little": it captures **both** wins —
weave on 100% of stores, fold on (100% − prefix) of stores — where today we get WA on 100% but
overlap on only ~4/N. As long as the prefix is short (a store retires ~10 regs; the fold needs
~12–14), the WA lost on the prefix is tiny and the overlap gained across all stores is new.

## Design / implementation

Both emitters already exist, so the core work is **dispatch + accounting**, not new codegen.

1. **Make the fold a per-store decision, not a per-kernel gate.**
   - Today: `isSubtileCoalesce` (KWA `globalWriteElements`, ~19255) is computed once and every store
     coalesces. Replace the unconditional dispatch to `_emit16bitSubtilePairedStoreCoalesce`
     (call sites ~2685/2699/4314 in `GlobalWriteBatch.py`) with a per-store choice.
2. **Track retirement.** Maintain a running count of registers freed by stores already issued
   in this store sequence (each pair's ValuC retires after its `buffer_store`). The retirement
   order is deterministic/compile-time (the element loop order), so this is a static walk.
3. **Switch point = first store where running-free ≥ fold-need.** Emit
   `_emit16bitSubtilePairedStoreWoven` before it, `_emit16bitSubtilePairedStoreCoalesce` from it on.
   Fold-need = coalesce batch extra + the 2 DPP scratch regs; the fold's data then lives in the
   now-free retired holes (see the K=0-confinement note — the folded stores past the switch point
   must still place their blend data in retired holes, not back onto live last-K).
4. **Keep the store-clobber guard** (`_planCapturedTerminalMfmas`, LogicalScheduler) unchanged — it
   is the load-bearing correctness fix and will simply pass more moves once the folded stores stop
   writing last-K.

## Metric & verification

- Report: **unfolded-prefix length** (count of weave-only stores), stores folded, stores
  woven-to-quota, and `.vgpr_count` (must stay ≤ 256). Dry-run instrumentation already in place
  (`TENSILE_DRYRUN_REGS`).
- `--verify` norm_error at the MXFP4 floor across STA shapes (4096³, 8192³, 2048×2048×8192,
  4096²×1024, 8192×2048×4096) — the load-bearing correctness gate. Silent-wrong-data is the risk
  class we have already hit; every build gets the full sweep.
- Perf A/B vs the current all-fold build on the tiles that dispatch MT256x256 (8192×6144×8192,
  6144×8192×8192) and the already-good tiles (256x320: 8192×5120×8192), auto clocks, median-of-7.

## Risks / open questions

- **Prefix may not be short if retirement is coarse.** If the first store doesn't retire enough
  to afford the fold, the prefix grows. Need to confirm per-store retirement granularity.
- **WA accounting on the prefix.** The unfolded stores have WA>1.0 on their own bytes. Net perf
  must beat the current all-fold build; measure, don't assume.
- **Folded stores past the switch still need their data in retired holes** (the K=0-confinement
  sub-problem) or they re-introduce the clobber. The switch point gives them the room; the
  allocation must actually use it.
- **Edge/partial-M tiles** already fall back to un-coalesced stores; keep that path intact.

## Relationship to the K=0-confinement alternative

The K=0-confinement idea (keep fold-on-all, but force the fold's data into the retired K=0 holes
instead of onto live last-K operands) targets the same end — more stores fold+weave — with a
smaller, less-invasive change, but it keeps fighting register pressure on every store. The
weave-baseline plan here instead *avoids* the fight on the register-starved top of the store by not
folding there. They are not mutually exclusive: the folded stores in this plan must themselves
place their data in retired holes, i.e. the K=0-confinement mechanism is a component of step 3.
