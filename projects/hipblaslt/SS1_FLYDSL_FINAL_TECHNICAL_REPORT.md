# SS1–FlyDSL Optimization Investigation

## Final Technical Report and Last-Push Recommendation

**Date:** 28 September 2026  
**Target:** AMD gfx950, SS1 MXFP4 GEMM  
**Primary shape:** `M=32768, N=28672, K=4096`  
**Repository:** `hipBLASLt` / TensileLite  
**Working branch:** `users/muzasyed/rolling-peeled-exit`  
**Last committed revision:** `255fa490b3` (`Optimize SS1 toward FlyDSL scheduling`)  
**Validation policy:** focused validation only; no repository-wide test suite was run  

---

## 1. Executive summary

This investigation attempted to close the performance gap between the TensileLite
SS1 MXFP4 kernel and FlyDSL on gfx950 by reproducing FlyDSL's scheduling,
operand-lifetime, scale-transport, LDS-layout, MFMA-order, and output-store
structure.

The work established that SS1 can express nearly all visible structural
properties of the FlyDSL kernel:

- the same 640 productive MFMAs;
- the same principal global-load, LDS-read, and output-store counts;
- direct packed scale transport from a Fly-compatible host layout;
- a scheduler-native full-K rolling peel;
- Fly-style N-major accumulator/store mapping;
- Fly-style dwordx2 output packets;
- parity-split A LDS addressing;
- path-specific deferred continuations for rolling exits;
- exact one-tile and mapped multi-workgroup output for the corrected candidate.

The final literal-order attempt corrected an important experimental error:
earlier SS1 traces forced a Bias epilogue while the FlyDSL reference had no
Bias or ScaleAlphaVec epilogue. With matched semantics, a runtime identity
store path and producer-native output retirement reduced the SS1 trace from
5,333 instructions and 62,132 cycles to 4,670 instructions and 49,068 cycles.
FlyDSL remains smaller at 4,047 instructions and 48,840 cycles.

The retained four-cell retirement queue is exact at `K=256`, `K=768`, and
`K=4096`. It measured **4.96854 PF/s** in the candidate experiment and
**4.89511–4.92849 PF/s** in the final three-run bracket, compared with
**5.03317 PF/s** for FlyDSL. A subsequent production stability bracket with
20 cold iterations measured **4.98564–5.01275 PF/s**, with a **5.00111 PF/s**
mean, only **0.64%** below FlyDSL. Literal instruction-for-instruction identity
was not achieved, but dynamic-cycle and production-throughput parity were
approached closely.

The main remaining difference is not generic store packetization. FlyDSL
retires output from the final MFMA producer stream with a shorter effective
accumulator lifetime and less address/control overhead. A one-cell delay made
the trace still closer but reduced production throughput; immediate retirement
was numerically incorrect. Four completed cells are therefore the minimum
validated practical latency queue for this implementation.

Sections 8–15 preserve the chronology and conclusions that existed before the
matched-semantics correction. Section 16 supersedes their final recommendation.

---

## 2. Repository and implementation state

### 2.1 Branch state

At the time of this report:

```text
branch: users/muzasyed/rolling-peeled-exit
HEAD:   255fa490b3 Optimize SS1 toward FlyDSL scheduling
```

The working tree contains uncommitted structural work:

```text
15 files changed
1,065 insertions
151 deletions
```

The modified areas are:

- client-side MX scale generation;
- SS1 Equality logic;
- parameter registration;
- instruction and logical schedulers;
- StoreInNLL;
- subtile global-read and scale emitters;
- KernelWriter and KernelWriterAssembly;
- focused Python tests;
- MXDataGen C++ tests.

The uncommitted work must not be confused with the last safe committed result.
It is a diagnostic and architectural candidate, not a shipping state.

### 2.2 Current focused logic configuration

The focused Equality solution currently enables:

```yaml
MXLoadInst: BufferLoad
MXScaleFormat: LaneContiguousGFX950
SubtileDirectToVgprScale: 3
SubtileFirstMfmaAccZero: 0
SubtileFlyKInner: 1
SubtileUnifiedKSchedule: 0
SubtileNativeWholeLoopScheduler: 0
SubtileFlyPhaseWait: 1
SubtileFlyNMajor: 1
SubtileFlyAParityLds: 1
SubtileFlyStorePacketGeometry: 0
SubtilePostBarrierNop: 1
SubtileScaleBatchPack: 0
SubtileSingleMainloopBarrier: 0
SubtileStoreInNLL: 1
SubtileRollingPeel: 1
```

The generated default has been restored after every rejected diagnostic. The
experimental exact-D-limit change is not active.

### 2.3 Validation state

The focused test set currently passes:

```text
241 passed
```

The tests are:

```text
tensilelite/Tensile/Tests/unit/test_subtile_store_in_nll.py
tensilelite/Tensile/Tests/unit/test_SubtileBasedLogicalScheduler.py
```

`git diff --check` passes.

No repository-wide validation was run, by explicit request.

`tensilelite-client` was not used to validate swizzled A/B inputs because it
cannot validate `swizzleA` and `swizzleB`. Swizzled correctness was checked with
`hipblaslt-bench`.

---

## 3. Performance reference points

### 3.1 Primary matched result

| Implementation | PF/s | Relative to FlyDSL | Status |
|---|---:|---:|---|
| FlyDSL reference | 5.03317 | baseline | external reference |
| Safe committed SS1 composition | 4.70607 | -6.50% | retained at `255fa490b3` |
| Current structural candidate | 4.61–4.62 | approximately -8.2% | exact successful runs, production fault |

The current structural work is therefore both slower and less stable than the
last committed composition.

### 3.2 Historical fast-drain milestone

An earlier implementation line reached:

| Shape | SS1 PF/s | Improvement | Gap to FlyDSL |
|---|---:|---:|---:|
| `32768×28672×4096` | 4.858 | +0.74% over its matched baseline | -3.48% |
| `32768×14336×4096` | 4.811 | +0.48% over its matched baseline | -4.03% |

That milestone reduced post-K instructions from 696 to 445 and reduced post-K
latency by 15–21%.

It is included for historical context only. It belongs to an earlier source
line and cannot be treated as the current branch's baseline without
reconstruction and a new matched comparison.

### 3.3 Best current composition before the structural rewrite

The retained scheduling composition produced:

```text
candidate runs: 4.70743, 4.70470 PF/s
candidate mean: 4.70607 PF/s
closing baseline: 4.66233 PF/s
matched improvement: +0.94%
```

The candidate was exact at `K=768` and `K=4096`.

Trace:

```text
/tmp/ss1-scheduling-compose-att
```

---

## 4. Methodology

The investigation followed a consistent experimental protocol:

1. make one structural or scheduling change;
2. generate only the focused gfx950 Equality kernel;
3. verify a small exact case;
4. capture a trace before rejecting the candidate;
5. benchmark on GPU 6 only;
6. avoid concurrent GPU jobs;
7. use matched or bracketed performance runs where practical;
8. remove rejected behavior from the active configuration;
9. restore the retained generated library;
10. remove generated `MatchTable.yaml` and matrix dumps.

This protocol matters because many candidates looked structurally closer to
FlyDSL while being slower, and several apparently plausible mappings produced
numerically valid one-tile output while failing in larger or alternate paths.

Evidence in this report is labeled implicitly as follows:

- **Proven:** directly established by execution, correctness, trace, or static
  address analysis.
- **Ruled out:** a controlled isolation retained the fault or regression.
- **Strong inference:** supported by multiple isolations but not yet directly
  observed at the faulting instruction.
- **Historical:** valid for an earlier source state, not directly comparable to
  the current candidate.

---

## 5. Architectural comparison with FlyDSL

### 5.1 Core instruction geometry

The retained scheduling composition reached FlyDSL's visible core operation
counts:

| Operation | SS1 composition | FlyDSL |
|---|---:|---:|
| Productive MFMA | 640 | 640 |
| `ds_read_b128` | 176 | 176 |
| `buffer_load_dwordx4` | 90 | 90 |
| output stores | 64 | 64 |

This was an important negative result: **static instruction-count parity is not
sufficient for throughput parity**.

The remaining performance difference is therefore dominated by:

- dependency placement;
- register lifetime;
- issue timing;
- phase order;
- wait placement;
- interaction between AGPR reads, conversion arithmetic, VMEM, and subsequent
  MFMA issue.

### 5.2 Synchronization

The original composition had seven barriers versus six in FlyDSL. A targeted
first-C0 change reached six barriers and exact core counts, but regressed:

```text
runs: 4.70392, 4.65138 PF/s
mean: 4.67765 PF/s
delta from retained composition: -0.60%
```

This proves that barrier count alone is not the controlling variable. The
location and surrounding phase schedule matter.

Similarly, reducing hot-loop barriers from four to two was confirmed in trace
but produced a mean of 4.64542 PF/s, 0.46% below its matched baseline.

### 5.3 Scale transport

FlyDSL consumes scales from a preshuffled workspace with far less scale-LDS
traffic. SS1 originally performed 48 scale LDS reads where FlyDSL used eight
direct loads.

The current work implemented:

- `MXScaleFormat::LaneContiguousGFX950`;
- optional client-side packed workspace generation;
- direct A+B scale loads;
- packed scale refill transport;
- SRD limit saturation;
- cross-phase carry for refill suffixes;
- trace-based scale-value comparison.

Important findings:

1. The packed host layout can be CPU-validated exactly.
2. Direct packed scale values consumed by MFMA can match the control trace.
3. A zero-limit SRD bug initially caused packed refills to return zero.
4. Correcting SRD limit construction fixed the regular long-loop path.
5. Additional short/NGLL path work was required for `K=512`.
6. Disabling packed scale transport entirely did **not** remove the production
   illegal-memory fault.

Packed scale transport is therefore not the remaining fault source.

### 5.4 Operand scheduling

A true Fly-style peel requires more than reordering the existing NLL. It must
coordinate:

- final NGLL and NLL phases;
- loop bounds;
- physical A/B tile ownership;
- refill issue and completion;
- last-use hazards;
- LDS read/write parity;
- distinct rolling-exit paths.

The current scheduler work introduced:

- rolling full-K peel support;
- explicit operand lifetime constraints;
- packed refill carry;
- path-specific deferred continuations;
- native schedule representations;
- tests for physical-path divergence.

A critical correctness result was that multiple rolling-exit paths use
different physical operand mappings. A single shared deferred continuation is
incorrect. Path-specific final macros are required.

### 5.5 A LDS mapping

Fly-style A parity-split LDS mapping was implemented and made exact.

Disabling it while retaining Fly N-major stores produced exact one-tile output,
but the production fault remained and appeared on the second K=2048 stress
process.

Therefore:

- A-parity LDS is not the source of the illegal-memory fault;
- it is not independently sufficient to improve performance;
- its value is structural compatibility with the Fly schedule.

### 5.6 N-major store mapping

The Fly N-major path required several correctness repairs:

- disabling first-MFMA accumulator zeroing in the incompatible mode;
- correcting producer-to-store mapping;
- correcting multi-workgroup D addressing;
- rebasing `SrdD` by `WorkGroup1` rather than duplicating the N coordinate in
  the vector address;
- using a 128-byte second-half offset;
- using a 256-byte vector-LDS stride for Bias and ScaleAlpha;
- preserving separate deferred macros for each physical rolling-exit path.

The resulting one-tile and mapped-grid stores are exact.

The output packet geometry is:

- four components;
- two row quarters;
- dwordx2 stores;
- 64 total stores;
- N-major tile/component traversal.

---

## 6. Experiment ledger

### 6.1 Retained or historically successful work

#### Packed alpha, rate-8 scheduling, paired alpha-one drain

Historical outcome:

- primary shape: 4.858 PF/s;
- secondary shape: 4.811 PF/s;
- post-K instructions: 696 to 445;
- post-K latency: 15–21% lower;
- exact across four orientations and alpha values 1.0 and 0.5.

The associated implementation combined:

- packed alpha scaling;
- issue rate 8;
- one runtime alpha branch;
- a paired no-multiply post-tail drain.

#### Scheduling composition

Retained at the current branch's last commit:

- 4.70607 PF/s mean;
- +0.94% over closing baseline;
- exact at K=768 and K=4096;
- 640 productive MFMAs;
- focused tests passing.

### 6.2 Rejected performance experiments

| Experiment | Result | Decision | Technical reason |
|---|---:|---|---|
| Two-column final-producer block | -1.7–1.8% | reject | loss of operand locality |
| Faithful full-K peel prototype | -3.9–4.0% | reject | structural peel overhead exceeded benefit |
| Fly N-major slab, 8 phases | 4.560–4.567 PF/s | reject | excess LR address swaps and VALU |
| Fly N-major slab, 2 phases | 4.616–4.628 PF/s | reject | improved over 8 phases, still below control |
| Single main-loop barrier | 4.64542 PF/s | reject | fewer barriers did not improve issue schedule |
| 256-byte hot-loop alignment | 4.63874 PF/s | reject | 28 executed nops, -0.60% |
| First-C0 synchronization parity | 4.67765 PF/s | reject | exact six-barrier parity, -0.60% |
| Full grouped MFMA/store interleave | 4.56286 PF/s | reject | 456–932-cycle VMEM stalls |
| Stop interleave one group early | approximately 3.76 PF/s | reject | severe throughput loss |
| 16-VGPR half staging | approximately 2.92 PF/s | reject | AGPR read/write serialization |
| Packet-paced dwordx2 stores | approximately 3.75 PF/s | reject | distributed VMEM pressure through MFMA stream |
| Null-bias alpha-one fast path | approximately 3.82 PF/s | reject | slow and often not selected |
| Bias-preserving alpha-one path | incorrect | reject | K=768 error ~0.332574; K=4096 error ~0.0699905 |

### 6.3 Store-interleaving trace result

The correct grouped interleave trace showed:

```text
52 stores before the final MFMA
12 stores after the final MFMA
median cycles: 53,768
serialized candidate: 52,480
retained control: 50,922
```

Trace:

```text
/tmp/ss1-path-interleave-correct-att
```

The primary stalls were large 456–932-cycle MFMA gaps associated with VMEM.

This closes the direction of inserting more stores into SS1's MFMA stream.
FlyDSL's overlap depends on a register allocation and schedule designed around
those stores from the start; late insertion does not reproduce that schedule.

---

## 7. Correctness history

### 7.1 Errors discovered and repaired

The following issues were found during the investigation:

1. first-MFMA accumulator zeroing was incompatible with Fly N-major mode;
2. initial Fly producer mapping selected the wrong accumulator quarters;
3. `WorkGroup1` was represented both in `SrdD` and vector addressing;
4. the second row-quarter offset was incorrectly treated as 256 bytes instead
   of 128;
5. Bias and ScaleAlpha LDS vectors used the wrong quarter stride;
6. one deferred continuation was reused across paths with different physical
   operand mappings;
7. packed scale SRD limits could become zero and return zero refill values;
8. the pre-Fly store path had a dormant missing scratch-VGPR alias;
9. the post-loop grouped-drain path had incorrectly become unconditional for
   all folded-store modes.

The last two issues were exposed while constructing the fault-isolation
candidate. They affect dormant pre-Fly generation rather than the active Fly
default.

### 7.2 Why correctness was not caught earlier

Several early checks were insufficient:

- one-tile validation cannot expose multi-workgroup rebasing errors;
- a successful hash at one K does not exercise all NGLL/NLL paths;
- `tensilelite-client` cannot validate swizzled A/B layouts;
- some incorrect addresses land in mapped allocations and do not fault;
- different rolling exits can consume different physical operand mappings;
- successful long benchmark processes can survive hundreds of dispatches,
  making the fault appear timing-related when it is allocation-dependent.

The improved correctness matrix eventually included:

- K=256, 512, 768, 4096 as path probes;
- both relevant orientations;
- one-tile output;
- 2-workgroup and larger mapped grids;
- grids through 16×16 workgroups;
- active and null vector operands where available;
- production-size repeated fresh processes.

---

## 8. Production illegal-memory fault

### 8.1 Observed behavior

The corrected structural candidate is exact on small and mapped grids, but
production execution is unstable.

Observed examples:

- K=4096: five fresh 300-dispatch processes succeeded; the sixth faulted;
- K=2048: faults can occur on the first or second fresh process;
- K=256: eight fresh processes completed;
- K=768: eight fresh processes completed;
- successful production runs measure approximately 4.61–4.62 PF/s.

The production stress shape is:

```text
M=32768
N=28672
K=2048 or 4096
cold iterations=100
timed iterations=200
rotating buffer=0
GPU=6
```

The process-level variability is important. A process often either survives
hundreds of dispatches or fails at synchronization. This is more consistent
with allocation-dependent memory behavior than a simple deterministic
arithmetic error.

### 8.2 Ruled-out causes

#### Packed scale transport

The kernel was regenerated with canonical LDS-backed scales while keeping the
scheduler and store composition.

Result:

```text
K=2048 faulted immediately
```

Trace:

```text
/tmp/ss1-canonical-scale-isolation-att
```

Conclusion: packed scale transport is not the fault source.

#### Scale SRD mutation before VMEM completion

A diagnostic `vmcnt(0)` was inserted before scale-SRD updates.

Result:

- exact small case;
- production K=2048 still faulted;
- execution became pathologically serialized.

Trace:

```text
/tmp/ss1-packed-srd-wait-att
```

Conclusion: this is not the root cause.

#### Cross-phase packed refill carry

Packed refills were forced to flush every phase.

Result:

- exact small case;
- production K=2048 faulted on the second process.

Trace:

```text
/tmp/ss1-packed-refill-flush-att
```

Conclusion: carry/deferral is not the root cause.

#### A-parity LDS

Fly N-major stores were retained while A-parity LDS was disabled.

Result:

- exact one-tile output;
- production K=2048 faulted on process 2.

Trace:

```text
/tmp/ss1-no-a-parity-att
```

Conclusion: A-parity LDS is not the fault source.

#### D-store out-of-bounds access

Static enumeration found that the nominal final dwordx2 store ends exactly at
the D allocation boundary:

```text
D allocation bytes:        1,879,048,192
last store start:           1,879,048,184
last store end, exclusive:  1,879,048,192
remaining margin:           0 bytes
```

A diagnostic replaced the unbounded D SRD limit with the exact number of bytes
remaining after each `WorkGroup1` SRD rebase.

Result:

- exact one-tile output;
- trace captured;
- production K=2048 still faulted immediately.

Trace:

```text
/tmp/ss1-exact-d-limit-att
```

Conclusion: the illegal access is not an out-of-range Fly D store.

### 8.3 Pre-Fly store isolation

Disabling Fly N-major stores also required disabling the dependent A-parity LDS
mode.

After repairing dormant pre-Fly generation:

- eight K=2048 stress processes completed without an illegal-memory fault;
- one-tile output was numerically wrong with norm error `0.32887`.

Trace:

```text
/tmp/ss1-prefly-store-isolation-att
```

This result associates the fault with Fly-only post-loop behavior, but it is not
a valid proof that the Fly stores themselves are wrong because the isolation
also changed A layout and produced incorrect output.

The later exact no-A-parity test and exact-D-bound test are stronger:

- Fly post-loop path retained;
- output exact;
- fault retained.

### 8.4 Remaining fault domain

The active Fly post-loop path performs the following:

1. reconstruct post-loop epilogue SGPRs;
2. reconstruct `SrdD`, `SrdC`, `SrdBias`, and `SrdScaleAlphaVec`;
3. rebase `SrdD` by `WorkGroup1`;
4. issue Bias and ScaleAlphaVec global loads;
5. write vectors to LDS;
6. read vector quarters from LDS;
7. stage accumulators;
8. apply alpha, ScaleAlpha, and Bias;
9. issue bounded or unbounded D stores.

The production performance command uses no bias vector, yet the generated path
still issues:

```asm
buffer_load_dword ... SrdBias
buffer_load_dword ... SrdScaleAlphaVec
```

The null pointers are represented through null SRD bases and zero limits. Such
loads should be suppressed by buffer bounds and return zero, but they are now
the highest-probability remaining memory operations because:

- packed scales are ruled out;
- D stores are ruled out;
- A LDS is ruled out;
- the fault appears only with the Fly post-loop composition;
- this vector initializer is unique to that composition.

This is a strong inference, not yet a direct fault-instruction capture.

---

## 9. Trace index

| Purpose | Trace path | Principal evidence |
|---|---|---|
| Retained scheduling composition | `/tmp/ss1-scheduling-compose-att` | 640 productive MFMAs; matched core counts |
| Single-mainloop barrier | `/tmp/ss1-single-mainloop-barrier-att` | hot-loop barriers reduced 4 to 2 |
| N-major slab phases | `/tmp/ss1-fly-nmajor-slab-phase-{v1,correct,phase2}-att` | phase overhead |
| Correct grouped interleave | `/tmp/ss1-path-interleave-correct-att` | VMEM-induced MFMA gaps |
| Stop interleave early | `/tmp/ss1-interleave-stop-early-att` | severe regression |
| Half staging | `/tmp/ss1-halfstage-interleave-att` | AGPR/VGPR serialization |
| Packet pacing | `/tmp/ss1-packet-paced-interleave-att` | scattered VMEM pressure |
| Alpha-one shortcut | `/tmp/ss1-alpha-one-tail-att` | executed incorrect candidate |
| Initial deferred interleave | `/tmp/ss1-deferred-interleave-att` | failed path experiment |
| Boundary-control interleave | `/tmp/ss1-deferred-boundary-control-att` | failed boundary experiment |
| Packed scale addresses | `/tmp/ss1-packed-address-k512-tracer` | zero SRD limit and zero refill banks |
| Corrected folded candidate | `/tmp/ss1-folded-correct-att` | production post-loop drain |
| SRD wait diagnostic | `/tmp/ss1-packed-srd-wait-att` | wait did not fix fault |
| Refill-flush diagnostic | `/tmp/ss1-packed-refill-flush-att` | flush did not fix fault |
| Canonical-scale isolation | `/tmp/ss1-canonical-scale-isolation-att` | fault without packed transport |
| Pre-Fly store isolation | `/tmp/ss1-prefly-store-isolation-att` | no fault, but incorrect output |
| No-A-parity isolation | `/tmp/ss1-no-a-parity-att` | exact; fault remained |
| Exact D-store bounds | `/tmp/ss1-exact-d-limit-att` | exact bounded stores; fault remained |

An attempted all-store production register trace was stopped. The tracer could
not select a boundary workgroup and trapped millions of waves at entry. It
produced no usable address snapshots and is not treated as evidence.

---

## 10. What should not be tried again

The following directions have sufficient negative evidence and should be
considered closed:

1. inserting more output stores into the MFMA stream;
2. reducing store staging from 32 VGPRs to 16 VGPRs;
3. packet-paced stores;
4. stopping interleaving early;
5. alpha-one arithmetic shortcuts;
6. barrier-count matching without a dependency-level reason;
7. hot-loop alignment padding;
8. arbitrary nop insertion;
9. forcing packed scale VMEM waits;
10. forcing packed refill flushes;
11. disabling packed transport as a fault fix;
12. disabling A-parity LDS as a fault fix;
13. changing D-store packet width as a fault fix;
14. composing every FlyDSL difference simultaneously;
15. broad launch tuning before the fault is resolved.

These directions are not merely speculative failures. Most were generated,
executed, verified, traced, and benchmarked.

---

## 11. Durable engineering outcomes

Even if the project stops without reaching FlyDSL parity, the investigation
produced reusable capabilities:

1. **Packed MX scale ABI**
   - lane-contiguous gfx950 scale layout;
   - host generation;
   - CPU tests;
   - direct kernel loading.

2. **Scheduler infrastructure**
   - rolling peel representation;
   - packed refill carry;
   - explicit lifetime events;
   - path-specific continuations.

3. **Fly-compatible store geometry**
   - N-major producer mapping;
   - dwordx2 packets;
   - multi-workgroup address handling;
   - vector epilogue staging.

4. **A-parity LDS support**
   - explicit geometry;
   - validation and rejection logic;
   - focused tests.

5. **Focused test expansion**
   - 241 scheduler/store tests;
   - physical-path correctness assertions;
   - store-coordinate tests;
   - generated structure checks.

6. **Trace discipline**
   - candidates were traced before rejection;
   - instruction-count parity was separated from dependency quality;
   - correctness was checked across path-sensitive K values.

---

## 12. Interpretation of the remaining FlyDSL gap

The investigation does not support the conclusion that TensileLite or SS1 lacks
the necessary ISA capability. The kernel can emit the relevant instructions
and reproduce FlyDSL's broad architecture.

The evidence instead supports three conclusions.

### 12.1 FlyDSL's advantage is schedule quality, not an isolated optimization

FlyDSL's register allocation, producer order, scale transport, and store issue
were designed together. Copying any one visible feature into SS1 can make the
static instruction stream look closer while worsening the dynamic schedule.

### 12.2 Operand lifetime is the central constraint

Most regressions arose when an optimization extended a live range or inserted
an AGPR/VGPR/VMEM dependency into a region previously optimized for MFMA issue.

The successful structural work required explicit lifetime modeling. The failed
performance work generally violated or lengthened those lifetimes even when it
reduced instruction count.

### 12.3 Tail optimization is phase-order sensitive

The user hypothesis that tail tuning should occur after the main schedule was
substantially validated. Tail changes that looked attractive in isolation
regressed after composition because they changed:

- phase boundaries;
- wait placement;
- address swaps;
- accumulator availability;
- VMEM clustering.

There is no evidence that another isolated tail reorder will close the gap.

---

## 13. Recommended final push

### 13.1 Objective

Produce a stable, explainable final repository state. Performance improvement is
secondary.

### 13.2 One permitted experiment: null-vector post-loop path

Modify only the Fly post-loop vector initialization:

- if Bias is null, do not issue Bias VMEM;
- initialize Bias LDS values to `0.0`;
- if ScaleAlphaVec is null, do not issue ScaleAlphaVec VMEM;
- initialize ScaleAlpha LDS values to `1.0`;
- preserve the current active-pointer path unchanged;
- preserve packed scales, rolling scheduler, A-parity LDS, accumulator staging,
  and D stores unchanged.

This is the cleanest remaining isolation because the production performance
case uses null vector operands and still executes the vector loads.

### 13.3 Required validation order

1. Run the 241 focused tests.
2. Generate only the focused Equality kernel.
3. Verify `256×256×768`.
4. Verify `256×256×4096`.
5. Capture ATT before a retention decision.
6. Run 12 fresh production processes at K=2048:

   ```text
   M=32768
   N=28672
   cold=100
   timed=200
   rotating=0
   GPU=6
   ```

7. Run 12 fresh production processes at K=4096.
8. Verify active Bias separately.
9. Verify active ScaleAlphaVec separately.
10. Run one matched performance bracket only after stability is established.

No GPU jobs should run concurrently.

### 13.4 Decision procedure

#### Case A: fault disappears

Then the null-vector load path is the root cause or necessary trigger.

Required follow-up:

- verify active Bias and ScaleAlphaVec;
- compare trace structure;
- benchmark once;
- retain only if correctness and stability pass;
- commit the final state;
- stop.

#### Case B: fault remains

Stop immediately.

Actions:

- reject the null-vector experiment;
- restore generated artifacts;
- remove `MatchTable.yaml` and dumps;
- return to commit `255fa490b3`;
- preserve this report and trace index;
- do not start another optimization family.

#### Case C: fault disappears but performance regresses

Choose based on intended deliverable:

- if the structural infrastructure itself must be preserved, retain the stable
  path and record the regression;
- if the goal is the best production kernel, restore `255fa490b3`.

Do not tune the stable path further.

#### Case D: correctness fails

Reject immediately and restore `255fa490b3`.

---

## 14. Final stop criteria

The project should end after the final push when one of these is true:

1. the null-vector path is stable, correct, traced, benchmarked, and committed;
2. the null-vector path fails and `255fa490b3` is restored;
3. active vector operands cannot be validated exactly;
4. any new unexplained production fault appears.

Before final handoff:

- run focused tests only;
- run `git diff --check`;
- remove generated `MatchTable.yaml`;
- remove matrix dumps;
- record the selected commit;
- record final trace paths;
- record final production stress counts;
- do not leave a rejected candidate in the generated library;
- do not leave uncommitted diagnostics mixed with the chosen final state.

---

## 15. Final conclusion

The project did not achieve FlyDSL parity.

It did demonstrate that the difference is not a simple architecture or opcode
limitation. SS1 can reproduce FlyDSL's broad compute geometry, scale transport,
LDS organization, rolling peel, and store packetization. The remaining
performance gap is primarily a dynamic scheduling and lifetime-allocation
problem.

The current uncommitted structural implementation is not suitable for final
retention in its present state:

- it is slower than the last safe commit;
- it has an unresolved production illegal-memory fault;
- it changes 15 files and more than one thousand lines;
- the fault survives the principal scale, scheduler, A-LDS, and D-store
  isolations.

The technically responsible conclusion is to permit exactly one more isolation
of the Fly-only null Bias/ScaleAlpha vector-load path. If that does not produce
a stable result, preserve `255fa490b3` and end the project.

---

## 16. Final literal-order attempt

This section records the last experiment and supersedes the stop recommendation
in Sections 13–15.

### 16.1 Matched-semantics correction

The original plain ATT helper added `--bias_vector`. The FlyDSL trace did not
execute Bias or ScaleAlphaVec work. Consequently, the earlier comparison mixed
instruction-order differences with different mathematical epilogues.

The corrected SS1 trace used:

```text
alpha=1
beta=0
Bias=null
ScaleAlphaVec=null
activation=none
M=32768
N=28672
K=4096
```

The corrected trace input is `/tmp/ss1_literal_trace.yaml`.

### 16.2 Baseline after semantic matching

```text
FlyDSL:
  dynamic instructions:       4,047
  dynamic cycles:             48,840
  instructions after MFMA:    32

SS1 before identity retirement:
  dynamic instructions:       5,333
  dynamic cycles:             62,132
  instructions after MFMA:    1,162
  v_pk_mul_f32:                256
  v_pk_add_f32:                128
```

This showed that the dominant late-trace discrepancy was a real epilogue
structure problem, but its magnitude had been overstated by unmatched Bias
semantics.

### 16.3 Implemented identity path

`SubtileIdentityStoreFastPath` is selected only when all of the following hold:

- `alpha == 1`;
- Bias is null;
- ScaleAlphaVec is null;
- the existing kernel/layout eligibility checks pass.

The generic epilogue remains the fallback. The fast path:

- hoists and checks the Bias and ScaleAlphaVec pointers;
- retires output from the final rolling MFMA schedule;
- reads four accumulator values for each completed output cell;
- converts directly to packed BF16;
- emits four `buffer_store_dwordx2` packets per completed group;
- reuses the four output addresses across the two row halves;
- terminates locally after the completed identity path.

### 16.4 Accumulator-latency result

Producer retirement cannot be immediate. The tested queue depths established:

```text
immediate retirement:
  K=768 norm error:   approximately 0.4026
  K=4096 norm error:  approximately 0.0829
  decision:           reject

one-cell delay:
  trace cycles:       49,060
  post-MFMA count:    28
  throughput:         approximately 4.88–4.92 PF/s
  decision:           reject on throughput

four-cell delay:
  exact:              K=256, K=768, K=4096
  throughput:         up to 4.97131 PF/s
  decision:           retain
```

The four-cell queue is therefore not arbitrary padding. It is the shortest
tested schedule that was both exact and consistently competitive.

### 16.5 Retained trace

The retained address-reuse composition is:

```text
trace:
  /tmp/ss1-reuse-address-att

dynamic instructions:        4,670
dynamic cycles:              49,068
instructions after MFMA:     52
v_mul_lo_u32:                 32
v_add_u32_e32:                64
```

Relative to the matched SS1 baseline, this removes 663 dynamic instructions,
13,064 cycles, and 1,110 instructions after the final MFMA. Relative to FlyDSL,
the residual is 623 instructions, 228 cycles, and 20 post-MFMA instructions.

An incremental-address variant removed eight more `v_mul_lo_u32`
instructions:

```text
trace:
  /tmp/ss1-incremental-address-att

dynamic instructions:        4,662
representative wave cycles:  49,164
instructions after MFMA:     51
v_mul_lo_u32:                 24
v_add_u32_e32:                64
throughput:                   4.91967 PF/s
```

It was traced before rejection. Despite the lower instruction count, it was
slower than the retained address-reuse composition, confirming that static or
dynamic instruction count alone is not a sufficient retention criterion.

### 16.6 Final focused validation

No repository-wide validation was performed.

```text
focused Python tests:
  241 passed

hipblaslt-bench exactness:
  256x256x256:   norm_error=0
  256x256x768:   norm_error=0
  256x256x4096:  norm_error=0

production repetitions:
  4.89511 PF/s
  4.92849 PF/s
  4.91386 PF/s
  mean: 4.91249 PF/s

production stability, 20 cold + 20 timed:
  K=2048: 3/3 completed without a fault
  K=4096: 3/3 completed without a fault
  K=4096: 5.00493, 4.98564, 5.01275 PF/s
  K=4096 mean: 5.00111 PF/s
```

The earlier retained-candidate measurement was 4.96854 PF/s. The observed
range indicates nontrivial warm-up and environment sensitivity. Under the
final stability protocol the mean residual gap is 0.64%; across all final
brackets the observed mean gap is approximately 0.6–2.4%.

### 16.7 Final technical conclusion

The final attempt did not recreate FlyDSL instruction-for-instruction. It did
establish the mechanism responsible for most of the gap:

1. epilogue semantics must be matched before interpreting traces;
2. output must retire from the final producer stream rather than from a large
   generic post-loop epilogue;
3. accumulator reads require a nonzero producer-to-consumer latency;
4. queue depth changes can improve trace similarity while hurting throughput;
5. address arithmetic reductions can likewise regress throughput despite
   reducing instruction count.

The best retained structure is the four-cell producer-retirement queue with
per-tile address initialization and reuse across both row halves. It closes
most of the measured gap while preserving exactness. The remaining 228 traced
cycles are small enough that further source-level imitation would require
hardware-sensitive allocation and issue-order control, not another broad
optimization family.

