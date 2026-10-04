---
theme: default
colorSchema: light
title: Emulating FP64 on low-precision accelerators
info: |
  ## Emulating FP64 on low-precision accelerators
  par_gemul8 and the Ozaki scheme on JUPITER.
fonts:
  sans: Arimo
  mono: Cousine
date: October 2026
layout: cover
author: Conor McLeod
institute: JSC Guest Student Programme
drawings:
  persist: false
transition: slide-left
comark: true
---

# Node-level parallelisation of Ozaki Scheme II

Accelerating a Multi-GPU implementation of Ozaki Scheme II

Optimising Multi-GPU FP64 GEMM emulation on JUPITER GH200s

<!--
TODO: confirm title, subtitle and affiliation line.
-->

---

# Motivation: hardware is going low precision

Driven by the AI boom, Nvidia's accelerators increasingly prioritise low-precision work.

- Deep learning (LLM training and inference) works well in low precision, even as low as FP4
- That is where the money is being deployed
- Nvidia is rationally incentivised to follow it

<!--
Unlike scientific computing, deep learning workloads like LLM training and inference work well in low precision, even as low as FP4.

There are currently trillions of dollars being deployed.

Nvidia, as a profit-making enterprise, is rationally incentivised to pursue this.

TODO: finish these two sentences; add a source for the spending figure.
-->

---

# Motivation

NVIDIA hardware increasingly favours low precision

<img src="./plots/nvidia_precision.png" alt="Nvidia flagship GPUs: low-precision tensor throughput versus native FP64, 2020 to 2026" class="mt-4 w-full bg-white rounded p-2" />

<!--
Low-precision throughput has grown by orders of magnitude, while native FP64 has stagnated and even been cut (B300).

TODO: verify B200/Rubin figures and the Rubin emulated-FP64 marker before presenting.
-->

---

# Motivation: where that leaves us

Scientific simulations have long operated in the high-precision FP64 regime.

The onus falls on us to find ways to _emulate_ FP64 algorithms in lower precision.

<!--
This leaves scientific simulations, long operating in FP64, in a pickle.

TODO: finish "Nvidia will ..." and "To obtain high precision performance, ...".
-->

---

# Ozaki scheme II

An algorithm for emulating high-precision matrix multiplication using low-precision operations, using the _Chinese Remainder Theorem_.

- Split each matrix into slices small enough that their products are exact
- Multiply the slices with low-precision GEMMs
- Sum the partial products to recover the FP64 result

<!--
TODO: check these bullets against how you want to present scheme I; add a diagram of the slicing.
-->

---

# Chinese Remainder Theorem

Let $x \in \mathbb{Z}$, and let $p_1, \dots, p_N \in \mathbb{N}_{\ge 2}$ be pairwise coprime with $\mathcal{P} := \prod_{i=1}^{N} p_i$.

- Let $q_i \in \mathbb{N}$ be the modular inverse of $\mathcal{P}/p_i$, such that $\frac{\mathcal{P}}{p_i}\, q_i \equiv 1 \pmod{p_i}$
- Suppose $x$ is known only through its residues:

$$
x \equiv y_1 \pmod{p_1}, \quad \dots, \quad x \equiv y_N \pmod{p_N}
$$

- Then $x$ is recovered modulo $\mathcal{P}$:

$$
x \equiv \sum_{i=1}^{N} \frac{\mathcal{P}}{p_i}\, q_i\, y_i \pmod{\mathcal{P}}
$$

- This is a **weighted sum** of the residues: each $y_i$ gets a fixed weight $\frac{\mathcal{P}}{p_i}\, q_i$ that depends only on the moduli

<!--
N pairwise coprime moduli, each at least 2; their product is P.

TODO: why N_{>=2}? (a modulus of 1 carries no information.)

The modular inverse q_i is the number you multiply P/p_i by to get 1 mod p_i. It exists because P/p_i is coprime to p_i.

The system is N congruences with the same x on the left; each y_i is x mod p_i.

Spelled out: x = P/p_1 q_1 y_1 + P/p_2 q_2 y_2 + ... + P/p_N q_N y_N (mod P).

The weights are where the work happens: the residues change with x, the weights never do. Next slide: why these particular numbers.
-->

---

# Why the weights work

Call $e_i := \frac{\mathcal{P}}{p_i}\, q_i$ the $i$-th weight. Each factor has one job:

- $\frac{\mathcal{P}}{p_i}$ contains every **other** modulus, so $e_i \equiv 0 \pmod{p_j}$ for all $j \ne i$
- $q_i$ rescales it so that $e_i \equiv 1 \pmod{p_i}$

So $e_i$ is a **switch**: $1$ through modulus $p_i$, $0$ through all the others. Reducing $\sum_i e_i\, y_i$ mod $p_j$ kills every term but $e_j y_j \equiv y_j$.

<div class="grid grid-cols-2 gap-8 mt-4 text-sm">
<div>

| $p = (3, 5, 7)$, $\mathcal{P} = 105$ | $\bmod 3$ | $\bmod 5$ | $\bmod 7$ |
|---|:-:|:-:|:-:|
| $e_1 = 35 \cdot 2 = 70$ | $1$ | $0$ | $0$ |
| $e_2 = 21 \cdot 1 = 21$ | $0$ | $1$ | $0$ |
| $e_3 = 15 \cdot 1 = 15$ | $0$ | $0$ | $1$ |

</div>
<div>

$x = 52$ has residues $y = (1, 2, 3)$:

$$
70 \cdot 1 + 21 \cdot 2 + 15 \cdot 3 = 157 \equiv 52 \pmod{105}
$$

The weights depend only on the moduli, so they are precomputed once (`qPi` in par_gemmul8).

</div>
</div>

<!--
Think of the residues (y_1, ..., y_N) as coordinates of x. The weights are the unit vectors in those coordinates: e_1 looks like (1, 0, 0), e_2 like (0, 1, 0), and so on. x is then just y_1 e_1 + y_2 e_2 + ..., exactly like writing a vector in the standard basis.

Same idea as Lagrange interpolation: each basis polynomial is 1 at its own node and 0 at the others, so the sum hits every data point.

Why P/p_i: it is the product of all the moduli except p_i, so it is divisible by every p_j with j != i. That gives the zeros. It is coprime to p_i, so it has an inverse q_i mod p_i; multiplying by q_i turns its residue mod p_i into 1 without disturbing the zeros.

Example: 35 = 5 * 7 is 2 mod 3; the inverse of 2 mod 3 is 2, so e_1 = 70. 21 = 3 * 7 is already 1 mod 5, and 15 = 3 * 5 is already 1 mod 7.

The sum is only determined mod P: adding any multiple of P keeps every residue the same. That is why we reduce mod P at the end, and why x must fit in P (the next slide).
-->

---

# The CRT in action

<CrtExplorer />

<!--
Each dial is one int8 modulus from par_gemmul8; the hand is the signed residue y_i, which is what fits in int8.

Press reconstruct: the dot walks round the ring Z/P one term (P/p_i) q_i y_i at a time and always lands on x mod P.

Whether that is x itself depends on P: try int64 max with N = 8 (wraps), then N = 9 (exact).
-->

---

# Ozaki scheme II

Uses the Chinese Remainder Theorem.

- Scale $A$ and $B$ to integer matrices $A'$, $B'$
- Compute $C_i = A'B' \bmod m_i$ for pairwise-coprime moduli $m_i$
- Reconstruct $A'B'$ from the $C_i$ with the CRT, then scale back

<!--
TODO: check these bullets; say why scheme II beats scheme I (GEMM count), and how many moduli are needed for FP64.
-->

---
clicks: 5
---

# Ozaki scheme II, step by step

<OzakiSteps :step="$clicks" />

<!--
A real 4x4 by 4x3 product, run through the same accurate-mode pipeline as par_gemmul8's seq::ozaki_gemm (ported from ozaki-numpy).

1. FP64 inputs, spread over six decades.
2. Each row of A and column of B gets its own power of two, chosen as large as the bit budget allows; then truncate. This is the only place we lose accuracy.
3. Slice: residues mod each p_i, all int8.
4. N independent int8 GEMMs, this is the expensive part and what we parallelise.
5. CRT gives back A'B' exactly (see the CRT slides before).
6. Undo the powers of two. Drag N: error falls ~4 bits per modulus until FP64's own limit.
-->

---

# The platform

JUPITER

- Each node has 4x Grace Hopper 200 superchips

<!--
I'm grateful to have been able to spend this summer using JUPITER nodes, each of which has 4x Grace Hopper 200 superchips.

TODO: node diagram or photo; memory and interconnect figures if relevant.
-->

---

# par_gemull8

- Started from a functional prototype put together using AI tools
- Around 14k lines to productionise and streamline

Three paths:

- `run_par_ozaki`
- `run_par_ozaki_async`
- `run_ref_par_ozaki`

<!--
We had a functional prototype put together using AI tools, but work had to be done to productionise and streamline around 14k lines.

TODO: one line on what par_gemul8 is before the history.
-->

---

# Distributing the matrices

Each MPI rank owns one tile of every matrix.

<img src="./svg/input_output_tiles.svg" alt="A, B and C each split into tiles across a 2x2 process grid, one tile per MPI rank" class="mt-8 w-full bg-white rounded p-4" />

<!--
TODO: talk through the process grid (prow/pcol) and the local tile sizes.
-->

---
clicks: 6
---

# How the tiles move

<TileExchange :step="$clicks" />

<!--
Each panel is one GPU, laid out in its place on the 2x2 grid. The mini grids show which tiles of A, B and C that GPU holds; colour is the rank the tile belongs to, ticks are the int8 residue planes it holds, one per modulus.

Clicks step through the par_ozaki stages. The toggle switches to ref_par_ozaki, which only has three stages.

ref: swap FP64 tiles with row/column peers, then two full sequential Ozaki runs per rank.
par: split the moduli, not the tiles. Expand locally, all-to-all the planes, one big GEMM per owned modulus, all-to-all the C tiles back, CRT locally.

Click a rank to isolate its traffic in the all-to-all stages.
-->

---

# Step one: profile the code

We added NVTX ranges throughout the codebase.

<!--
The first step was to profile the code. We added nvtx ranges throughout the codebase to do this.

TODO: a snippet showing an NVTX range, and a profiler timeline screenshot.
-->

---

# CUTLASS

CUTLASS is .

- Fused moduli + gemm step
- Tuned to the GPU, uses all the SMs

---

# Side quest: every profile in one database <img src="/svg/duckdb_inline_lightmode.svg" alt="DuckDB" class="inline h-10 align-middle ml-1" />

Opening `.nsys-rep` files one at a time doesn't scale to 4 ranks × 3 methods × dozens of jobs.

<div class="grid grid-cols-5 gap-2 mt-3 text-center">
  <div class="stat"><b>28</b><span>profiled jobs</span></div>
  <div class="stat"><b>336</b><span>rank profiles</span></div>
  <div class="stat"><b>15</b><span>GH200 nodes</span></div>
  <div class="stat"><b>27 GB→385 MB</b><span>SQLite → parquet</span></div>
  <div class="stat"><b>0.3 s</b><span>per question</span></div>
</div>

<div class="pipe mt-3">
  <span>jupiter<br><small>nsys, 4 ranks</small></span><i>→</i>
  <span>.nsys-rep<br><small>sync.sh</small></span><i>→</i>
  <span>.sqlite<br><small>nsys export</small></span><i>→</i>
  <span>facts/*.parquet<br><small>build.py</small></span><i>→</i>
  <span class="hl">DuckDB views<br><small>setup.sql</small></span><i>→</i>
  <span class="hl">questions/*.py<br><small>one file each</small></span>
</div>

<div class="grid grid-cols-[1fr_1.05fr] gap-6 mt-3">
<div>

- A `.nsys-rep` **is a SQLite database**: `nsys export` dumps it, and DuckDB reads it directly
- Every row carries **`job_id · method · rank · hostname`**, and every timestamp is shifted to absolute UTC, so ranks and jobs share one time axis
- Comparing across runs, ranks or methods becomes a `GROUP BY`

</div>
<div>

```sql
-- every fused GEMM launch, every job, every rank
select hostname, rank, avg(dur_us) as gemm_us
from kernels join profiles using (job_id, method, rank)
where mod_idx is not null
group by all order by gemm_us desc
```

</div>
</div>

<style>
.stat { border: 1px solid #e4e3df; border-top: 3px solid var(--fzj-blue); border-radius: 6px; padding: 6px 4px; }
.stat b { display: block; font-size: 1.3rem; line-height: 1.2; white-space: nowrap; }
.stat span { font-size: 0.72rem; color: #52514e; }
.pipe { display: flex; align-items: center; justify-content: space-between; font-size: 0.78rem; }
.pipe span { font-family: var(--slidev-code-font-family, monospace); background: var(--fzj-gray); border-radius: 5px; padding: 3px 8px; text-align: center; line-height: 1.25; }
.pipe span.hl { background: color-mix(in srgb, var(--fzj-lightblue) 55%, white); color: var(--fzj-blue); font-weight: 700; }
.pipe small { font-family: var(--slidev-font-family, sans-serif); font-weight: 400; color: #52514e; font-size: 0.66rem; }
.pipe i { color: #8a8984; font-style: normal; }
</style>

<!--
Every profile we've taken is in one place. sync.sh pulls the reports and the job logs off jupiter with one rsync, so one TOTP prompt. nsys export turns each .nsys-rep into SQLite, and build.py projects the tables we care about (kernels, NVTX ranges, memcpys, MPI calls, GPU metrics) into parquet. 27 GB turns into 385 MB because almost all of a profile's size is OSRT_API, about 400k rows per profile, 90% of them `accept`, and nothing reads it.

The trick that makes cross-rank work possible: every nsys timestamp is relative to that process's own session start. We add TARGET_INFO_SESSION_START_TIME's UTC epoch to every one, so rank 0's kernels and rank 3's land on the same time axis. build.py also checks that the rank in the filename matches MPI_RANKS, and that every profile has the same export schema version.

The DuckDB session is disposable. Parquet + runs.csv are the durable artifacts; everything is rebuilt from them in seconds.
-->

---

# A question is one file

<div class="grid grid-cols-[1.15fr_1fr] gap-5">
<div>

```python {all|3-4|7-11|13-16}
from common import connect, show

JOB = "1852819"            # edit, re-run
METHOD = "par_ozaki"

con = connect()            # setup.sql: views over all 336 profiles
con.execute("""
    create or replace temp table stage_rows as
    select stage, rank, dur_ms from stages
    where job_id = $job and method = $method and not is_warmup
""", {"job": JOB, "method": METHOD})

show(con.sql("""
    pivot stage_rows on rank using round(avg(dur_ms), 3) as ms
    group by stage order by stage
""").df(), f"{JOB} / {METHOD}: stage ms per rank")
```

<div class="text-xs opacity-70 mt-1">questions/stage_by_rank.py, trimmed</div>

</div>
<div v-click="3" class="tight">

`uv run questions/stage_by_rank.py`

| stage | r0 | r1 | r2 | r3 |
|---|--:|--:|--:|--:|
| assemble | 2.235 | 2.215 | 2.182 | 2.228 |
| dsm+invscal+crt | 0.697 | 0.698 | 0.696 | 0.644 |
| modexpand | 0.640 | 0.633 | 0.643 | 0.641 |
| moduli GEMM+conv | 6.642 | 6.665 | <span class="text-[var(--fzj-red)] font-bold">7.331</span> | 6.700 |
| scaling | 2.693 | 2.687 | 2.662 | 2.694 |

<div v-click="4" class="mt-3 text-sm">

Rank 2 is 10% slower on the GEMM. Why? That's the next file:

<div class="chain">
  <code>stage_by_rank</code><i>→</i><code>kernel_clock</code><i>→</i><code>pinned-node rerun</code>
</div>

Three questions, two days, ending in "the slow GPU is hardware"

</div>
</div>
</div>

<style>
.tight table { font-size: 0.8rem; margin-top: 0.4rem; }
.tight th, .tight td { padding: 0.2rem 0.5rem; }
.tight td { font-variant-numeric: tabular-nums; }
.chain { display: flex; align-items: center; gap: 6px; margin: 6px 0; font-size: 0.8rem; }
.chain code { background: color-mix(in srgb, var(--fzj-lightblue) 45%, white); color: var(--fzj-blue); }
.chain i { color: #8a8984; font-style: normal; }
</style>

<!--
Each question is a Python file with its parameters as literals at the top. You don't pass CLI arguments: you edit the literal and re-run. A literal can be a list of ranks or two job ids to compare, which argparse can't really express.

All the logic is SQL against the views in setup.sql. DuckDB's PIVOT discovers the rank columns from the data, so a different-shaped run doesn't silently show empty columns.

The question gets committed, so when the next job lands it's one command to ask it again. That is the iteration speed: the expensive part, getting the data into shape, is done once.

This particular table started the GPU clock thread. Rank 2's GEMM was consistently slower. kernel_clock.py joined every GEMM launch to the GPU metrics samples inside it, which showed equal cycles at a lower clock. Then a pinned-node rerun showed which GPU is slow changes with the node. Those are the two clock slides later on.
-->

---

# Questions that steered the project

<div class="text-sm -mt-2">Twelve question files so far, each comparing along a different axis of the same tables</div>

<ReportGallery class="mt-2" />

<!--
Each card is real output, trimmed to the rows that mattered.

Hotspots: on the NCCL baseline, ncclDevKernel_SendRecv was 43% of the GPU time on rank 0, competing with the fused GEMM for SMs. That's what sent us to the copy engines.

Copy order: by putting each peer copy's start and end on the same time axis as the GEMMs, we could see each link runs one copy at a time, and A(1) was going before B(0), which GEMM 0 needs.

Kernel clock: the slow rank does the same number of cycles at a lower clock.

Timing vs NVTX: the driver's own stage timers and the NVTX ranges agree, so we can trust the NVTX numbers. NVTX sees every rank, whereas the driver only prints rank 0's breakdown.
-->

---

# Copy engine based collectives

The fused CUTLASS GEMM launches 132 blocks: **one per SM** on the GH200. Anything else on the SMs competes with it.

<div class="grid grid-cols-2 gap-8 mt-4">
<div>

**NCCL** `SendRecv`: a kernel

- Takes 24 SMs, pushing 24 GEMM blocks into a second wave: GEMM **3.1–3.8 ms → 4.8–5.1 ms**
- Can't make progress while the GEMMs fill every SM: an exchange that takes 1.05 ms alone took **4.78 ms**, with no overlap
- Even a plain send/recv with no reduction needs SMs

</div>
<div>

**Peer copy**: CUDA IPC + `cudaMemcpy2DAsync`

- Runs on the **copy engines**, so **zero SMs** are used
- Each rank maps its peers' buffers once over MPI, then:
  - **pulls** the A/B int8 planes straight into its assembled buffer, so one strided copy replaces recv + unpack
  - **pushes** each C tile to its owner as soon as that GEMM ends
- The ordering NCCL used to provide now comes from events + two host barriers

</div>
</div>

<!--
The moduli GEMM assumes it owns the device. NCCL's communication kernels are still kernels: 24 blocks of ncclDevKernel_SendRecv take SMs away, so 24 GEMM blocks run as a second wave. Worse, a communication kernel sitting on an SM makes no progress while the rank's own GEMM saturates the device, so the exchange only finished when the GEMMs did.

Peer copies are DMA on the copy engines, which don't care how busy the SMs are. Setup: cuMemGetAddressRange for each allocation's base, cudaIpcGetMemHandle / cudaIpcOpenMemHandle brokered over MPI, done once.

Pull for A/B: the reader knows when it needs the data. Push for C: the writer knows from a local event when GEMM j is done.

NCCL's send/recv pairing also gave us cross-rank ordering for free. A pull doesn't, so there are host barriers: one so no rank reads a plane before its peer has produced it, and one so no rank overwrites a plane while a peer is still reading it.

Behind PGEMM_PEER_COPY=1. It needs 4 ranks on one node with distinct P2P-capable GPUs, and falls back to NCCL otherwise.
-->

---
clicks: 3
---

# Copy engine based collectives: the timeline

<StageTimeline :step="$clicks" />

<!--
Real nsys data: stage 3 (assemble + moduli GEMM) on rank 2, rep 0 of the round-0 job of each step, all on jpbo-103-23. Rank 2 is the slowest GPU on that node, so its stage ends on its own last GEMM rather than waiting at a barrier for someone else. Regenerate with analysis/plots/stage3_export.py.

Top band: what runs on the SMs. Bottom band: the copy engines, one lane per peer link in, one per peer out. Each click is one commit; bars slide to where they ran in that step. The dashed lines are where the other steps' stages ended.

NCCL: the copy engines sit idle. The A/B exchange for GEMM 1 is launched at 0.3 ms but can't start until GEMM 0's blocks begin retiring at ~4.7 ms (hatched). It's a kernel and every SM is taken. So GEMM 1 starts at ~6.2 ms. The C exchanges after each GEMM are NCCL kernels too.

A/B planes on the copy engines: all of both GEMMs' inputs land by ~1.4 ms, on the copy engines, with no SMs used. GEMM 1 now starts as soon as GEMM 0 lets it. But each link runs one copy at a time, A0, A1, B0, B1: B0 is ready but stuck behind A1 (hatched), and GEMM 0 needs B0, not A1.

Copies reordered: hold A(j) until B(j-1) lands, so each link runs A0, B0, A1, B1. GEMM 0 starts ~330 µs earlier, and everything behind it moves with it.

C pushed: the C exchange leaves NCCL too. Each tile is pushed straight into its owner's buffer as soon as its GEMM ends, so GEMM 0's tiles go out while GEMM 1 is still running.

Hover any bar for its times. "zoom" shows the first 1.6 ms, where the copy reordering is easier to see.
-->

---

# Copy engine based collectives: results

`par_ozaki_async`, 12000³, 8 moduli. Every step rebuilt and rerun on **one node**, interleaved. Slowest rank, mean of 2 jobs.

| Step | Total (ms) | Δ (ms) | Assemble + GEMM (ms) |
|---|--:|--:|--:|
| NCCL | 14.89 | | 11.08 |
| **A/B planes on the copy engines** | **13.59** | **−1.30** | 9.80 |
| **Copies reordered: GEMM 0's inputs first** | **13.19** | **−0.40** | 9.45 |
| **C tiles pushed peer-to-peer** | **12.83** | **−0.36** | 8.97 |

- **14.9 → 12.8 ms (−14%)**, same accuracy (max rel. error 7.3 × 10⁻¹⁵)
- Every step helps, and no job of one step overlaps a job of the next
- Under NCCL each GEMM needs **23% more cycles**: the second wave, measured
- First attempt silently ran NCCL: `srun` gave each rank one GPU, so all four saw "device 0"

<!--
Jobs 2161528–2161535, all on jpbo-103-23 (scripts/peer_copy_campaign.sh). Four builds: HEAD with PGEMM_PEER_COPY=0, c0af880^, c0af880 and HEAD with it on. Nothing under src/ or include/ differs between them except the peer-copy commits, so each step is exactly one commit. Two rounds, the order rotated between them so drift on the node doesn't line up with a step.

Why rerun: the original jobs each landed on a different node, and which GPU is slow (13–15% per GEMM) depends on the node. That was bigger than every step but the first. On mixed nodes the first step looked like −2.0 ms and the reordering looked like nothing; on one node they're −1.3 and −0.4.

"Total" is the north star: per rep, sum the NVTX stage ranges on each rank, keep the slowest rank, average over the 2 timed reps. Per job: NCCL 14.87 / 14.91, A/B 13.57 / 13.62, reordered 13.26 / 13.12, C push 12.96 / 12.70. Two jobs per step is not a significance test, but the ranges don't overlap.

The reordering: each peer link runs one copy at a time, and it ran A0, A1, B0, B1. GEMM 0 needs B0 but not A1, so B0 waited ~350 us behind A1. Holding A(j) until B(j-1) lands: GEMM 0 starts ~330 us earlier (1140 → 815 us into the stage), and GEMM 1 ~270 us earlier.

The C push: the C_mid exchange still went over NCCL and all landed after the last GEMM. Pushing each tile straight into its owner's buffer, with no packing and no NCCL kernel, lets plane 0's pushes overlap GEMM 1.

Cycles: from the nsys GPU metrics, each GEMM launch takes 4.36–4.66 M cycles under NCCL against 3.53–3.63 M with peer copy. The GPUs actually clock higher under NCCL (~1140 vs ~1000 MHz on GPU 0) because the GEMM is less tensor-dense, which partly hides it. NCCL is also noisy: its two reps differ by 1.2–1.3 ms, against 0.01–0.4 ms with peer copy.

The fallback story (job 1703699): --gpu-bind=single:1 and --gpus-per-task=1 make srun set a per-rank CUDA_VISIBLE_DEVICES, so every rank reported device ordinal 0 and the peer-copy setup refused. Fixed with --gpu-bind=none.
-->

---

# The GPUs halve their clock under the GEMMs

Sampled with `nsys --gpu-metrics-devices`: ~1900 MHz on other kernels, **~900–1000 MHz** once the int8 tensor cores are saturated.

<img src="./plots/gpu_clock.png" alt="Left: GPC clock against time in a fused GEMM launch, four GPUs settling at 890 to 990 MHz against 1917 MHz on non-tensor kernels. Right: per GPU, GEMM time is 7 to 13 percent over GPU 0 while cycles are only about 2 percent over" class="mt-2 mx-auto h-[330px] bg-white rounded p-2" />

<!--
Five profiled jobs on one node, par_ozaki_async, 12000 cubed, 8 moduli.

Left: the clock drops within about 250 us of a GEMM launch and stays there for the whole 3.5 ms kernel. On the same GPUs, non-tensor kernels in the same reps run at about 1900 MHz. ref_par_ozaki's GEMMs, which keep the tensor cores less busy (82% vs 92%), hold about 1300 MHz: the denser the tensor work, the lower the clock.

So peak-TOPS comparisons at the boost clock understate our efficiency by about 2x.

Right: GEMM time differs by up to 13% between ranks, but time x clock -- cycles -- by only about 2%. The slow GPUs do the same work at a lower clock.

Caveat: nsys GPU metrics have no power counter, so "power capped" is an inference -- it fits (tensor-dense work, fast onset) but is not measured. To confirm: nvidia-smi power.draw and clocks_throttle_reasons during a run.
-->

---

# The slow GPU is hardware, not the code

Rank $r$ runs on GPU $r$ on every node, yet which GPU is slow changes with the node.

<img src="./plots/gpu_clock_nodes.png" alt="Mean GEMM clock per GPU on three nodes, 5 to 6 jobs each. jpbo-103-23: GPU 0 fastest, GPU 2 slowest. jpbo-007-32: GPU 1 fastest, tied with GPU 2; GPU 3 slowest. jpbo-014-42: GPU 3 fastest, GPU 2 slowest. Slowest GPU takes 13 to 15 percent longer per GEMM" class="mt-1 mx-auto h-[290px] bg-white rounded p-2" />

- Same fastest and slowest GPU in **all 16 jobs**; repeats scatter by <15 MHz against gaps of ~100–135 MHz
- Upper bound on the cost: ~7–9% of par_ozaki_async wall time, if every GPU ran as fast as the node's best

<!--
16 jobs over three nodes, 5 or 6 each, pinned with sbatch -w. If the slowness came from the rank's share of the work, the same rank would be slow everywhere. Instead the slowest GPU is 2 on jpbo-103-23, 3 on jpbo-007-32 and 2 on jpbo-014-42, and the fastest is 0, 1 and 3.

The fastest and slowest GPU of each node are the same in every job and in all three methods, ref included -- 48 out of 48. The only order swaps are near-ties: GPUs 1 and 2 on jpbo-007-32 (965 vs 966 MHz), GPUs 0 and 1 on jpbo-014-42. Job-to-job standard deviation per GPU is 3 to 12 MHz.

The slowest GPU on a node takes 13 to 15% longer per GEMM launch. The 7-9% is an upper bound: (slowest - fastest rank GEMM time) x 2 launches per rep, over the timing-pass wall time -- 0.9 ms of 12.8 on jpbo-103-23 (7%), 1.1 of 13.1 on jpbo-014-42 (8%), 1.2 of 13.2 on jpbo-007-32 (9%). It assumes the slowest rank holds the others up at the exchange, which I have not checked.

Small leftover: rank 0 needs 1-2.5% fewer cycles per GEMM than the other ranks in all 32 par_ozaki / par_ozaki_async runs, on whichever GPU it lands. That one is a rank effect, not yet explained.

Open: chip-to-chip variation vs the slot's cooling/power delivery -- the data can't separate these. Power is still not measured.
-->
