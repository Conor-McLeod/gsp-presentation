<script setup lang="ts">
// Real output from analysis/questions/*.py, trimmed to the rows that carried
// the finding. Job and parameters are in each card's `run` line.

type Card = {
  file: string
  axis: string
  q: string
  run: string
  head: string[]
  rows: (string | number)[][]
  hot?: [number, number][] // [row, col] cells to highlight
  took: string
}

const CARDS: Card[] = [
  {
    file: 'kernel_hotspots.py',
    axis: 'across kernels',
    q: 'Where does the GPU time go?',
    run: '1807553 · par_ozaki_async · rank 0 · rep 0',
    head: ['kernel', 'launches', 'ms', '%'],
    rows: [
      ['ncclDevKernel_SendRecv', 6, '8.38', '42.8'],
      ['fused GEMM, mod 0', 1, '4.64', '23.7'],
      ['fused GEMM, mod 1', 1, '3.35', '17.1'],
      ['everything else', 19, '3.21', '16.4'],
    ],
    hot: [[0, 0], [0, 3]],
    took: 'The exchange is a kernel fighting the GEMM for SMs → move it to the copy engines',
  },
  {
    file: 'stage3_copy_order.py',
    axis: 'across peers, in time',
    q: 'In what order do the peer copies really run?',
    run: '2160162 · par_ozaki_async · mean over ranks, reps, peers',
    head: ['j', 'plane', 'start µs', 'end µs'],
    rows: [
      [0, 'A', '151', '454'],
      [0, 'B', '472', '772'],
      [1, 'A', '789', '1097'],
      [1, 'B', '1113', '1407'],
    ],
    hot: [[1, 3]],
    took: 'One copy per link at a time, and GEMM 0 waits for B(0). A(1) used to run first and hold B(0) back ~345 µs; now ~22 µs',
  },
  {
    file: 'kernel_clock.py',
    axis: 'across ranks',
    q: 'Is the slow rank slow because of its GPU clock?',
    run: '2158780 · par_ozaki_async · fused GEMM launches',
    head: ['rank', 'ms vs fastest', 'clock MHz', 'cycles vs fastest'],
    rows: [
      [0, '+0.0%', 986, '+0.0%'],
      [1, '+6.1%', 947, '+1.9%'],
      [2, '+13.2%', 887, '+1.8%'],
      [3, '+10.0%', 912, '+1.7%'],
    ],
    hot: [[2, 1], [2, 2], [2, 3]],
    took: 'Same cycles at a lower clock: it is the GPU, not the code → pin nodes and repeat',
  },
  {
    file: 'timing_vs_nvtx.py',
    axis: 'across jobs',
    q: 'Do the driver\'s own timers agree with NVTX?',
    run: 'every job · rank 0 · reported − NVTX',
    head: ['stage', 'pairs', 'mean diff ms', 'max diff ms'],
    rows: [
      ['scaling', 53, '−0.006', '0.328'],
      ['modexpand', 53, '0.020', '0.044'],
      ['moduli GEMM+conv', 27, '0.142', '0.324'],
      ['dsm+invscal+crt', 53, '0.027', '0.114'],
    ],
    took: 'They agree, so NVTX can be the north star, and unlike the driver it sees every rank',
  },
]

const isHot = (c: Card, r: number, k: number) => c.hot?.some(([a, b]) => a === r && b === k)
</script>

<template>
  <div class="gallery">
    <div v-for="c in CARDS" :key="c.file" class="card">
      <div class="top">
        <code class="file">questions/{{ c.file }}</code>
        <span class="axis">{{ c.axis }}</span>
      </div>
      <div class="q">{{ c.q }}</div>
      <table>
        <thead><tr><th v-for="h in c.head" :key="h">{{ h }}</th></tr></thead>
        <tbody>
          <tr v-for="(row, r) in c.rows" :key="r">
            <td v-for="(v, k) in row" :key="k" :class="{ hot: isHot(c, r, k), num: typeof v === 'number' || /^[−+\d.]/.test(String(v)) }">{{ v }}</td>
          </tr>
        </tbody>
      </table>
      <div class="run">{{ c.run }}</div>
      <div class="took">{{ c.took }}</div>
    </div>
  </div>
</template>

<style scoped>
.gallery {
  --ink-2: #52514e;
  --ink-3: #8a8984;
  --line: #e4e3df;
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 8px 14px;
}
.card {
  border: 1px solid var(--line);
  border-top: 3px solid var(--fzj-blue);
  border-radius: 6px;
  padding: 5px 10px 6px;
  background: white;
}
.top { display: flex; justify-content: space-between; align-items: baseline; }
.file { font-size: 11px; color: var(--fzj-blue); background: none; padding: 0; }
.axis {
  font-size: 10px; text-transform: uppercase; letter-spacing: 0.04em;
  color: var(--fzj-blue); background: color-mix(in srgb, var(--fzj-lightblue) 40%, white);
  padding: 1px 6px; border-radius: 3px;
}
.q { font-weight: 700; font-size: 13px; margin: 1px 0 2px; }
table { width: 100%; border-collapse: collapse; font-size: 11px; margin: 0; }
th { text-align: left; font-weight: 400; color: var(--ink-3); border-bottom: 1px solid var(--line); padding: 1px 4px; }
td { padding: 0 4px; line-height: 1.35; border: none; font-family: var(--slidev-code-font-family, monospace); }
td.num { text-align: right; font-variant-numeric: tabular-nums; }
th:not(:first-child) { text-align: right; }
td.hot { color: var(--fzj-red); font-weight: 700; }
tr { border: none; }
.run { font-size: 10px; color: var(--ink-3); margin-top: 2px; }
.took { font-size: 11.5px; margin-top: 1px; line-height: 1.25; color: var(--ink-2); }
</style>
