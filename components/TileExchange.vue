<script setup lang="ts">
import { computed, ref, watch } from 'vue'

// What every rank holds, stage by stage, for the two distributed Ozaki paths.
// Tile layout follows par_gemmul8 (include/dist/grid.hpp): rank r sits at
// prow = r % 2, pcol = r / 2 and starts with the (prow, pcol) block of A, B, C.

type Mode = 'ref' | 'par'
type Cell = null | 'fp' | 'res' | number[] // number[] = residue planes for these moduli
type Hold = Record<'A' | 'B' | 'C', Cell[]>

const props = withDefaults(defineProps<{
  mode?: Mode
  step?: number
  m?: number
  n?: number
  k?: number
  moduli?: number
}>(), { mode: 'par', m: 12000, n: 12000, k: 12000, moduli: 8 })

const mode = ref<Mode>(props.mode)
const step = ref(props.step ?? 0)
const focus = ref<number | null>(null)
const hover = ref('')

const NM = props.moduli
const RANKS = [0, 1, 2, 3]
const prow = (r: number) => r % 2
const pcol = (r: number) => r >> 1
const rowPeer = (r: number) => r ^ 2
const colPeer = (r: number) => r ^ 1

// dist::moduli_range: contiguous near-equal ranges, lower ranks take the remainder.
function ownModuli(r: number) {
  const base = Math.floor(NM / 4)
  const rem = NM % 4
  const begin = r * base + Math.min(r, rem)
  return Array.from({ length: base + (r < rem ? 1 : 0) }, (_, i) => begin + i)
}
const ALL = Array.from({ length: NM }, (_, i) => i)

// dist::tile_dims_2x2_for_rank
const half = (x: number, takesRem: boolean) => Math.floor(x / 2) + (takesRem ? x % 2 : 0)
function dims(r: number) {
  return {
    m: half(props.m, prow(r) === 0),
    n: half(props.n, pcol(r) === 1),
    kA: half(props.k, pcol(r) === 0),
    kB: half(props.k, prow(r) === 0),
  }
}

const STAGES: Record<Mode, { name: string, desc: string }[]> = {
  ref: [
    { name: 'Own tiles', desc: 'Each rank starts with one FP64 tile of A and one of B: the same (prow, pcol) block of each.' },
    { name: 'Swap with peers', desc: 'A tiles swap along grid rows and B tiles along grid columns, one NCCL send/recv each. Every rank now holds a block-row of A and a block-column of B.' },
    { name: 'Two Ozaki strips', desc: 'C(p,q) = A(p,0)·B(0,q) + A(p,1)·B(1,q). Each strip is a complete sequential Ozaki run over all N moduli. Diagonal ranks start the first strip on local data while the swap is in flight.' },
  ],
  par: [
    { name: 'Own tiles', desc: 'Same start: one FP64 tile of A and of B per rank.' },
    { name: 'Scaling', desc: 'Row maxima of A are max-reduced along grid rows, column maxima of B along grid columns. Only vectors move.' },
    { name: 'Moduli expand', desc: 'Each rank turns its own tiles into N int8 residue planes, one per modulus. No communication.' },
    { name: 'Assemble', desc: 'Moduli, not tiles, are shared out: each rank owns N/4 of them. Every rank sends each peer the planes for that peer\'s moduli, diagonals included.' },
    { name: 'Moduli GEMM', desc: 'Each rank multiplies the full A and B for its own moduli. Its C planes cover every tile, but only its moduli.' },
    { name: 'C exchange', desc: 'Each rank sends every peer that peer\'s C tile for its own moduli. Every rank now has its own C tile for all N moduli.' },
    { name: 'CRT + inverse scaling', desc: 'Local: rebuild each entry from its N residues with the CRT and undo the scaling. Out comes the FP64 C tile.' },
  ],
}
const stages = computed(() => STAGES[mode.value])
const s = computed(() => Math.min(step.value, stages.value.length - 1))

watch(() => props.step, (v) => { if (v !== undefined) step.value = v })
function setMode(m: Mode) {
  mode.value = m
  focus.value = null
}

function holdings(md: Mode, st: number, r: number): Hold {
  const h: Hold = { A: [null, null, null, null], B: [null, null, null, null], C: [null, null, null, null] }
  h.A[r] = 'fp'
  h.B[r] = 'fp'
  if (md === 'ref') {
    if (st >= 1) { h.A[rowPeer(r)] = 'fp'; h.B[colPeer(r)] = 'fp' }
    if (st >= 2) h.C[r] = 'res'
    return h
  }
  const own = ownModuli(r)
  if (st >= 2) { h.A[r] = ALL; h.B[r] = ALL }
  if (st >= 3) for (const t of RANKS) if (t !== r) { h.A[t] = own; h.B[t] = own }
  if (st >= 4) for (const t of RANKS) h.C[t] = own
  if (st >= 5) h.C[r] = ALL
  if (st >= 6) h.C[r] = 'res'
  return h
}
const hold = computed(() => RANKS.map(r => holdings(mode.value, s.value, r)))

interface Flow { from: number, to: number, label: string }
const flows = computed<Flow[]>(() => {
  const out: Flow[] = []
  for (const r of RANKS) {
    if (mode.value === 'ref' && s.value === 1) {
      out.push({ from: r, to: rowPeer(r), label: 'A' }, { from: r, to: colPeer(r), label: 'B' })
    }
    if (mode.value === 'par' && s.value === 1) {
      out.push({ from: r, to: rowPeer(r), label: 'max' }, { from: r, to: colPeer(r), label: 'max' })
    }
    if (mode.value === 'par' && (s.value === 3 || s.value === 5)) {
      for (const q of RANKS) if (q !== r) out.push({ from: r, to: q, label: s.value === 3 ? 'A B' : 'C' })
    }
  }
  return out.filter(f => focus.value === null || f.from === focus.value || f.to === focus.value)
})

// --- numbers for the sidebar -------------------------------------------------

const MB = (b: number) => b >= 1e6 ? `${Math.round(b / 1e6)} MB` : `${Math.max(1, Math.round(b / 1e3))} kB`

function sentBytes(md: Mode, st: number, r: number) {
  const d = dims(r)
  if (md === 'ref') return st === 1 ? (d.m * d.kA + d.kB * d.n) * 8 : 0
  if (st === 1) return (d.m + d.n) * 4 * 2 // A/B maxima plus the C-estimate maxima
  if (st === 3) {
    return RANKS.filter(q => q !== r)
      .reduce((a, q) => a + ownModuli(q).length * (d.m * d.kA + d.kB * d.n), 0)
  }
  if (st === 5) {
    return RANKS.filter(q => q !== r)
      .reduce((a, q) => a + ownModuli(r).length * dims(q).m * dims(q).n, 0)
  }
  return 0
}
function totalSent(md: Mode, r: number) {
  return STAGES[md].reduce((a, _, st) => a + sentBytes(md, st, r), 0)
}

const statRank = computed(() => focus.value ?? 0)
const stats = computed(() => {
  const r = statRank.value
  const d = dims(r)
  const lines: string[] = []
  const b = sentBytes(mode.value, s.value, r)
  if (b > 0) {
    const what = mode.value === 'ref' ? 'FP64 tiles' : s.value === 1 ? 'row/column maxima' : 'int8 planes'
    lines.push(`rank ${r} sends ${MB(b)} of ${what}`)
  }
  if (mode.value === 'ref' && s.value === 2)
    lines.push(`${2 * NM} int8 GEMMs of ${d.m}×${d.n}×${half(props.k, true)} per rank`)
  if (mode.value === 'par' && s.value === 4)
    lines.push(`${ownModuli(r).length} int8 GEMMs of ${props.m}×${props.n}×${props.k} on rank ${r}: the same int8 flops as ref, in fewer, bigger GEMMs`)
  if (s.value === stages.value.length - 1) {
    lines.push(`total sent by rank ${r}: ${MB(totalSent(mode.value, r))} (${mode.value === 'ref' ? 'par' : 'ref'}: ${MB(totalSent(mode.value === 'ref' ? 'par' : 'ref', r))})`)
  }
  return lines
})

// --- geometry ----------------------------------------------------------------

const PW = 226
const PH = 140
const GX = 112
const GY = 92
const W = 2 * PW + GX
const H = 2 * PH + GY
const CS = 28 // tile cell size
const GRID_X = [14, 84, 154] // A, B, C grid offsets inside a panel
const GRID_Y = 48

const panelX = (r: number) => pcol(r) * (PW + GX)
const panelY = (r: number) => prow(r) * (PH + GY)
const center = (r: number) => ({ x: panelX(r) + PW / 2, y: panelY(r) + PH / 2 })

function flowPath(f: Flow) {
  const a = center(f.from)
  const b = center(f.to)
  const dx = b.x - a.x
  const dy = b.y - a.y
  const len = Math.hypot(dx, dy)
  const ux = dx / len
  const uy = dy / len
  const exit = Math.min(dx ? (PW / 2) / Math.abs(dx) : Infinity, dy ? (PH / 2) / Math.abs(dy) : Infinity) * len + 6
  const off = 7 // keep the two directions apart
  const ox = -uy * off
  const oy = ux * off
  const x1 = a.x + ux * exit + ox
  const y1 = a.y + uy * exit + oy
  const x2 = b.x - ux * exit + ox
  const y2 = b.y - uy * exit + oy
  return { d: `M ${x1} ${y1} L ${x2} ${y2}`, x1, y1, x2, y2 }
}

const PACKET_S = 2.4

const reduceMotion =typeof window !== 'undefined'
  && window.matchMedia?.('(prefers-reduced-motion: reduce)').matches
const uid = Math.random().toString(36).slice(2, 8)

const TILE = (t: number) => `(${prow(t)},${pcol(t)})`

function modList(ms: number[]) {
  if (ms.length === NM) return `all ${NM} moduli`
  return ms.length === 1 ? `modulus ${ms[0]}` : `moduli ${ms[0]}–${ms[ms.length - 1]}`
}

function describe(r: number, X: 'A' | 'B' | 'C', t: number) {
  const v = hold.value[r][X][t]
  const name = `${X}${TILE(t)}`
  if (v === null) return `rank ${r} does not hold ${name}`
  if (v === 'res') return `rank ${r}: ${name}, the finished FP64 result`
  const from = t === r ? 'its own tile' : `rank ${t}'s tile`
  if (v === 'fp') return `rank ${r}: ${name} in FP64, ${from}`
  if (X === 'C') {
    return v.length === NM
      ? `rank ${r}: ${name} for all ${NM} moduli, its own moduli computed here and the rest received`
      : `rank ${r}: ${name} for ${modList(v)}, computed here`
  }
  return `rank ${r}: ${name} as int8 planes for ${modList(v)}, ${from}`
}

// A cell is "active" in the current compute stage if it feeds or is C(r).
function active(r: number, X: 'A' | 'B' | 'C', t: number) {
  if (mode.value === 'ref' && s.value === 2) {
    if (X === 'A') return prow(t) === prow(r)
    if (X === 'B') return pcol(t) === pcol(r)
    return t === r
  }
  return true
}
</script>

<template>
  <div class="tx">
    <div class="stage-area">
      <svg :viewBox="`-4 -4 ${W + 8} ${H + 8}`" class="grid-svg" role="img"
           :aria-label="`${mode === 'ref' ? 'ref_par_ozaki' : 'par_ozaki'}, stage ${s + 1}: ${stages[s].name}`">
        <defs>
          <marker :id="`arrow-${uid}`" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto">
            <path d="M0,0 L8,4 L0,8 z" class="arrow-head" />
          </marker>
        </defs>

        <!-- flows -->
        <g v-for="(f, i) in flows" :key="`${mode}-${s}-${focus}-${i}`" class="flow">
          <line v-bind="{ x1: flowPath(f).x1, y1: flowPath(f).y1, x2: flowPath(f).x2, y2: flowPath(f).y2 }"
                :marker-end="`url(#arrow-${uid})`" class="flow-line" />
          <g v-if="!reduceMotion" class="packet" :class="`r${f.from}`" opacity="0">
            <rect x="-13" y="-7" width="26" height="14" rx="3" />
            <text y="3.5" text-anchor="middle">{{ f.label }}</text>
            <!-- Ease along the arrow and fade at both ends, so the loop back to the start is never seen.
                 No begin offset: every packet runs on the SVG's one clock, so all of a stage's
                 transfers leave and land together, as in the single NCCL group they model. -->
            <animateMotion :path="flowPath(f).d" :dur="`${PACKET_S}s`" repeatCount="indefinite"
                           calcMode="spline" keyPoints="0;1" keyTimes="0;1" keySplines="0.45 0 0.55 1" />
            <animate attributeName="opacity" values="0;1;1;0" keyTimes="0;0.12;0.85;1" :dur="`${PACKET_S}s`"
                     repeatCount="indefinite" />
          </g>
        </g>

        <!-- rank panels -->
        <g v-for="r in RANKS" :key="r" :transform="`translate(${panelX(r)} ${panelY(r)})`"
           class="panel" :class="{ dim: focus !== null && focus !== r, focused: focus === r }"
           @click="focus = focus === r ? null : r">
          <rect :width="PW" :height="PH" rx="6" class="panel-bg" />
          <rect x="12" y="11" width="10" height="10" rx="2" :class="`sw r${r}`" />
          <text x="28" y="20" class="panel-title">rank {{ r }}</text>
          <text x="80" y="20" class="panel-sub">prow {{ prow(r) }}, pcol {{ pcol(r) }}</text>

          <g v-for="(X, gi) in (['A', 'B', 'C'] as const)" :key="X" :transform="`translate(${GRID_X[gi]} ${GRID_Y})`">
            <text :x="CS" y="-7" text-anchor="middle" class="grid-label">{{ X }}</text>
            <g v-for="t in RANKS" :key="t" :transform="`translate(${pcol(t) * CS} ${prow(t) * CS})`"
               class="cell" :class="{ inactive: !active(r, X, t) }"
               @mouseenter="hover = describe(r, X, t)" @mouseleave="hover = ''">
              <template v-if="hold[r][X][t] === null">
                <rect x="1" y="1" :width="CS - 2" :height="CS - 2" rx="2" class="empty" />
              </template>
              <template v-else-if="hold[r][X][t] === 'res'">
                <rect x="1" y="1" :width="CS - 2" :height="CS - 2" rx="2" :class="`solid r${t}`" />
                <text :x="CS / 2" :y="CS / 2 + 3.5" text-anchor="middle" class="res-mark">64</text>
              </template>
              <template v-else>
                <rect x="1" y="1" :width="CS - 2" :height="CS - 2" rx="2" :class="`tint r${t}`" />
                <text v-if="hold[r][X][t] === 'fp'" :x="CS / 2" :y="CS / 2 + 3" text-anchor="middle" class="fp-mark">f64</text>
                <g v-else>
                  <rect v-for="j in ALL" :key="j"
                        :x="4 + j * ((CS - 8) / NM)" :y="CS - 9" :width="(CS - 8) / NM - 0.8" height="5"
                        class="tick" :class="{ on: (hold[r][X][t] as number[]).includes(j) }" />
                </g>
              </template>
            </g>
          </g>
          <text :x="GRID_X[0] + 2 * CS + 5" :y="GRID_Y + CS + 4" text-anchor="middle" class="op">×</text>
          <text :x="GRID_X[1] + 2 * CS + 5" :y="GRID_Y + CS + 4" text-anchor="middle" class="op">→</text>

          <text x="12" :y="PH - 12" class="panel-foot">
            <template v-if="mode === 'par'">owns {{ modList(ownModuli(r)) }}</template>
            <template v-else>C tile {{ dims(r).m }} × {{ dims(r).n }}</template>
          </text>
        </g>
      </svg>
      <div class="legend">
        <span><svg width="14" height="14"><rect x="1" y="1" width="12" height="12" rx="2" class="tint r0" /></svg>FP64 tile</span>
        <span><svg width="14" height="14"><rect x="1" y="1" width="12" height="12" rx="2" class="tint r0" /><rect x="3" y="8" width="8" height="3" class="tick on" /></svg>int8 planes, one tick per modulus</span>
        <span><svg width="14" height="14"><rect x="1" y="1" width="12" height="12" rx="2" class="empty" /></svg>not held</span>
      </div>
      <p class="hover">{{ hover || 'Hover a tile for details; click a rank to isolate its traffic.' }}</p>
    </div>

    <div class="side">
      <div class="modes" role="group" aria-label="Implementation">
        <button :class="{ on: mode === 'ref' }" @click="setMode('ref')">ref_par_ozaki</button>
        <button :class="{ on: mode === 'par' }" @click="setMode('par')">par_ozaki</button>
      </div>

      <ol class="stages">
        <li v-for="(st, i) in stages" :key="st.name" :class="{ on: i === s, past: i < s }" @click="step = i">
          <span class="num">{{ i + 1 }}</span>{{ st.name }}
        </li>
      </ol>

      <p class="desc">{{ stages[s].desc }}</p>
      <ul v-if="stats.length" class="stats">
        <li v-for="l in stats" :key="l">{{ l }}</li>
      </ul>
      <p v-if="mode === 'par' && (s === 3 || s === 5)" class="note">
        par_ozaki_async pipelines this per modulus; with PGEMM_PEER_COPY it runs on the copy engines.
      </p>

    </div>
  </div>
</template>

<style scoped>
.tx {
  /* Rank colours match svg/input_output_tiles.svg. */
  --c0: #2563eb;
  --c1: #d97706;
  --c2: #059669;
  --c3: #7c3aed;
  --ink: #0b0b0b;
  --ink-2: #52514e;
  --ink-3: #8a8984;
  --line: #d6d5d0;
  --surface: #fcfcfb;
  --panel: #ffffff;
  --accent: #2a78d6;

  display: grid;
  grid-template-columns: 1fr 270px;
  gap: 18px;
  align-items: start;
  color: var(--ink);
  font-size: 13px;
  line-height: 1.35;
  text-align: left;
}
html.dark .tx {
  --c0: #60a5fa;
  --c1: #fbbf24;
  --c2: #34d399;
  --c3: #a78bfa;
  --ink: #ffffff;
  --ink-2: #c3c2b7;
  --ink-3: #8a8984;
  --line: #3a3a37;
  --surface: #1a1a19;
  --panel: #222221;
  --accent: #3987e5;
}

.grid-svg { width: 100%; height: auto; display: block; overflow: visible; }

.r0 { --c: var(--c0); }
.r1 { --c: var(--c1); }
.r2 { --c: var(--c2); }
.r3 { --c: var(--c3); }

.panel { cursor: pointer; transition: opacity 0.25s; }
.panel.dim { opacity: 0.3; }
.panel-bg { fill: var(--panel); stroke: var(--line); stroke-width: 1; }
.panel.focused .panel-bg { stroke: var(--ink-3); stroke-width: 1.5; }
.sw { fill: var(--c); }
.panel-title { font-size: 13px; font-weight: 700; fill: var(--ink); }
.panel-sub, .panel-foot { font-size: 10.5px; fill: var(--ink-2); }
.grid-label { font-size: 11px; font-weight: 700; fill: var(--ink-2); }
.op { font-size: 14px; fill: var(--ink-3); }

.cell { transition: opacity 0.25s; }
.cell.inactive { opacity: 0.25; }
.empty { fill: none; stroke: var(--line); stroke-dasharray: 3 2; }
.tint { fill: color-mix(in srgb, var(--c) 20%, var(--panel)); stroke: var(--c); stroke-width: 1.2; }
.solid { fill: var(--c); stroke: var(--panel); stroke-width: 1; }
.fp-mark { font-size: 8.5px; fill: var(--ink-2); font-family: var(--slidev-code-font-family, ui-monospace, monospace); }
.res-mark { font-size: 10px; font-weight: 700; fill: var(--panel); font-family: var(--slidev-code-font-family, ui-monospace, monospace); }
.tick { fill: color-mix(in srgb, var(--ink-3) 25%, transparent); }
.tick.on { fill: var(--ink); }

.flow-line { stroke: var(--ink-3); stroke-width: 1.2; }
.arrow-head { fill: var(--ink-3); }
.packet rect { fill: var(--c); stroke: var(--surface); stroke-width: 1.5; }
.packet text { font-size: 9px; font-weight: 700; fill: #fff; }
html.dark .packet text { fill: #111; }

.side { display: flex; flex-direction: column; gap: 6px; }
.modes { display: flex; border: 1px solid var(--line); border-radius: 5px; overflow: hidden; }
.modes button {
  flex: 1;
  padding: 4px 6px;
  border: none;
  background: var(--surface);
  color: var(--ink-2);
  font-size: 12px;
  font-family: var(--slidev-code-font-family, ui-monospace, monospace);
  cursor: pointer;
}
.modes button.on { background: var(--accent); color: #fff; }

.stages { list-style: none; margin: 0; padding: 0; }
.stages li {
  display: flex;
  align-items: center;
  gap: 8px;
  margin: 0;
  padding: 1px 6px;
  line-height: 1.5;
  border-radius: 4px;
  color: var(--ink-3);
  cursor: pointer;
}
.stages li.past { color: var(--ink-2); }
.stages li.on { color: var(--ink); font-weight: 700; background: color-mix(in srgb, var(--accent) 14%, transparent); }
.num {
  display: inline-grid;
  place-items: center;
  width: 17px;
  height: 17px;
  border-radius: 50%;
  border: 1px solid currentColor;
  font-size: 10px;
  line-height: 1;
}

.desc { margin: 0; color: var(--ink); font-size: 12.5px; line-height: 1.4; }
.stats { margin: 0; padding-left: 16px; color: var(--ink-2); font-size: 12px; line-height: 1.4; }
.stats li { margin: 0; }
.note { margin: 0; color: var(--ink-3); font-size: 11.5px; line-height: 1.35; }
.legend { display: flex; gap: 16px; margin-top: 8px; color: var(--ink-2); font-size: 11px; }
.legend span { display: flex; align-items: center; gap: 6px; }
.legend .tint { --c: var(--ink-3); }
.hover { margin: 2px 0 0; min-height: 1.4em; color: var(--ink-2); font-size: 11.5px; font-style: italic; }
</style>
