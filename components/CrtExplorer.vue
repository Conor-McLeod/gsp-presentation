<script setup lang="ts">
import katex from 'katex'
import 'katex/dist/katex.min.css'
import { computed, onBeforeUnmount, ref, watch } from 'vue'

// The int8 moduli from par_gemmul8 (include/ozaki/crt_table_int8_data.hpp), in
// library order: N moduli means the first N of these.
const MODULI = [
  256, 255, 253, 251, 247, 241, 239, 233, 229, 227,
  223, 217, 211, 199, 197, 193, 191, 181, 179, 173,
].map(BigInt)

const props = withDefaults(defineProps<{ x?: string, n?: number }>(), {
  x: '123456789',
  n: 8,
})

const xText = ref(props.x)
const N = ref(props.n)

// --- BigInt helpers --------------------------------------------------------

const mod = (a: bigint, m: bigint) => ((a % m) + m) % m

// Signed residue, as in par_gemmul8: in [-p/2, p/2), so it fits in int8.
function rmod(a: bigint, p: bigint) {
  const r = mod(a, p)
  return 2n * r >= p ? r - p : r
}

function modinv(a: bigint, m: bigint) {
  let [r0, r1] = [mod(a, m), m]
  let [s0, s1] = [1n, 0n]
  while (r1 !== 0n) {
    const q = r0 / r1;
    [r0, r1] = [r1, r0 - q * r1];
    [s0, s1] = [s1, s0 - q * s1]
  }
  return mod(s0, m)
}

function log2Big(v: bigint) {
  if (v <= 0n) return 0
  const len = v.toString(2).length
  if (len <= 53) return Math.log2(Number(v))
  return len - 53 + Math.log2(Number(v >> BigInt(len - 53)))
}

// Accepts plain integers and 2^k, 2^k+c, -2^k-c.
function parseBig(s: string): bigint | null {
  const t = s.replace(/[\s_,]/g, '').replace('−', '-')
  const m = t.match(/^(-?)2\^(\d{1,3})([+-]\d+)?$/)
  try {
    if (m) {
      let v = 2n ** BigInt(m[2])
      if (m[3]) v += BigInt(m[3])
      return m[1] ? -v : v
    }
    if (/^-?\d{1,60}$/.test(t)) return BigInt(t)
  }
  catch {}
  return null
}

// Labels are rendered with KaTeX, same as the $…$ math in the slides.
const tex = (s: string) => katex.renderToString(s, { throwOnError: false })

// Plain text, for the small residues under the dials.
const fmt = (v: bigint) => v.toString().replace('-', '−')

// TeX: digits grouped in threes, or scientific notation past 19 digits.
function fmtTex(v: bigint) {
  const neg = v < 0n
  const s = (neg ? -v : v).toString()
  const sign = neg ? '-' : ''
  if (s.length <= 19) return sign + s.replace(/\B(?=(\d{3})+(?!\d))/g, '\\,')
  return `${sign}${s[0]}.${s.slice(1, 4)} \\times 10^{${s.length - 1}}`
}

// --- the CRT -----------------------------------------------------------------

const x = computed(() => parseBig(xText.value))
const active = computed(() => MODULI.slice(0, N.value))
const P = computed(() => active.value.reduce((a, b) => a * b, 1n))
// (P / p_i) * q_i, with q_i the inverse of P / p_i mod p_i. par_gemmul8: qPi.
const weights = computed(() => active.value.map((p) => {
  const Pi = P.value / p
  return Pi * modinv(Pi, p)
}))
const residues = computed(() => x.value === null ? [] : MODULI.map(p => rmod(x.value!, p)))

// How many terms of the sum have been added; N means done.
const step = ref(N.value)
const partial = computed(() => {
  let s = 0n
  for (let i = 0; i < Math.min(step.value, N.value); i++)
    s += weights.value[i] * residues.value[i]
  return mod(s, P.value)
})
const done = computed(() => step.value >= N.value)
// Map back into the symmetric range [-P/2, P/2).
const xhat = computed(() => 2n * partial.value >= P.value ? partial.value - P.value : partial.value)
const exact = computed(() => x.value !== null && xhat.value === x.value)
const wraps = computed(() => x.value === null ? 0n : (x.value - xhat.value) / P.value)
const wrapTex = computed(() => `\\hat{x} = x ${wraps.value > 0n ? '-' : '+'} ${fmtTex(wraps.value < 0n ? -wraps.value : wraps.value)}\\,P`)

let timer: ReturnType<typeof setInterval> | undefined
function stop() {
  if (timer) clearInterval(timer)
  timer = undefined
}
function replay() {
  stop()
  step.value = 0
  timer = setInterval(() => {
    step.value++
    if (step.value >= N.value) stop()
  }, 450)
}
watch([x, N], () => {
  stop()
  step.value = N.value
})
onBeforeUnmount(stop)

function randomX() {
  const bits = 24 + Math.floor(Math.random() * 120)
  let v = 0n
  for (let i = 0; i < bits; i++) v = (v << 1n) | BigInt(Math.random() < 0.5 ? 1 : 0)
  v |= 1n << BigInt(bits - 1)
  xText.value = (Math.random() < 0.3 ? '-' : '') + v.toString()
}

const presets = [
  { label: '123456789', v: '123456789' },
  { label: '2^{40}', v: '2^40' },
  { label: '\\text{int64 max}', v: '2^63-1' },
  { label: '-2^{100}', v: '-2^100' },
]

// --- geometry ----------------------------------------------------------------

const DIAL = 52
const DR = 19

function dialHand(i: number) {
  const p = Number(MODULI[i])
  const y = Number(residues.value[i] ?? 0n)
  const a = (y / p) * 2 * Math.PI
  return { x: DIAL / 2 + DR * 0.82 * Math.sin(a), y: DIAL / 2 - DR * 0.82 * Math.cos(a) }
}

const RING = 210
const RR = 82
// Position of v on the ring Z/P, 0 at the top and ±P/2 at the bottom.
function ringPoint(v: bigint) {
  const frac = Number((mod(v, P.value) * 1_000_000n) / P.value) / 1_000_000
  const a = frac * 2 * Math.PI
  return { x: RING / 2 + RR * Math.sin(a), y: RING / 2 - RR * Math.cos(a) }
}
const target = computed(() => ringPoint(x.value ?? 0n))
const dot = computed(() => ringPoint(partial.value))

// Capacity meter, in bits. 20 moduli give log2 P ≈ 157.
const METER_MAX = 160
const capBits = computed(() => log2Big(P.value) - 1) // |x| < P/2
const needBits = computed(() => x.value === null ? 0 : log2Big(x.value < 0n ? -x.value : x.value))
const pct = (b: number) => `${Math.min(100, (b / METER_MAX) * 100)}%`
</script>

<template>
  <div class="crt">
    <div class="left">
      <div class="controls">
        <label class="xin">
          <span class="k" v-html="tex('x =')" />
          <input v-model="xText" spellcheck="false" :class="{ bad: x === null }" @keydown.stop>
        </label>
        <div class="presets">
          <button v-for="p in presets" :key="p.v" @click="xText = p.v" v-html="tex(p.label)" />
          <button @click="randomX">random</button>
        </div>
      </div>

      <div class="controls">
        <label class="nin">
          <span class="k" v-html="tex(`N = ${N}`)" />
          <input v-model.number="N" type="range" min="2" max="20">
        </label>
        <button class="primary" :disabled="x === null" @click="replay">▶︎ reconstruct</button>
      </div>

      <div class="dials">
        <button
          v-for="(p, i) in MODULI" :key="i" class="dial"
          :class="{ off: i >= N, cur: !done && i === step - 1 }"
          :title="`use the first ${i + 1} moduli`"
          @click="N = Math.max(2, i + 1)"
        >
          <span class="p">{{ p }}</span>
          <svg :width="DIAL" :height="DIAL" aria-hidden="true">
            <circle :cx="DIAL / 2" :cy="DIAL / 2" :r="DR" class="face" />
            <line :x1="DIAL / 2" :y1="DIAL / 2 - DR - 3" :x2="DIAL / 2" :y2="DIAL / 2 - DR + 3" class="zero" />
            <line
              v-if="i < N && x !== null"
              :x1="DIAL / 2" :y1="DIAL / 2" :x2="dialHand(i).x" :y2="dialHand(i).y" class="hand"
            />
            <circle :cx="DIAL / 2" :cy="DIAL / 2" r="2.5" class="hub" />
          </svg>
          <span class="y">{{ i < N && x !== null ? fmt(residues[i]) : '·' }}</span>
        </button>
      </div>

      <div class="meter">
        <div class="meter-head">
          <span v-html="tex(`P/2 \\approx 2^{${capBits.toFixed(1)}}`)" />
          <span v-if="x !== null" v-html="tex(`|x| \\approx 2^{${needBits.toFixed(1)}}`)" />
        </div>
        <div class="track">
          <div class="fill" :style="{ width: pct(capBits) }" />
          <div v-if="x !== null" class="need" :class="exact ? 'ok' : 'ko'" :style="{ left: pct(needBits) }" />
        </div>
        <div class="ticks">
          <span v-for="b in [0, 32, 64, 96, 128, 160]" :key="b" :style="{ left: pct(b) }">{{ b }}</span>
        </div>
        <div class="axis-label">bits</div>
      </div>
    </div>

    <div class="right">
      <div class="ring-wrap" :style="{ width: `${RING}px`, height: `${RING}px` }">
        <svg :width="RING" :height="RING" role="img" aria-label="Running CRT sum on the ring of integers mod P">
          <circle :cx="RING / 2" :cy="RING / 2" :r="RR" class="track-ring" />
          <line :x1="RING / 2" :y1="RING / 2 - RR - 6" :x2="RING / 2" :y2="RING / 2 - RR + 6" class="zero" />
          <circle v-if="x !== null" :cx="target.x" :cy="target.y" r="9" class="target" />
          <circle v-if="x !== null" :cx="dot.x" :cy="dot.y" r="5.5" class="dot" />
        </svg>
        <!-- KaTeX can't render inside <svg>, so the ring's labels sit on top of it. -->
        <span class="lbl" :style="{ top: `${RING / 2 - RR - 24}px` }" v-html="tex('0')" />
        <span class="lbl" :style="{ top: `${RING / 2 + RR + 6}px` }" v-html="tex('\\pm P/2')" />
        <span class="lbl big" :style="{ top: `${RING / 2 - 20}px` }" v-html="tex('\\mathbb{Z}/P\\mathbb{Z}')" />
        <span class="lbl" :style="{ top: `${RING / 2 + 4}px` }">integers mod P</span>
      </div>

      <div class="legend">
        <span><svg width="20" height="20" aria-hidden="true"><circle cx="10" cy="10" r="8" class="target" /></svg>target <span v-html="tex('x')" /></span>
        <span><svg width="20" height="20" aria-hidden="true"><circle cx="10" cy="10" r="5.5" class="dot" /></svg>running sum <span v-html="tex('\\hat{x}')" /></span>
      </div>

      <div class="readout">
        <div v-html="tex(`P = ${fmtTex(P)}`)" />
        <div v-if="x !== null" v-html="tex(`\\hat{x} = \\textstyle\\sum_{i=1}^{${Math.min(step, N)}} \\frac{P}{p_i} q_i y_i \\bmod P`)" />
        <div v-if="x !== null" class="xhat" v-html="tex(`\\phantom{\\hat{x}} = ${fmtTex(xhat)}`)" />
      </div>

      <div v-if="x === null" class="verdict ko">✗ not an integer</div>
      <div v-else-if="!done" class="verdict pending">summing term {{ step }} of {{ N }}…</div>
      <div v-else-if="exact" class="verdict ok">✓ <span v-html="tex('\\hat{x} = x')" />: recovered exactly</div>
      <div v-else class="verdict ko">✗ <span v-html="tex(wrapTex)" />: wrapped</div>
    </div>
  </div>
</template>

<style scoped>
.crt {
  --ink: #0b0b0b;
  --ink-2: #52514e;
  --ink-3: #8a8984;
  --line: #d6d5d0;
  --surface: #fcfcfb;
  --accent: #2a78d6;
  --good: #0ca30c;
  --good-text: #006300;
  --bad: #d03b3b;

  display: grid;
  grid-template-columns: 1fr 230px;
  gap: 20px;
  color: var(--ink);
  font-size: 13px;
  line-height: 1.3;
  text-align: left;
}
html.dark .crt {
  --ink: #ffffff;
  --ink-2: #c3c2b7;
  --ink-3: #8a8984;
  --line: #3a3a37;
  --surface: #1a1a19;
  --accent: #3987e5;
  --good-text: #3fc43f;
  --bad: #e25555;
}

.k { color: var(--ink-2); font-weight: 600; }
.dim { color: var(--ink-3); }
input { font-family: var(--slidev-code-font-family, ui-monospace, monospace); }

.controls {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 10px;
}
.xin { display: flex; align-items: center; gap: 6px; flex: 1; }
.xin input {
  flex: 1;
  min-width: 0;
  padding: 3px 8px;
  border: 1px solid var(--line);
  border-radius: 4px;
  background: var(--surface);
  color: var(--ink);
}
.xin input.bad { border-color: var(--bad); }
.presets { display: flex; gap: 4px; }
button {
  padding: 3px 8px;
  border: 1px solid var(--line);
  border-radius: 4px;
  background: var(--surface);
  color: var(--ink-2);
  font-size: 12px;
  cursor: pointer;
}
button:hover { color: var(--ink); border-color: var(--ink-3); }
button.primary { background: var(--accent); border-color: var(--accent); color: #fff; }
button:disabled { opacity: 0.4; cursor: default; }
.nin { display: flex; align-items: center; gap: 8px; flex: 1; }
.nin .k { min-width: 48px; }
.nin input { flex: 1; accent-color: var(--accent); }

.dials {
  display: grid;
  grid-template-columns: repeat(10, 1fr);
  gap: 2px 0;
}
.dial {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 2px 0;
  border: 1px solid transparent;
  background: none;
  font-size: 11px;
}
.dial .p { color: var(--ink-2); font-weight: 600; }
.dial .y { color: var(--ink); font-family: var(--slidev-code-font-family, ui-monospace, monospace); min-height: 1.3em; }
.dial.off { opacity: 0.3; }
.dial.cur { border-color: var(--accent); }
.face { fill: none; stroke: var(--line); stroke-width: 1.5; }
.zero { stroke: var(--ink-3); stroke-width: 1.5; }
.hand { stroke: var(--accent); stroke-width: 2; stroke-linecap: round; }
.hub { fill: var(--accent); }

.meter { margin-top: 12px; }
.meter-head { display: flex; justify-content: space-between; color: var(--ink-2); margin-bottom: 4px; }
.track {
  position: relative;
  height: 10px;
  border-radius: 4px;
  background: color-mix(in srgb, var(--line) 50%, transparent);
}
.fill {
  height: 100%;
  border-radius: 4px;
  background: var(--accent);
  transition: width 0.3s;
}
.need {
  position: absolute;
  top: -4px;
  width: 3px;
  height: 18px;
  margin-left: -1.5px;
  border-radius: 2px;
  outline: 2px solid var(--surface);
  transition: left 0.3s;
}
.need.ok { background: var(--good); }
.need.ko { background: var(--bad); }
.ticks { position: relative; height: 14px; color: var(--ink-3); font-size: 10px; }
.ticks span { position: absolute; transform: translateX(-50%); top: 2px; }
.axis-label { color: var(--ink-3); font-size: 10px; text-align: right; }

.right { display: flex; flex-direction: column; align-items: center; gap: 6px; }
.track-ring { fill: none; stroke: var(--line); stroke-width: 2; }
.ring-wrap { position: relative; }
.lbl { position: absolute; left: 50%; transform: translateX(-50%); white-space: nowrap; color: var(--ink-3); font-size: 11px; }
.lbl.big { font-size: 15px; color: var(--ink-2); }
.legend { display: flex; gap: 12px; color: var(--ink-2); font-size: 11px; }
.legend > span { display: flex; align-items: center; gap: 3px; }
.target { fill: none; stroke: var(--ink-2); stroke-width: 1.5; stroke-dasharray: 3 2; }
.dot {
  fill: var(--accent);
  stroke: var(--surface);
  stroke-width: 2;
  transition: cx 0.35s, cy 0.35s;
}
.readout { width: 100%; color: var(--ink); }
.readout .xhat { overflow-x: auto; }
.verdict {
  width: 100%;
  padding: 5px 8px;
  border-radius: 4px;
  font-weight: 600;
  border: 1px solid currentColor;
}
.verdict.ok { color: var(--good-text); }
.verdict.ko { color: var(--bad); }
.verdict.pending { color: var(--ink-2); }
</style>
