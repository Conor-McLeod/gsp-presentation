<script setup lang="ts">
import { computed, onBeforeUnmount, ref, watch } from 'vue'

// Ozaki scheme II on a small matrix, step by step. A port of the accurate-mode
// pipeline in ozaki-numpy (ozaki2/scaling.py, moduli.py, inverse_scaling.py),
// which in turn mirrors par_gemmul8's seq::ozaki_gemm. The CRT here is done in
// exact BigInt arithmetic rather than par_gemmul8's double-double sums.

const props = withDefaults(defineProps<{ step?: number, n?: number, seed?: number }>(), {
  n: 8,
  seed: 2003,
})

const MODULI = [
  256, 255, 253, 251, 247, 241, 239, 233, 229, 227,
  223, 217, 211, 199, 197, 193, 191, 181, 179, 173,
]
const M = 4
const K = 4
const NC = 3

const N = ref(props.n)
const seed = ref(props.seed)
const step = ref(props.step ?? 0)
watch(() => props.step, (v) => { if (v !== undefined) step.value = v })

// --- inputs --------------------------------------------------------------------

function mulberry32(a: number) {
  return () => {
    a |= 0
    a = (a + 0x6D2B79F5) | 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}
// Like the notebook: magnitudes 10^U(-3, 3), random signs.
function randMat(rows: number, cols: number, rnd: () => number) {
  return Array.from({ length: rows }, () => Array.from({ length: cols },
    () => 10 ** (rnd() * 6 - 3) * (rnd() < 0.5 ? -1 : 1)))
}
const inputs = computed(() => {
  const rnd = mulberry32(seed.value)
  return { A: randMat(M, K, rnd), B: randMat(K, NC, rnd) }
})

// --- exact float helpers -------------------------------------------------------

const dv = new DataView(new ArrayBuffer(8))
function ilogb(x: number) { // floor(log2|x|) for normal x
  dv.setFloat64(0, x)
  return ((dv.getUint32(0) >>> 20) & 0x7FF) - 1023
}
interface Ex { m: bigint, e: number } // m * 2^e, exactly
function toExact(x: number): Ex {
  if (x === 0) return { m: 0n, e: 0 }
  dv.setFloat64(0, x)
  const hi = dv.getUint32(0)
  const lo = dv.getUint32(4)
  const ex = (hi >>> 20) & 0x7FF
  let m = (BigInt(hi & 0xFFFFF) << 32n) | BigInt(lo)
  let e = -1074
  if (ex !== 0) { m |= 1n << 52n; e = ex - 1075 }
  return { m: hi >>> 31 ? -m : m, e }
}
const exAdd = (a: Ex, b: Ex): Ex => {
  const e = Math.min(a.e, b.e)
  return { m: (a.m << BigInt(a.e - e)) + (b.m << BigInt(b.e - e)), e }
}
const exMul = (a: Ex, b: Ex): Ex => ({ m: a.m * b.m, e: a.e + b.e })
const bitlen = (v: bigint) => (v < 0n ? -v : v).toString(2).length
function exToNumber(a: Ex) {
  let { m, e } = a
  const L = bitlen(m)
  if (L > 60) { m >>= BigInt(L - 60); e += L - 60 }
  return Number(m) * 2 ** e
}
function log2Big(v: bigint) {
  const L = bitlen(v)
  if (L <= 53) return Math.log2(Number(v))
  return L - 53 + Math.log2(Number(v >> BigInt(L - 53)))
}

// --- the pipeline -----------------------------------------------------------------

// crt_tables.log2P: (log2(𝒫 - 1)/2 - 0.5), rounded down to float32. The N = 2
// entry is par_gemmul8's vendored value.
function log2P(n: number) {
  if (n === 2) return 0x1DFD1EC / 2 ** 22
  const P = MODULI.slice(0, n).reduce((a, p) => a * BigInt(p), 1n)
  const exact = log2Big(P - 1n) / 2 - 0.5
  let f = Math.fround(exact)
  if (f > exact) f = Math.fround(f - 2 ** (Math.floor(Math.log2(f)) - 23))
  return f
}

// rmod(x, p) into [-p/2, p/2], then the int8 cast (+128 wraps to -128 for p = 256).
function rmod(x: bigint, p: number) {
  const P = BigInt(p)
  let r = x % P
  if (2n * r > P) r -= P
  if (2n * r < -P) r += P
  const v = Number(r)
  return v === 128 ? -128 : v
}

const calc = computed(() => {
  const { A, B } = inputs.value
  const n = N.value
  const mods = MODULI.slice(0, n)

  // Phase A: first-pass shifts so each row/col max lands in [32, 64), and int8 upper bounds.
  const shiftA0 = A.map(row => 5 - ilogb(Math.max(...row.map(Math.abs))))
  const shiftB0 = Array.from({ length: NC }, (_, j) => 5 - ilogb(Math.max(...B.map(r => Math.abs(r[j])))))
  const ub = (x: number, s: number) => x === 0 ? 0 : Math.max(1, Math.ceil(Math.abs(x) * 2 ** s))
  const Abar = A.map((row, i) => row.map(x => ub(x, shiftA0[i])))
  const Bbar = B.map(row => row.map((x, j) => ub(x, shiftB0[j])))
  // Phase B: one int8 GEMM bounds the product.
  const Cbar = Abar.map(row => Array.from({ length: NC }, (_, j) => row.reduce((a, x, h) => a + x * Bbar[h][j], 0)))
  // Phase C: spend the per-factor bit budget log2P.
  const l2P = log2P(n)
  const coef = -(0.5 + 3 * 2 ** -23)
  const extra = (amax: number) => amax === 0 ? 0 : Math.floor(l2P + coef * Math.fround(Math.log2(Math.fround(amax))))
  const sA = shiftA0.map((s0, i) => s0 + extra(Math.max(...Cbar[i]))) // A' = trunc(A * 2^sA)
  const sB = shiftB0.map((s0, j) => s0 + extra(Math.max(...Cbar.map(r => r[j]))))
  // Phase D: truncate. Integer-valued doubles, so the BigInt conversion is exact.
  const Ac = A.map((row, i) => row.map(x => BigInt(Math.trunc(x * 2 ** sA[i]))))
  const Bc = B.map(row => row.map((x, j) => BigInt(Math.trunc(x * 2 ** sB[j]))))

  // Phase E: one int8 slice per modulus.
  const Alo = mods.map(p => Ac.map(row => row.map(x => rmod(x, p))))
  const Blo = mods.map(p => Bc.map(row => row.map(x => rmod(x, p))))
  // Moduli GEMM loop: int8 x int8 -> int32, then back to a residue.
  const Cmid = mods.map((p, t) => Alo[t].map(row => Array.from({ length: NC }, (_, j) =>
    rmod(BigInt(row.reduce((a, x, h) => a + x * Blo[t][h][j], 0)), p))))

  // Direct CRT reconstruction, as in the paper and par_gemmul8's inverse_scaling
  // (exact here; the kernel uses double-double): S = sum_i w_i * U_i with weights
  // w_i = (P / p_i) * q_i, then one correction C'' = S - P * round(S / P).
  // S[k - 1] is the running sum after k terms, for the animation.
  const P = mods.reduce((a, p) => a * BigInt(p), 1n)
  const w = mods.map((p) => {
    const Pi = P / BigInt(p)
    let q = 1n
    while ((Pi * q) % BigInt(p) !== 1n) q++
    return Pi * q
  })
  const S: bigint[][][] = []
  mods.forEach((_, t) => {
    S.push(Array.from({ length: M }, (_, i) => Array.from({ length: NC }, (_, j) =>
      (t ? S[t - 1][i][j] : 0n) + w[t] * BigInt(Cmid[t][i][j]))))
  })
  const Q = S[n - 1].map(row => row.map((v) => {
    let r = ((v % P) + P) % P
    if (2n * r > P) r -= P
    return (v - r) / P // round(S / P)
  }))
  const Cpp = S[n - 1].map((row, i) => row.map((v, j) => v - Q[i][j] * P))
  const exactProd = Ac.map(row => Array.from({ length: NC }, (_, j) => row.reduce((a, x, h) => a + x * Bc[h][j], 0n)))
  const crtOk = Cpp.every((row, i) => row.every((v, j) => v === exactProd[i][j]))

  // Undo the scaling.
  const C = Cpp.map((row, i) => row.map((v, j) => Number(v) * 2 ** (-sA[i] - sB[j])))

  // Error against the exact product of the FP64 inputs.
  const ref = A.map(row => Array.from({ length: NC }, (_, j) =>
    row.reduce<Ex>((a, x, h) => exAdd(a, exMul(toExact(x), toExact(B[h][j]))), { m: 0n, e: 0 })))
  const refMax = Math.max(...ref.flat().map(r => Math.abs(exToNumber(r))))
  const relErr = (X: number[][]) => Math.max(...X.flatMap((row, i) => row.map((x, j) => {
    const d = exAdd(toExact(x), { m: -ref[i][j].m, e: ref[i][j].e })
    return Math.abs(exToNumber(d))
  }))) / refMax
  const fp64 = A.map(row => Array.from({ length: NC }, (_, j) => row.reduce((a, x, h) => a + x * B[h][j], 0)))

  return { mods, l2P, sA, sB, Ac, Bc, Alo, Blo, Cmid, P, w, S, Q, Cpp, crtOk, C, err: relErr(C), errFp64: relErr(fp64) }
})

// --- presentation state ---------------------------------------------------------

const STEPS = [
  { name: 'FP64 inputs', text: 'Entries span six orders of magnitude.' },
  { name: 'Scale & truncate', text: 'Scale each row of A and column of B by a power of two, then truncate to integers: A′ = trunc(diag(μ)·A), B′ = trunc(B·diag(ν)). The powers are as large as possible while keeping every |A′B′| < 𝒫/2. Truncation is the only rounding step.' },
  { name: 'Slice', text: 'For each modulus, A′ᵢ = A′ mod pᵢ (signed). N integer matrices become N int8 slices each.' },
  { name: 'N int8 GEMMs', text: 'Cᵢ = A′ᵢ·B′ᵢ mod pᵢ, one int8 tensor-core GEMM per modulus. They are independent: this is the work par_gemmul8 distributes.' },
  { name: 'CRT', text: 'Direct CRT reconstruction: weight each slice by wᵢ = (𝒫/pᵢ)·qᵢ, which is ≡ 1 mod pᵢ and ≡ 0 mod every other modulus, and add. The N terms are independent, one fma each per entry. The sum S is far outside [−𝒫/2, 𝒫/2); one correction S − 𝒫·round(S/𝒫) brings it back, and there it equals A′B′ exactly.' },
  { name: 'Unscale', text: 'Undo the powers of two: C = diag(μ)^{−1}·C″·diag(ν)^{−1}.' },
]
const s = computed(() => Math.min(Math.max(step.value, 0), STEPS.length - 1))

const sel = ref(0) // which slice is in front
const pinned = ref(false)
// CRT step: how many weighted terms are in the sum; N + 1 means the final
// S - P * round(S / P) correction has been applied.
const merged = ref(1)
let timer: ReturnType<typeof setInterval> | undefined
function stopTimer() {
  if (timer) clearInterval(timer)
  timer = undefined
}
function replayCrt() {
  stopTimer()
  merged.value = 1
  // About six seconds for the whole stack, whatever N is.
  const ms = Math.min(900, Math.max(350, 6000 / (N.value + 1)))
  timer = setInterval(() => {
    if (merged.value <= N.value) merged.value++
    else stopTimer()
  }, ms)
}
watch([s, N], () => {
  stopTimer()
  pinned.value = false
  sel.value = Math.min(sel.value, N.value - 1)
  merged.value = N.value + 1
  if (s.value === 3) {
    sel.value = 0
    timer = setInterval(() => { if (!pinned.value) sel.value = (sel.value + 1) % N.value }, 900)
  }
  if (s.value === 4) {
    sel.value = 0
    replayCrt()
  }
}, { immediate: true })
onBeforeUnmount(stopTimer)
function pick(t: number) {
  sel.value = t
  pinned.value = true
}

const hover = ref('')

// --- formatting ---------------------------------------------------------------

// Unicode superscript digits come from two blocks (¹²³ vs ⁴–⁹) and most fonts
// set them at different heights, so exponents are marked ^{…} in strings and
// rendered as real <sup> elements by rich().
const sup = (v: number | string) => `^{${String(v).replace(/-/g, '−')}}`
function rich(text: string) {
  return text.split(/(\^\{[^}]*\})/).filter(Boolean)
    .map(t => t.startsWith('^{') ? { t: t.slice(2, -1), sup: true } : { t, sup: false })
}
const minus = (t: string) => t.replace(/-/g, '−')
function fmtFp(x: number) {
  const [m, e] = x.toExponential(2).split('e')
  return minus(`${m}e${Number(e)}`)
}
function fmtInt(v: bigint) {
  const t = v.toString()
  const neg = t.startsWith('-')
  const d = neg ? t.slice(1) : t
  if (t.length <= 7) return minus(t)
  // Rounded for display only; hover shows the exact value.
  const r = Math.round(Number(d.slice(0, 3)) / 10)
  const [m, e] = r >= 100 ? ['1.0', d.length] : [`${String(r)[0]}.${String(r)[1]}`, d.length - 1]
  return minus(`${neg ? '-' : ''}${m}e${e}`)
}
const fmtErr = (x: number) => x === 0 ? '0' : minus(x.toExponential(1).replace('e', 'e').replace('+', ''))

// Sequential tint by magnitude: FP64 by decade, integers by share of their bit budget.
const tint = (f: number) => ({ background: `color-mix(in srgb, var(--seq) ${Math.round(6 + 44 * Math.min(1, Math.max(0, f)))}%, var(--surface))` })
const tintFp = (x: number) => tint((Math.log10(Math.abs(x)) + 3) / 6)
const tintBits = (v: bigint, budget: number) => tint(v === 0n ? 0 : bitlen(v) / budget)

// --- geometry for the slice stacks ----------------------------------------------

const D = computed(() => Math.min(5, 44 / Math.max(1, N.value - 1)))
const STACK_PAD = 44
// The selected slice sits at the front and the rest follow it round in modulus
// order, like cards cycled through a deck. Lifting a back slice to the top in
// place would cover the slices that are meant to be in front of it.
function planeStyle(t: number, expanded: boolean) {
  const pos = (t - sel.value + N.value) % N.value
  const off = expanded ? pos * D.value : 0
  return {
    transform: `translate(${off}px, ${-off}px)`,
    zIndex: N.value - pos,
  }
}
// CRT step: the first `merged` weighted slices have been added into the front
// card and the rest of the stack slides up behind it.
function cPlaneStyle(t: number) {
  if (s.value !== 4) return planeStyle(t, cSliced.value)
  const pos = Math.max(0, t - merged.value + 1)
  return { transform: `translate(${pos * D.value}px, ${-pos * D.value}px)`, zIndex: N.value - t }
}
const merging = computed(() => s.value === 4)
const terms = computed(() => Math.min(merged.value, N.value))
const corrected = computed(() => merged.value > N.value)
// What the front card shows: the running sum, or C'' once corrected.
const cur = computed(() => corrected.value ? calc.value.Cpp : calc.value.S[terms.value - 1])
const maxBits = (X: bigint[][]) => Math.max(...X.flat().map(v => log2Big(v < 0n ? -v : v)))
const aSliced = computed(() => s.value >= 2)
const cSliced = computed(() => s.value === 3)

// --- hover text ---------------------------------------------------------------

function infoA(i: number, h: number) {
  const c = calc.value
  const x = inputs.value.A[i][h]
  if (s.value === 0) return `A[${i},${h}] = ${x}`
  if (s.value === 1) return `A′[${i},${h}] = trunc(${fmtFp(x)} × 2${sup(c.sA[i])}) = ${c.Ac[i][h]}`
  return `A′${sub(sel.value)}[${i},${h}] = ${c.Ac[i][h]} mod ${c.mods[sel.value]} = ${c.Alo[sel.value][i][h]}`
}
function infoB(h: number, j: number) {
  const c = calc.value
  const x = inputs.value.B[h][j]
  if (s.value === 0) return `B[${h},${j}] = ${x}`
  if (s.value === 1) return `B′[${h},${j}] = trunc(${fmtFp(x)} × 2${sup(c.sB[j])}) = ${c.Bc[h][j]}`
  return `B′${sub(sel.value)}[${h},${j}] = ${c.Bc[h][j]} mod ${c.mods[sel.value]} = ${c.Blo[sel.value][h][j]}`
}
function infoC(i: number, j: number) {
  const c = calc.value
  if (s.value === 3) {
    const t = sel.value
    return `C${sub(t)}[${i},${j}] = row ${i} of A′${sub(t)} · column ${j} of B′${sub(t)}, mod ${c.mods[t]} = ${c.Cmid[t][i][j]}`
  }
  if (s.value === 4) {
    const k = terms.value
    const sum = c.Cmid.slice(0, k).map((m, t) => `w${sub(t)}·${m[i][j]}`).join(' + ')
    if (!corrected.value) return `S[${i},${j}] = ${sum} = ${fmtInt(c.S[k - 1][i][j])}`
    return `S[${i},${j}] = ${fmtInt(c.S[N.value - 1][i][j])}, round(S/𝒫) = ${c.Q[i][j]}, C″ = S − ${c.Q[i][j]}·𝒫 = ${c.Cpp[i][j]}`
  }
  if (s.value === 5) return `C[${i},${j}] = C″ × 2${sup(-c.sA[i] - c.sB[j])} = ${c.C[i][j]}`
  return ''
}
const SUB: Record<string, string> = { '0': '₀', '1': '₁', '2': '₂', '3': '₃', '4': '₄', '5': '₅', '6': '₆', '7': '₇', '8': '₈', '9': '₉' }
const sub = (v: number) => String(v + 1).split('').map(c => SUB[c]).join('')

const caption = computed(() => {
  const c = calc.value
  if (s.value === 4) {
    if (!corrected.value) return `${terms.value} of ${N.value} terms: |S| ≈ 2${sup(maxBits(cur.value).toFixed(0))}, but 𝒫/2 ≈ 2${sup((log2Big(c.P) - 1).toFixed(0))}`
    return c.crtOk ? '✓ S − 𝒫·round(S/𝒫) = A′B′ exactly' : '✗ CRT mismatch'
  }
  if (s.value === 5) return `max relative error ${fmtErr(c.err)} with N = ${N.value} (plain FP64 A·B: ${fmtErr(c.errFp64)})`
  if (s.value === 1) return `per-factor budget: ${c.l2P.toFixed(1)} bits with N = ${N.value}`
  return ''
})
</script>

<template>
  <div class="oz">
    <div class="figure" @mouseleave="hover = ''">
      <!-- A -->
      <div class="mat">
        <div class="label">{{ s === 0 ? 'A' : s === 1 ? 'A′' : 'A′ᵢ' }}<span class="dim"> FP64 → int</span></div>
        <div class="with-shifts">
          <div class="shifts rows" :class="{ show: s === 1 }">
            <span v-for="(v, i) in calc.sA" :key="i">×2<sup>{{ String(v).replace('-', '−') }}</sup></span>
          </div>
          <div class="stack" :style="{ paddingTop: `${STACK_PAD}px`, paddingRight: `${STACK_PAD}px` }">
            <div v-for="t in N" :key="t" class="plane" :class="{ front: t - 1 === sel || !aSliced }"
                 :style="[planeStyle(t - 1, aSliced), { gridTemplateColumns: `repeat(${K}, var(--cw))` }]">
              <template v-if="t - 1 === sel || (!aSliced && t === 1)">
                <template v-for="(row, i) in inputs.A" :key="i">
                  <div v-for="(x, h) in row" :key="h" class="cell"
                       :style="s === 0 ? tintFp(x) : s === 1 ? tintBits(calc.Ac[i][h], calc.l2P + 1) : {}"
                       @mouseenter="hover = infoA(i, h)">
                    {{ s === 0 ? fmtFp(x) : s === 1 ? fmtInt(calc.Ac[i][h]) : calc.Alo[sel][i][h] }}
                  </div>
                </template>
              </template>
              <span v-if="aSliced && t - 1 === sel" class="tag">mod {{ calc.mods[sel] }}</span>
            </div>
          </div>
        </div>
      </div>

      <div class="op">×</div>

      <!-- B -->
      <div class="mat">
        <div class="label">{{ s === 0 ? 'B' : s === 1 ? 'B′' : 'B′ᵢ' }}</div>
        <div class="shifts cols" :class="{ show: s === 1 }" :style="{ gridTemplateColumns: `repeat(${NC}, var(--cw))` }">
          <span v-for="(v, j) in calc.sB" :key="j">×2<sup>{{ String(v).replace('-', '−') }}</sup></span>
        </div>
        <div class="stack" :style="{ paddingTop: `${STACK_PAD - 16}px`, paddingRight: `${STACK_PAD}px` }">
          <div v-for="t in N" :key="t" class="plane" :class="{ front: t - 1 === sel || !aSliced }"
               :style="[planeStyle(t - 1, aSliced), { gridTemplateColumns: `repeat(${NC}, var(--cw))` }]">
            <template v-if="t - 1 === sel || (!aSliced && t === 1)">
              <template v-for="(row, h) in inputs.B" :key="h">
                <div v-for="(x, j) in row" :key="j" class="cell"
                     :style="s === 0 ? tintFp(x) : s === 1 ? tintBits(calc.Bc[h][j], calc.l2P + 1) : {}"
                     @mouseenter="hover = infoB(h, j)">
                  {{ s === 0 ? fmtFp(x) : s === 1 ? fmtInt(calc.Bc[h][j]) : calc.Blo[sel][h][j] }}
                </div>
              </template>
            </template>
            <span v-if="aSliced && t - 1 === sel" class="tag">mod {{ calc.mods[sel] }}</span>
          </div>
        </div>
      </div>

      <div class="op">=</div>

      <!-- C -->
      <div class="mat">
        <div class="label">{{ s < 3 ? 'C' : s === 3 ? 'Cᵢ' : s === 4 ? (corrected ? 'C″ = A′B′' : 'S') : 'C' }}</div>
        <div class="stack" :style="{ paddingTop: `${STACK_PAD - 16}px`, paddingRight: `${STACK_PAD}px` }">
          <div v-for="t in N" :key="t" class="plane" :class="{ front: merging ? t === 1 : t - 1 === sel || !cSliced, pending: s < 3 }"
               :style="[cPlaneStyle(t - 1), { gridTemplateColumns: `repeat(${NC}, var(--cw))` }]">
            <template v-if="merging ? t === 1 : t - 1 === sel || (!cSliced && t === 1)">
              <template v-for="i in M" :key="i">
                <div v-for="j in NC" :key="j" class="cell" :class="{ settled: merging && corrected }"
                     :style="s === 4 ? tintBits(cur[i - 1][j - 1], log2Big(calc.P) + 12) : s === 5 ? tintFp(calc.C[i - 1][j - 1]) : {}"
                     @mouseenter="hover = infoC(i - 1, j - 1)">
                  <template v-if="s < 3">?</template>
                  <template v-else-if="s === 3">{{ calc.Cmid[sel][i - 1][j - 1] }}</template>
                  <template v-else-if="s === 4">{{ fmtInt(cur[i - 1][j - 1]) }}</template>
                  <template v-else>{{ fmtFp(calc.C[i - 1][j - 1]) }}</template>
                </div>
              </template>
            </template>
            <span v-if="cSliced && t - 1 === sel" class="tag">mod {{ calc.mods[sel] }}</span>
            <span v-if="merging && t === 1" class="tag">{{ corrected ? '− 𝒫·round(S/𝒫)' : `Σ wᵢ·Cᵢ: ${terms} of ${N}` }}</span>
          </div>
        </div>
      </div>
    </div>

    <div class="slices" :class="{ show: s === 2 || s === 3 }">
      <span class="dim">slice</span>
      <button v-for="(p, t) in calc.mods" :key="t" :class="{ on: t === sel }" @click="pick(t)">{{ p }}</button>
    </div>

    <div class="bottom">
      <div class="explain">
        <div class="step-name">
          <span class="num">{{ s + 1 }}/{{ STEPS.length }}</span> {{ STEPS[s].name }}
          <span v-if="caption" class="caption" :class="{ ok: s === 4 && corrected && calc.crtOk }"><template v-for="(part, k) in rich(caption)" :key="k"><sup v-if="part.sup">{{ part.t }}</sup><template v-else>{{ part.t }}</template></template></span>
          <button v-if="s === 4" class="replay" @click="replayCrt">↻ replay</button>
        </div>
        <p><template v-for="(part, k) in rich(STEPS[s].text)" :key="k"><sup v-if="part.sup">{{ part.t }}</sup><template v-else>{{ part.t }}</template></template></p>
        <p class="hover"><template v-for="(part, k) in rich(hover || 'Hover a cell to see where its value comes from.')" :key="k"><sup v-if="part.sup">{{ part.t }}</sup><template v-else>{{ part.t }}</template></template></p>
      </div>
      <div class="controls">
        <div class="nav">
          <button :disabled="s === 0" @click="step = s - 1">←</button>
          <button :disabled="s === STEPS.length - 1" @click="step = s + 1">→</button>
        </div>
        <label>N = {{ N }}
          <input v-model.number="N" type="range" min="2" max="20" @keydown.stop>
        </label>
        <button @click="seed = (seed * 48271) % 2147483647">new matrices</button>
      </div>
    </div>
  </div>
</template>

<style scoped>
.oz {
  --ink: #0b0b0b;
  --ink-2: #52514e;
  --ink-3: #8a8984;
  --line: #d6d5d0;
  --surface: #fcfcfb;
  --panel: #ffffff;
  --accent: #2a78d6;
  --seq: #2a78d6;
  --good-text: #006300;
  --good: #0ca30c;
  --cw: 60px;

  color: var(--ink);
  font-size: 13px;
  line-height: 1.35;
  text-align: left;
}
html.dark .oz {
  --ink: #ffffff;
  --ink-2: #c3c2b7;
  --ink-3: #8a8984;
  --line: #3a3a37;
  --surface: #1a1a19;
  --panel: #222221;
  --accent: #3987e5;
  --seq: #3987e5;
  --good-text: #3fc43f;
  --good: #0ca30c;
}
.dim { color: var(--ink-3); font-weight: 400; }
sup { font-size: 0.72em; line-height: 0; }

.figure { display: flex; align-items: flex-end; gap: 8px; }
.mat { display: flex; flex-direction: column; }
.label { font-weight: 700; color: var(--ink-2); margin-bottom: 2px; }
.label .dim { display: none; }
.op { font-size: 20px; color: var(--ink-3); padding-bottom: 50px; }

.with-shifts { display: flex; align-items: flex-end; }
.shifts { color: var(--ink-2); font-size: 10.5px; opacity: 0; transition: opacity 0.3s; font-family: var(--slidev-code-font-family, ui-monospace, monospace); }
.shifts.show { opacity: 1; }
.shifts.rows { display: grid; grid-auto-rows: 26px; align-items: center; padding-right: 4px; width: 40px; text-align: right; }
.shifts.cols { display: grid; height: 16px; text-align: center; }

.stack { position: relative; display: grid; }
.plane {
  grid-area: 1 / 1;
  display: grid;
  grid-auto-rows: 26px;
  border: 1px solid var(--line);
  border-radius: 4px;
  background: var(--panel);
  transition: transform 0.5s cubic-bezier(0.45, 0, 0.55, 1), box-shadow 0.3s;
  min-height: 104px;
  position: relative;
}
.plane.front { border-color: var(--ink-3); }
.plane.pending { border-style: dashed; }
.cell {
  display: grid;
  place-items: center;
  margin: 1px;
  border-radius: 2px;
  font-family: var(--slidev-code-font-family, ui-monospace, monospace);
  font-size: 10px;
  color: var(--ink);
  white-space: nowrap;
  overflow: hidden;
}
.plane.pending .cell { color: var(--ink-3); }
.cell.settled { box-shadow: inset 0 0 0 1.5px var(--good); }
.tag {
  position: absolute;
  top: -9px;
  left: 6px;
  padding: 0 4px;
  border-radius: 3px;
  background: var(--accent);
  color: #fff;
  font-size: 9.5px;
  line-height: 15px;
}

.slices { display: flex; align-items: center; gap: 3px; margin-top: 8px; flex-wrap: wrap; opacity: 0; pointer-events: none; transition: opacity 0.3s; }
.slices.show { opacity: 1; pointer-events: auto; }
.slices .dim { margin-right: 4px; font-size: 11px; }

button {
  padding: 2px 7px;
  border: 1px solid var(--line);
  border-radius: 4px;
  background: var(--surface);
  color: var(--ink-2);
  font-size: 11px;
  cursor: pointer;
}
button:hover:not(:disabled) { color: var(--ink); border-color: var(--ink-3); }
button.on { background: var(--accent); border-color: var(--accent); color: #fff; }
button:disabled { opacity: 0.35; cursor: default; }

.bottom { display: flex; gap: 20px; margin-top: 10px; align-items: flex-start; }
.explain { flex: 1; }
.explain p { margin: 2px 0 0; }
.step-name { font-weight: 700; display: flex; align-items: baseline; gap: 8px; flex-wrap: wrap; }
.num { color: var(--ink-3); font-weight: 400; font-size: 11px; }
.caption { font-weight: 400; color: var(--ink-2); font-size: 12px; }
.caption.ok { color: var(--good-text); font-weight: 600; }
.replay { padding: 0 6px; font-weight: 400; }
.hover { color: var(--ink-2); font-style: italic; font-size: 11.5px; min-height: 1.4em; font-family: var(--slidev-code-font-family, ui-monospace, monospace); font-style: normal; }
.controls { display: flex; flex-direction: column; gap: 6px; align-items: flex-end; color: var(--ink-2); font-size: 12px; }
.controls label { display: flex; align-items: center; gap: 6px; }
.controls input { width: 110px; accent-color: var(--accent); }
.nav { display: flex; gap: 4px; }
</style>
