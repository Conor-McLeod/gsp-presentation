<script setup lang="ts">
import { computed } from 'vue'

// Side-by-side sketch of how Ozaki scheme I and II turn one FP64 number into
// int8 data. Scheme I chops the significand into 7-bit chunks; scheme II keeps
// the whole integer and reduces it modulo pairwise-coprime int8 moduli.

const props = withDefaults(defineProps<{
  scheme: 'I' | 'II'
  x?: number
  slices?: number
  moduli?: number
}>(), {
  x: Math.PI,
  slices: 8,
  moduli: 14,
})

const MODULI = [256, 255, 253, 251, 247, 241, 239, 233, 229, 227, 223, 217, 211, 199]
const CHUNK = 7 // magnitude bits that fit in a signed int8

// 53-bit significand of x (implicit leading 1 included), as a bit string.
const bits = computed(() => {
  const dv = new DataView(new ArrayBuffer(8))
  dv.setFloat64(0, props.x)
  const m = (BigInt(dv.getUint32(0) & 0xFFFFF) << 32n) | BigInt(dv.getUint32(4)) | (1n << 52n)
  return m.toString(2)
})
const exponent = computed(() => Math.floor(Math.log2(Math.abs(props.x))))

const chunks = computed(() => {
  const out: { bits: string, value: number }[] = []
  for (let i = 0; i < bits.value.length; i += CHUNK) {
    const b = bits.value.slice(i, i + CHUNK)
    out.push({ bits: b, value: Number.parseInt(b, 2) })
  }
  return out
})

const residues = computed(() => {
  const m = BigInt(`0b${bits.value}`)
  return MODULI.slice(0, props.moduli).map((p) => {
    const P = BigInt(p)
    let r = m % P
    if (r >= P / 2n) r -= P // signed, so it fits in int8
    return { p, r: Number(r) }
  })
})

// Slice products A_i B_j that scheme I keeps: i + j <= s + 1.
const grid = computed(() => Array.from({ length: props.slices }, (_, i) =>
  Array.from({ length: props.slices }, (_, j) => i + j + 2 <= props.slices + 1)))
const kept = computed(() => props.slices * (props.slices + 1) / 2)
</script>

<template>
  <div class="oz">
    <div class="caption">
      significand of <b>x = {{ x === Math.PI ? 'π' : x }}</b>&nbsp;<span class="dim">(53 bits, × 2<sup>{{ exponent }}</sup>)</span>
    </div>

    <div v-if="scheme === 'I'" class="bar">
      <span
        v-for="(c, k) in chunks" :key="k"
        class="chunk" :class="k % 2 ? 'odd' : 'even'"
      >{{ c.bits }}</span>
    </div>
    <div v-else class="bar">
      <span class="chunk whole">{{ bits }}</span>
    </div>

    <div class="step">
      {{ scheme === 'I' ? '↓ chop into 7-bit pieces' : '↓ reduce the whole integer mod each pᵢ' }}
    </div>

    <div v-if="scheme === 'I'" class="boxes">
      <div v-for="(c, k) in chunks" :key="k" class="box" :class="k % 2 ? 'odd' : 'even'">
        <div class="val">{{ c.value }}</div>
        <div class="tag">x<sub>{{ k + 1 }}</sub></div>
      </div>
    </div>
    <div v-else class="boxes">
      <div v-for="(r, k) in residues" :key="k" class="box res">
        <div class="val">{{ r.r }}</div>
        <div class="tag">mod {{ r.p }}</div>
      </div>
    </div>

    <div class="step">
      {{ scheme === 'I' ? '↓ one int8 GEMM per slice pair' : '↓ one int8 GEMM per modulus' }}
    </div>

    <div class="cost">
      <div v-if="scheme === 'I'" class="grid" :style="{ gridTemplateColumns: `repeat(${slices}, var(--cell))` }">
        <template v-for="(row, i) in grid" :key="i">
          <span v-for="(on, j) in row" :key="j" class="cell" :class="{ on }" />
        </template>
      </div>
      <div v-else class="grid" :style="{ gridTemplateColumns: `repeat(${moduli}, var(--cell))` }">
        <span v-for="k in moduli" :key="k" class="cell on" />
      </div>
      <div class="count">
        <template v-if="scheme === 'I'">
          <b>{{ kept }}</b> GEMMs<br>
          <span class="dim">s(s+1)/2, quadratic</span><br>
          <span class="dim">hollow: below FP64, dropped</span>
        </template>
        <template v-else>
          <b>{{ moduli }}</b> GEMMs<br>
          <span class="dim">N, linear</span><br>
          <span class="dim">independent until the CRT sum</span>
        </template>
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
  --a: #2a78d6;
  --a-soft: #cfe0f6;
  --b: #e07b1f;
  --b-soft: #f8dcc2;
  --cell: 14px;

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
  --a: #3987e5;
  --a-soft: #1d3a5e;
  --b: #f0913a;
  --b-soft: #5a3816;
}
.dim { color: var(--ink-3); }
sup { font-size: 0.72em; }

.caption { color: var(--ink-2); margin-bottom: 4px; }

.bar { display: flex; font-family: var(--slidev-code-font-family, monospace); font-size: 9.5px; letter-spacing: -0.2px; }
.chunk { padding: 2px 1px; border-radius: 2px; }
.chunk.even { background: var(--a-soft); }
.chunk.odd { background: var(--b-soft); }
.chunk.whole { background: var(--line); }

.step { color: var(--ink-3); font-size: 12px; margin: 6px 0; }

.boxes { display: flex; flex-wrap: wrap; gap: 4px; }
.box {
  min-width: 32px;
  border: 1.5px solid var(--line);
  border-radius: 3px;
  text-align: center;
  padding: 1px 2px;
}
.box.even { border-color: var(--a); }
.box.odd { border-color: var(--b); }
.box.res { border-color: var(--ink-2); }
.val { font-family: var(--slidev-code-font-family, monospace); font-weight: 700; }
.tag { font-size: 10px; color: var(--ink-3); }

.cost { display: flex; align-items: flex-start; gap: 14px; }
.grid { display: grid; gap: 2px; }
.cell {
  width: var(--cell);
  height: var(--cell);
  border: 1px solid var(--line);
  border-radius: 2px;
}
.cell.on { background: var(--a); border-color: var(--a); }
.count { font-size: 13px; }
</style>
