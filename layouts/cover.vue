<script setup lang="ts">
defineProps<{ date?: string; author?: string; institute?: string }>()
</script>

<template>
  <div class="slidev-layout fzj-cover">
    <img class="graphic" src="/title-graphic.jpg" alt="" />
    <div class="band">
      <slot />
      <p class="meta">
        <span v-for="(item, i) in [date, author, institute].filter(Boolean)" :key="i">{{ item }}</span>
      </p>
    </div>
  </div>
</template>
