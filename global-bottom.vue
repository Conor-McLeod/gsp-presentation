<script setup lang="ts">
import { useNav } from '@slidev/client'

const { currentPage, currentLayout } = useNav()
</script>

<template>
  <footer class="fzj-footer">
    <span>Member of the Helmholtz Association</span>
    <template v-if="currentLayout !== 'cover'">
      <span>{{ $slidev.configs.date }}</span>
      <span>Slide {{ currentPage }}</span>
    </template>
    <img src="/fzj.svg" alt="Forschungszentrum Jülich" />
  </footer>
</template>
