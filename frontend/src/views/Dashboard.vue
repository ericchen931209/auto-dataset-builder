<template>
  <div class="dashboard">
    <header class="page-header">
      <div>
        <h1>Dashboard</h1>
        <p class="subtitle">{{ store.datasets.length }} dataset(s) total</p>
      </div>
      <RouterLink to="/datasets/new" class="btn btn-primary">+ New Dataset</RouterLink>
    </header>

    <!-- Stats row -->
    <div class="stats-row">
      <div class="card stat-card">
        <div class="stat-value">{{ store.datasets.length }}</div>
        <div class="stat-label">Total Datasets</div>
      </div>
      <div class="card stat-card">
        <div class="stat-value">{{ totalImages }}</div>
        <div class="stat-label">Total Images</div>
      </div>
      <div class="card stat-card">
        <div class="stat-value">{{ readyCount }}</div>
        <div class="stat-label">Ready</div>
      </div>
      <div class="card stat-card">
        <div class="stat-value">{{ avgDqs }}</div>
        <div class="stat-label">Avg DQS</div>
      </div>
    </div>

    <!-- Loading -->
    <div v-if="store.loading" class="empty-state">Loading...</div>

    <!-- Empty -->
    <div v-else-if="store.datasets.length === 0" class="empty-state card">
      <div class="empty-icon">📂</div>
      <p>No datasets yet.</p>
      <RouterLink to="/datasets/new" class="btn btn-primary" style="margin-top:1rem">Create your first dataset</RouterLink>
    </div>

    <!-- Dataset grid -->
    <div v-else class="dataset-grid">
      <RouterLink
        v-for="ds in store.sortedDatasets"
        :key="ds.id"
        :to="`/datasets/${ds.id}`"
        class="card dataset-card"
      >
        <div class="ds-header">
          <span class="ds-name">{{ ds.name }}</span>
          <span :class="`badge badge-${ds.status}`">{{ ds.status }}</span>
        </div>

        <div class="ds-meta">
          <span v-if="ds.target_class">🎯 {{ ds.target_class }}</span>
          <span>🖼 {{ ds.total_images }} images</span>
          <span v-if="ds.region">📍 {{ ds.region }}</span>
        </div>

        <!-- DQS bar -->
        <div v-if="ds.dqs_score !== null" class="dqs-row">
          <span class="dqs-label">DQS</span>
          <div class="progress-bar-bg" style="flex:1">
            <div class="progress-bar-fill" :style="{ width: `${ds.dqs_score * 100}%`, background: dqsColor(ds.dqs_score) }" />
          </div>
          <span class="dqs-value">{{ (ds.dqs_score * 100).toFixed(0) }}</span>
        </div>

        <div class="ds-date">{{ formatDate(ds.created_at) }}</div>
      </RouterLink>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted } from 'vue'
import { RouterLink } from 'vue-router'
import { useDatasetStore } from '@/stores/dataset'

const store = useDatasetStore()
onMounted(() => store.fetchDatasets())

const totalImages = computed(() => store.datasets.reduce((s, d) => s + d.total_images, 0))
const readyCount = computed(() => store.datasets.filter(d => d.status === 'ready').length)
const avgDqs = computed(() => {
  const scored = store.datasets.filter(d => d.dqs_score !== null)
  if (!scored.length) return '—'
  return ((scored.reduce((s, d) => s + d.dqs_score!, 0) / scored.length) * 100).toFixed(0)
})

function dqsColor(score: number) {
  if (score >= 0.75) return 'var(--success)'
  if (score >= 0.5) return 'var(--warn)'
  return 'var(--danger)'
}

function formatDate(iso: string) {
  return new Date(iso).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })
}
</script>

<style scoped>
.dashboard { max-width: 1100px; }

.page-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 1.5rem;
}
h1 { font-size: 1.6rem; font-weight: 700; }
.subtitle { color: var(--text-muted); font-size: 0.875rem; margin-top: 2px; }

.stats-row {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 1rem;
  margin-bottom: 1.5rem;
}
.stat-card { text-align: center; }
.stat-value { font-size: 2rem; font-weight: 700; color: var(--accent); }
.stat-label { font-size: 0.75rem; color: var(--text-muted); margin-top: 4px; text-transform: uppercase; letter-spacing: 0.05em; }

.dataset-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
  gap: 1rem;
}

.dataset-card {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
  transition: border-color 0.15s, transform 0.15s;
}
.dataset-card:hover { border-color: var(--accent); transform: translateY(-2px); }

.ds-header { display: flex; align-items: center; justify-content: space-between; gap: 0.5rem; }
.ds-name { font-weight: 600; font-size: 0.95rem; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }

.ds-meta { display: flex; flex-wrap: wrap; gap: 0.5rem; font-size: 0.78rem; color: var(--text-muted); }

.dqs-row { display: flex; align-items: center; gap: 0.5rem; }
.dqs-label { font-size: 0.7rem; color: var(--text-muted); width: 28px; }
.dqs-value { font-size: 0.78rem; font-weight: 600; width: 24px; text-align: right; }

.ds-date { font-size: 0.72rem; color: var(--text-muted); }

.empty-state { text-align: center; padding: 3rem; color: var(--text-muted); }
.empty-icon { font-size: 3rem; margin-bottom: 1rem; }
</style>
