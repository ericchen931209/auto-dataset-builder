<template>
  <div class="detail-page" v-if="ds">
    <!-- Header -->
    <header class="page-header">
      <RouterLink to="/dashboard" class="btn btn-ghost">← Back</RouterLink>
      <div class="header-main">
        <h1>{{ ds.name }}</h1>
        <span :class="`badge badge-${ds.status}`">{{ ds.status }}</span>
      </div>
      <div class="header-actions">
        <button class="btn btn-ghost" @click="refresh">↻ Refresh</button>
        <button class="btn btn-danger" @click="handleDelete">Delete</button>
      </div>
    </header>

    <!-- Top row -->
    <div class="top-row">
      <!-- Info card -->
      <div class="card info-card">
        <h3>Dataset Info</h3>
        <table class="info-table">
          <tr><td>ID</td><td>#{{ ds.id }}</td></tr>
          <tr><td>Version</td><td>{{ ds.version }}</td></tr>
          <tr><td>Target</td><td>{{ ds.target_class ?? '—' }}</td></tr>
          <tr><td>Task</td><td>{{ ds.task_type ?? '—' }}</td></tr>
          <tr><td>Region</td><td>{{ ds.region ?? '—' }}</td></tr>
          <tr><td>Images</td><td>{{ ds.total_images }}</td></tr>
          <tr><td>Annotated</td><td>{{ ds.annotated_images }}</td></tr>
          <tr><td>Created</td><td>{{ formatDate(ds.created_at) }}</td></tr>
        </table>
      </div>

      <!-- DQS card -->
      <div class="card dqs-card">
        <div class="dqs-header">
          <h3>Neural DQS</h3>
          <button class="btn btn-primary" :disabled="evaluating" @click="handleEvaluate">
            {{ evaluating ? 'Evaluating...' : '▶ Evaluate' }}
          </button>
        </div>

        <div v-if="ds.dqs_score !== null" class="dqs-score-display">
          <div class="big-score" :style="{ color: dqsColor(ds.dqs_score) }">
            {{ (ds.dqs_score * 100).toFixed(1) }}
          </div>
          <div class="score-label">/ 100</div>
        </div>
        <div v-else class="dqs-empty">Not evaluated yet</div>

        <DQSRadarChart
          v-if="ds.dqs_annotation_quality !== null"
          :annotation-quality="ds.dqs_annotation_quality!"
          :diversity="ds.dqs_diversity!"
          :lighting="ds.dqs_lighting!"
          :pose="ds.dqs_pose!"
          :class-balance="ds.dqs_class_balance!"
        />

        <!-- Feature breakdown -->
        <div v-if="ds.dqs_annotation_quality !== null" class="feature-bars">
          <FeatureBar label="Annotation Quality" :value="ds.dqs_annotation_quality!" />
          <FeatureBar label="Diversity"           :value="ds.dqs_diversity!" />
          <FeatureBar label="Lighting"            :value="ds.dqs_lighting!" />
          <FeatureBar label="Pose"                :value="ds.dqs_pose!" />
          <FeatureBar label="Class Balance"       :value="ds.dqs_class_balance!" />
        </div>
      </div>
    </div>

    <!-- Pipeline progress -->
    <div class="card pipeline-card">
      <h3>Pipeline Status</h3>
      <div class="pipeline-steps">
        <PipelineStep
          v-for="step in pipelineSteps"
          :key="step.label"
          v-bind="step"
        />
      </div>
    </div>

    <!-- Description -->
    <div class="card" v-if="ds.description">
      <h3>Original Query</h3>
      <p class="description">{{ ds.description }}</p>
    </div>
  </div>

  <div v-else-if="store.loading" class="empty-state">Loading...</div>
  <div v-else class="empty-state">Dataset not found.</div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { RouterLink, useRouter, useRoute } from 'vue-router'
import { useDatasetStore } from '@/stores/dataset'
import DQSRadarChart from '@/components/DQSRadarChart.vue'

const route = useRoute()
const router = useRouter()
const store = useDatasetStore()
const evaluating = ref(false)
const id = Number(route.params.id)

const ds = computed(() => store.currentDataset)

onMounted(() => store.fetchDataset(id))

async function refresh() {
  await store.fetchDataset(id)
}

async function handleEvaluate() {
  evaluating.value = true
  try {
    await store.evaluateDqs(id)
    await store.fetchDataset(id)
  } catch (e: any) {
    alert(e.response?.data?.detail ?? e.message)
  } finally {
    evaluating.value = false
  }
}

async function handleDelete() {
  if (!confirm(`Delete "${ds.value?.name}"? This cannot be undone.`)) return
  await store.deleteDataset(id)
  router.push('/dashboard')
}

function dqsColor(score: number) {
  if (score >= 0.75) return 'var(--success)'
  if (score >= 0.5) return 'var(--warn)'
  return 'var(--danger)'
}

function formatDate(iso: string) {
  return new Date(iso).toLocaleString()
}

const STATUS_ORDER = ['collecting', 'annotating', 'cleaning', 'evaluating', 'ready']
const pipelineSteps = computed(() => {
  const current = ds.value?.status ?? 'pending'
  const currentIdx = STATUS_ORDER.indexOf(current)
  return [
    { label: 'Collecting', icon: '📥', status: stepStatus('collecting', currentIdx, 0) },
    { label: 'Annotating', icon: '🏷', status: stepStatus('annotating', currentIdx, 1) },
    { label: 'Cleaning',   icon: '🧹', status: stepStatus('cleaning',   currentIdx, 2) },
    { label: 'Evaluating', icon: '📊', status: stepStatus('evaluating', currentIdx, 3) },
    { label: 'Ready',      icon: '✅', status: stepStatus('ready',      currentIdx, 4) },
  ]
})

function stepStatus(name: string, currentIdx: number, stepIdx: number): 'done' | 'active' | 'pending' {
  if (ds.value?.status === 'failed') return 'pending'
  if (currentIdx > stepIdx) return 'done'
  if (currentIdx === stepIdx) return 'active'
  return 'pending'
}
</script>

<!-- Sub-components defined inline for simplicity -->
<script lang="ts">
import { defineComponent, h } from 'vue'

export const FeatureBar = defineComponent({
  props: { label: String, value: Number },
  setup(props) {
    const pct = ((props.value ?? 0) * 100).toFixed(1)
    const color = (props.value ?? 0) >= 0.75 ? 'var(--success)'
                : (props.value ?? 0) >= 0.5  ? 'var(--warn)'
                : 'var(--danger)'
    return () => h('div', { class: 'fb-row' }, [
      h('span', { class: 'fb-label' }, props.label),
      h('div', { class: 'progress-bar-bg', style: 'flex:1' }, [
        h('div', { class: 'progress-bar-fill', style: `width:${pct}%;background:${color}` }),
      ]),
      h('span', { class: 'fb-val' }, `${pct}%`),
    ])
  },
})

export const PipelineStep = defineComponent({
  props: { label: String, icon: String, status: String },
  setup(props) {
    return () => h('div', { class: `ps-step ps-${props.status}` }, [
      h('div', { class: 'ps-icon' }, props.icon),
      h('div', { class: 'ps-label' }, props.label),
    ])
  },
})
</script>

<style scoped>
.detail-page { max-width: 1000px; display: flex; flex-direction: column; gap: 1.25rem; }

.page-header { display: flex; align-items: center; gap: 1rem; flex-wrap: wrap; }
.header-main { display: flex; align-items: center; gap: 0.75rem; flex: 1; }
h1 { font-size: 1.4rem; font-weight: 700; }
.header-actions { display: flex; gap: 0.5rem; }

h3 { font-size: 0.95rem; font-weight: 600; margin-bottom: 1rem; }

/* Top row */
.top-row { display: grid; grid-template-columns: 280px 1fr; gap: 1.25rem; }

.info-table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
.info-table td { padding: 0.3rem 0; }
.info-table td:first-child { color: var(--text-muted); width: 100px; }

.dqs-card { display: flex; flex-direction: column; gap: 1rem; }
.dqs-header { display: flex; align-items: center; justify-content: space-between; }

.dqs-score-display { display: flex; align-items: baseline; gap: 0.25rem; }
.big-score { font-size: 3.5rem; font-weight: 800; line-height: 1; }
.score-label { font-size: 1.5rem; color: var(--text-muted); }

.dqs-empty { color: var(--text-muted); font-size: 0.875rem; }

.feature-bars { display: flex; flex-direction: column; gap: 0.5rem; }

:deep(.fb-row) { display: flex; align-items: center; gap: 0.5rem; font-size: 0.78rem; }
:deep(.fb-label) { color: var(--text-muted); width: 130px; flex-shrink: 0; }
:deep(.fb-val) { width: 42px; text-align: right; font-weight: 500; font-size: 0.75rem; }

/* Pipeline */
.pipeline-steps { display: flex; gap: 0; align-items: center; }

:deep(.ps-step) {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 0.4rem;
  flex: 1;
  padding: 0.75rem 0.5rem;
  border-radius: 8px;
  transition: all 0.15s;
}
:deep(.ps-icon) { font-size: 1.5rem; }
:deep(.ps-label) { font-size: 0.72rem; color: var(--text-muted); text-align: center; }
:deep(.ps-done)   { background: rgba(34,197,94,0.08); }
:deep(.ps-done .ps-icon)   { filter: none; }
:deep(.ps-active) { background: rgba(99,102,241,0.12); border: 1px solid var(--accent); }
:deep(.ps-active .ps-label) { color: var(--accent); font-weight: 600; }
:deep(.ps-pending .ps-icon) { opacity: 0.3; }

.description { color: var(--text-muted); font-size: 0.875rem; line-height: 1.6; }

.empty-state { text-align: center; padding: 3rem; color: var(--text-muted); }
</style>
