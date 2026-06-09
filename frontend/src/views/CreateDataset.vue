<template>
  <div class="create-page">
    <header class="page-header">
      <RouterLink to="/dashboard" class="btn btn-ghost">← Back</RouterLink>
      <h1>New Dataset</h1>
    </header>

    <div class="card create-card">
      <h2>Describe your dataset</h2>
      <p class="hint">ADB will automatically search, download, annotate, and evaluate your dataset.</p>

      <form @submit.prevent="handleSubmit">
        <div class="field">
          <label>Natural Language Request *</label>
          <textarea
            v-model="query"
            rows="3"
            placeholder="e.g. Build a Taiwan motorcycle detection dataset"
            required
          />
        </div>

        <div class="field">
          <label>Dataset Name <span class="opt">(optional)</span></label>
          <input v-model="name" type="text" placeholder="Auto-generated from query if empty" />
        </div>

        <div v-if="error" class="error-msg">{{ error }}</div>

        <div class="actions">
          <button type="submit" class="btn btn-primary" :disabled="submitting">
            {{ submitting ? 'Creating...' : '🚀 Create Dataset' }}
          </button>
        </div>
      </form>
    </div>

    <!-- Example queries -->
    <div class="examples">
      <p class="examples-title">Example queries</p>
      <div class="example-chips">
        <button
          v-for="ex in examples"
          :key="ex"
          class="chip"
          @click="query = ex"
        >{{ ex }}</button>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { RouterLink, useRouter } from 'vue-router'
import { useDatasetStore } from '@/stores/dataset'

const router = useRouter()
const store = useDatasetStore()

const query = ref('')
const name = ref('')
const submitting = ref(false)
const error = ref('')

const examples = [
  'Build a Taiwan motorcycle detection dataset',
  'Build a fall detection dataset for elderly people',
  'Create a traffic light recognition dataset for Taipei streets',
  'Build a construction worker safety helmet dataset',
]

async function handleSubmit() {
  if (!query.value.trim()) return
  submitting.value = true
  error.value = ''
  try {
    const ds = await store.createDataset(query.value.trim(), name.value.trim() || undefined)
    router.push(`/datasets/${ds.id}`)
  } catch (e: any) {
    error.value = e.response?.data?.detail ?? e.message
  } finally {
    submitting.value = false
  }
}
</script>

<style scoped>
.create-page { max-width: 680px; }

.page-header { display: flex; align-items: center; gap: 1rem; margin-bottom: 1.5rem; }
h1 { font-size: 1.4rem; font-weight: 700; }

.create-card h2 { font-size: 1.1rem; margin-bottom: 0.25rem; }
.hint { color: var(--text-muted); font-size: 0.875rem; margin-bottom: 1.5rem; }

.field { display: flex; flex-direction: column; gap: 0.4rem; margin-bottom: 1.25rem; }
.field label { font-size: 0.875rem; font-weight: 500; }
.opt { color: var(--text-muted); font-weight: 400; }

.actions { display: flex; justify-content: flex-end; margin-top: 0.5rem; }

.error-msg {
  background: rgba(239,68,68,0.1);
  border: 1px solid var(--danger);
  border-radius: 8px;
  padding: 0.75rem 1rem;
  color: var(--danger);
  font-size: 0.875rem;
  margin-bottom: 1rem;
}

.examples { margin-top: 1.5rem; }
.examples-title { font-size: 0.8rem; color: var(--text-muted); margin-bottom: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; }
.example-chips { display: flex; flex-wrap: wrap; gap: 0.5rem; }
.chip {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 999px;
  padding: 0.375rem 0.875rem;
  font-size: 0.8rem;
  color: var(--text-muted);
  cursor: pointer;
  transition: all 0.15s;
}
.chip:hover { border-color: var(--accent); color: var(--accent); }
</style>
