import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import { datasetsApi, jobsApi, type Dataset, type Job } from '@/api'

export const useDatasetStore = defineStore('dataset', () => {
  const datasets = ref<Dataset[]>([])
  const currentDataset = ref<Dataset | null>(null)
  const jobs = ref<Job[]>([])
  const loading = ref(false)
  const error = ref<string | null>(null)

  const sortedDatasets = computed(() =>
    [...datasets.value].sort(
      (a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime()
    )
  )

  async function fetchDatasets() {
    loading.value = true
    error.value = null
    try {
      datasets.value = await datasetsApi.list()
    } catch (e: any) {
      error.value = e.message
    } finally {
      loading.value = false
    }
  }

  async function fetchDataset(id: number) {
    loading.value = true
    try {
      currentDataset.value = await datasetsApi.get(id)
    } catch (e: any) {
      error.value = e.message
    } finally {
      loading.value = false
    }
  }

  async function createDataset(query: string, name?: string) {
    const ds = await datasetsApi.create(query, name)
    datasets.value.unshift(ds)
    return ds
  }

  async function deleteDataset(id: number) {
    await datasetsApi.delete(id)
    datasets.value = datasets.value.filter(d => d.id !== id)
    if (currentDataset.value?.id === id) currentDataset.value = null
  }

  async function evaluateDqs(id: number) {
    const result = await datasetsApi.evaluateDqs(id)
    // Update in place
    const idx = datasets.value.findIndex(d => d.id === id)
    if (idx !== -1) datasets.value[idx] = { ...datasets.value[idx], dqs_score: result.dqs_score, ...result.features }
    if (currentDataset.value?.id === id) {
      currentDataset.value = { ...currentDataset.value, dqs_score: result.dqs_score }
    }
    return result
  }

  async function fetchJobs(datasetId?: number) {
    jobs.value = await jobsApi.list(datasetId)
  }

  return {
    datasets, currentDataset, jobs, loading, error, sortedDatasets,
    fetchDatasets, fetchDataset, createDataset, deleteDataset, evaluateDqs, fetchJobs,
  }
})
