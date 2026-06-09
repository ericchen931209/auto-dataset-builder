import axios from 'axios'

const api = axios.create({
  baseURL: '/api/v1',
  timeout: 15000,
})

export interface Dataset {
  id: number
  name: string
  description: string | null
  status: string
  version: string
  target_class: string | null
  task_type: string | null
  region: string | null
  total_images: number
  annotated_images: number
  dqs_score: number | null
  dqs_annotation_quality: number | null
  dqs_diversity: number | null
  dqs_lighting: number | null
  dqs_pose: number | null
  dqs_class_balance: number | null
  created_at: string
  updated_at: string
}

export interface Job {
  id: number
  dataset_id: number
  job_type: string
  status: string
  progress: number
  progress_message: string | null
  error_message: string | null
  created_at: string
  updated_at: string
}

export interface DQSResult {
  dataset_id: number
  dqs_score: number
  features: {
    annotation_quality: number
    diversity: number
    lighting_diversity: number
    pose_diversity: number
    class_balance: number
  }
}

export const datasetsApi = {
  list: () => api.get<Dataset[]>('/datasets').then(r => r.data),
  get: (id: number) => api.get<Dataset>(`/datasets/${id}`).then(r => r.data),
  create: (query: string, name?: string) =>
    api.post<Dataset>('/datasets', { query, name }).then(r => r.data),
  delete: (id: number) => api.delete(`/datasets/${id}`),
  evaluateDqs: (id: number) =>
    api.post<DQSResult>(`/datasets/${id}/evaluate-dqs`).then(r => r.data),
}

export const jobsApi = {
  list: (datasetId?: number) =>
    api.get<Job[]>('/jobs', { params: datasetId ? { dataset_id: datasetId } : {} }).then(r => r.data),
  get: (id: number) => api.get<Job>(`/jobs/${id}`).then(r => r.data),
}

export default api
