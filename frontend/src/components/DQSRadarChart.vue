<template>
  <div class="radar-wrap">
    <Radar :data="chartData" :options="chartOptions" />
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { Radar } from 'vue-chartjs'
import {
  Chart as ChartJS,
  RadialLinearScale,
  PointElement,
  LineElement,
  Filler,
  Tooltip,
  Legend,
} from 'chart.js'

ChartJS.register(RadialLinearScale, PointElement, LineElement, Filler, Tooltip, Legend)

const props = defineProps<{
  annotationQuality: number
  diversity: number
  lighting: number
  pose: number
  classBalance: number
}>()

const chartData = computed(() => ({
  labels: ['Annotation Quality', 'Diversity', 'Lighting', 'Pose', 'Class Balance'],
  datasets: [
    {
      label: 'DQS Features',
      data: [
        props.annotationQuality,
        props.diversity,
        props.lighting,
        props.pose,
        props.classBalance,
      ],
      backgroundColor: 'rgba(99,102,241,0.2)',
      borderColor: 'rgba(99,102,241,0.9)',
      borderWidth: 2,
      pointBackgroundColor: 'rgba(99,102,241,1)',
      pointRadius: 4,
    },
  ],
}))

const chartOptions = {
  responsive: true,
  maintainAspectRatio: true,
  scales: {
    r: {
      min: 0,
      max: 1,
      ticks: { stepSize: 0.25, color: '#64748b', backdropColor: 'transparent', font: { size: 10 } },
      grid: { color: '#2d3050' },
      angleLines: { color: '#2d3050' },
      pointLabels: { color: '#94a3b8', font: { size: 11 } },
    },
  },
  plugins: {
    legend: { display: false },
    tooltip: {
      callbacks: {
        label: (ctx: any) => ` ${(ctx.raw * 100).toFixed(1)}%`,
      },
    },
  },
}
</script>

<style scoped>
.radar-wrap { width: 100%; max-width: 320px; margin: 0 auto; }
</style>
