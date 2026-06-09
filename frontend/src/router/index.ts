import { createRouter, createWebHistory } from 'vue-router'

const router = createRouter({
  history: createWebHistory(),
  routes: [
    { path: '/', redirect: '/dashboard' },
    { path: '/dashboard', component: () => import('@/views/Dashboard.vue'), meta: { title: 'Dashboard' } },
    { path: '/datasets/new', component: () => import('@/views/CreateDataset.vue'), meta: { title: 'New Dataset' } },
    { path: '/datasets/:id', component: () => import('@/views/DatasetDetail.vue'), meta: { title: 'Dataset' } },
  ],
})

router.afterEach(to => {
  document.title = `${to.meta.title ?? 'ADB'} — Auto Dataset Builder`
})

export default router
