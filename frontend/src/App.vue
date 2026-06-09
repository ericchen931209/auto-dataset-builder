<template>
  <div class="app-layout">
    <!-- Sidebar -->
    <aside class="sidebar">
      <div class="sidebar-logo">
        <span class="logo-icon">⚡</span>
        <span class="logo-text">ADB</span>
      </div>
      <nav class="sidebar-nav">
        <RouterLink to="/dashboard" class="nav-item">
          <span class="nav-icon">📊</span> Dashboard
        </RouterLink>
        <RouterLink to="/datasets/new" class="nav-item nav-item-accent">
          <span class="nav-icon">＋</span> New Dataset
        </RouterLink>
      </nav>
      <div class="sidebar-footer">
        <a href="https://github.com/ericchen931209/auto-dataset-builder" target="_blank" class="nav-item">
          <span class="nav-icon">🔗</span> GitHub
        </a>
      </div>
    </aside>

    <!-- Main content -->
    <main class="main-content">
      <RouterView />
    </main>
  </div>
</template>

<script setup lang="ts">
import { RouterLink, RouterView } from 'vue-router'
</script>

<style scoped>
.app-layout {
  display: flex;
  min-height: 100vh;
}

.sidebar {
  width: 220px;
  background: var(--bg-card);
  border-right: 1px solid var(--border);
  display: flex;
  flex-direction: column;
  padding: 1.25rem 0;
  flex-shrink: 0;
}

.sidebar-logo {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0 1.25rem 1.5rem;
  font-size: 1.2rem;
  font-weight: 700;
  letter-spacing: -0.02em;
}
.logo-icon { font-size: 1.5rem; }

.sidebar-nav { display: flex; flex-direction: column; gap: 2px; padding: 0 0.75rem; flex: 1; }

.nav-item {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  padding: 0.6rem 0.75rem;
  border-radius: 8px;
  font-size: 0.875rem;
  color: var(--text-muted);
  transition: all 0.15s;
}
.nav-item:hover, .nav-item.router-link-active {
  background: var(--bg-card2);
  color: var(--text);
}
.nav-item-accent { color: var(--accent); }
.nav-item-accent:hover { background: rgba(99,102,241,0.12); color: var(--accent-hover); }

.nav-icon { font-size: 1rem; }

.sidebar-footer { padding: 0.75rem; border-top: 1px solid var(--border); margin-top: auto; }

.main-content { flex: 1; overflow-y: auto; padding: 2rem; }
</style>
