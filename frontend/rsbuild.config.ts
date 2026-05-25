import { defineConfig } from '@rsbuild/core';
import { pluginReact } from '@rsbuild/plugin-react';

const backendTarget = process.env.NEXUS_BACKEND_URL || 'http://127.0.0.1:7861';
const frontendBase = process.env.NEXUS_FRONTEND_BASE || 'auto';

export default defineConfig({
  plugins: [pluginReact()],
  html: {
    title: 'Nexus BTA',
  },
  source: {
    entry: {
      index: './src/main.tsx',
    },
  },
  output: {
    assetPrefix: frontendBase,
  },
  server: {
    host: '127.0.0.1',
    port: 3000,
    proxy: {
      '/api': {
        target: backendTarget,
        changeOrigin: true,
      },
      '/outputs': {
        target: backendTarget,
        changeOrigin: true,
      },
      '/assets': {
        target: backendTarget,
        changeOrigin: true,
      },
      '/model-assets': {
        target: backendTarget,
        changeOrigin: true,
      },
    },
  },
});
