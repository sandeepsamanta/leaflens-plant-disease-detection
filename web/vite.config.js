import react from '@vitejs/plugin-react';
import { defineConfig } from 'vite';

// In development /api is proxied to the FastAPI server, so no CORS setup is needed.
const target = process.env.API_PROXY_TARGET || 'http://localhost:8000';
export default defineConfig({
  plugins: [react()],
  server: { port: 5173, proxy: { '/api': target } },
  test: { environment: 'node' },
});
