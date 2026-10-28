import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const target = process.env.VITE_API_URL || "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": target,
      "/metrics": target,
      "/ws": { target, ws: true },
    },
  },
  build: { outDir: "dist", sourcemap: false },
});
