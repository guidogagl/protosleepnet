import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// base './' keeps asset URLs relative so the build works from any subpath
// (e.g. a Hugging Face Static Space). The data bundle is fetched from
// VITE_DATA_URL (defaults to ./data for local dev / co-hosted data).
export default defineConfig({
  base: "./",
  plugins: [react()],
  server: { port: 5173, host: true },
});
