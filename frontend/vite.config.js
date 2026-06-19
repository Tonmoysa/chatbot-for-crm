import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

const apiProxy = {
  "/api": {
    target: "http://127.0.0.1:8000",
    changeOrigin: true,
  },
};

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");
  const useProxy = env.VITE_USE_PROXY !== "false";

  return {
    plugins: [react()],
    server: useProxy ? { proxy: apiProxy } : {},
    preview: useProxy ? { proxy: apiProxy } : {},
  };
});
