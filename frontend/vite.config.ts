import { defineConfig } from "vite";

export default defineConfig({
  base: "./",
  optimizeDeps: {
    // Signalsmith Stretch stringifies its WasmProcessor/Module source into an
    // AudioWorklet blob. Vite's dep optimizer wrapper breaks that source.
    exclude: ["signalsmith-stretch"],
  },
  resolve: {
    // onnxruntime-web extern-wasm build: do NOT bundle .wasm into dist —
    // the worker loads it from public/ort/ via ort.env.wasm.wasmPaths.
    conditions: ["onnxruntime-web-use-extern-wasm"],
  },
  build: {
    outDir: "dist",
    emptyOutDir: true,
  },
});
