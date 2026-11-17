/** Reuse first-layer projections of repeated stem/context chunks within a search.
 * Inputs are already standardized. Only the linear layer is cached; activation
 * stays after the full sum, preserving the trained network's behavior.
 */
export class CachedDense {
  private readonly chunks: Array<{
    start: number;
    end: number;
    entries: Array<{ input: Float32Array; output: Float64Array }>;
  }> = [];

  constructor(private readonly inputDim: number, stateDim: number, stemDim: number) {
    for (let state = 0; state < 5; state++) {
      for (let stem = 0; stem < 4; stem++) {
        const start = state * stateDim + stem * stemDim;
        this.chunks.push({ start, end: start + stemDim, entries: [] });
      }
      this.chunks.push({ start: state * stateDim + 4 * stemDim, end: (state + 1) * stateDim, entries: [] });
    }
    this.chunks.push({ start: 5 * stateDim, end: inputDim, entries: [] });
  }

  clear(): void {
    for (const chunk of this.chunks) chunk.entries.length = 0;
  }

  evaluate(input: Float32Array, weights: Float32Array, bias: Float32Array, out: Float32Array): void {
    const sums = new Float64Array(bias);
    for (const chunk of this.chunks) {
      let projection: Float64Array | undefined;
      search: for (const entry of chunk.entries) {
        for (let i = chunk.start; i < chunk.end; i++) {
          if (input[i] !== entry.input[i - chunk.start]) continue search;
        }
        projection = entry.output;
        break;
      }
      if (!projection) {
        projection = new Float64Array(bias.length);
        for (let j = 0; j < bias.length; j++) {
          let sum = 0;
          const offset = j * this.inputDim;
          for (let i = chunk.start; i < chunk.end; i++) sum += weights[offset + i] * input[i];
          projection[j] = sum;
        }
        // A bounded search cache, not a session-long phrase cache.
        if (chunk.entries.length >= 16) chunk.entries.shift();
        chunk.entries.push({ input: input.slice(chunk.start, chunk.end), output: projection });
      }
      for (let j = 0; j < sums.length; j++) sums[j] += projection[j];
    }
    for (let j = 0; j < out.length; j++) out[j] = Math.max(0, sums[j]);
  }
}
