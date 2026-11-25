// Self-contained AudioWorklet DJ Performance FX Processor (Blob URL loaded)
// Provides zero-allocation real-time:
//   - Beat-synced Stutter / Beat-Repeat Roll (1/4, 1/8, 1/16 synced to BPM)
//   - Smooth Vinyl Tape-Stop Brake & Snapback
//   - Sample-and-hold / Bit-Crush Glitch texture

const DJ_WORKLET_CODE = `
class WubDjFxProcessor extends AudioWorkletProcessor {
  static get parameterDescriptors() {
    return [
      { name: 'bpm', defaultValue: 126, minValue: 60, maxValue: 200, automationRate: 'k-rate' },
      { name: 'rollBeats', defaultValue: 0, minValue: 0, maxValue: 2, automationRate: 'k-rate' },
      { name: 'tapeStop', defaultValue: 0, minValue: 0, maxValue: 1, automationRate: 'k-rate' },
      { name: 'crush', defaultValue: 0, minValue: 0, maxValue: 1, automationRate: 'k-rate' }
    ];
  }

  constructor() {
    super();
    this.bufLen = Math.max(4096, Math.floor(sampleRate * 2.5));
    this.ringL = new Float32Array(this.bufLen);
    this.ringR = new Float32Array(this.bufLen);
    this.writePos = 0;

    // Beat-repeat state
    this.wasRolling = false;
    this.rollStartPos = 0;
    this.rollLen = 1024;
    this.rollPhase = 0;

    // Vinyl tape-stop state
    this.tapeSpeed = 1.0;
    this.tapeReadPos = 0;
    this.wasTapeStopped = false;

    // Bit-crush hold state
    this.crushCount = 0;
    this.holdL = 0;
    this.holdR = 0;
  }

  readInterp(buf, pos) {
    const len = this.bufLen;
    let p = pos % len;
    if (p < 0) p += len;
    const i0 = Math.floor(p);
    const i1 = (i0 + 1) % len;
    const f = p - i0;
    return buf[i0] + (buf[i1] - buf[i0]) * f;
  }

  process(inputs, outputs, parameters) {
    const input = inputs[0];
    const output = outputs[0];
    if (!output || output.length === 0) return true;
    const outL = output[0];
    const outR = output[1] || outL;
    const hasIn = input && input.length > 0 && input[0] && input[0].length > 0;
    const inL = hasIn ? input[0] : null;
    const inR = hasIn && input[1] ? input[1] : inL;

    const n = outL.length;
    const len = this.bufLen;
    const bpm = parameters.bpm[0] || 126;
    const rollBeats = parameters.rollBeats[0] || 0;
    const tapeTarget = parameters.tapeStop[0] >= 0.5;
    const crushAmt = parameters.crush[0] || 0;

    const rolling = rollBeats > 0.01;
    if (rolling && !this.wasRolling) {
      const sliceSamples = Math.round(sampleRate * (60.0 / Math.max(60, Math.min(200, bpm))) * rollBeats);
      this.rollLen = Math.max(256, Math.min(len - 512, sliceSamples));
      this.rollStartPos = (this.writePos - this.rollLen + len) % len;
      this.rollPhase = 0;
    }
    this.wasRolling = rolling;

    if (tapeTarget && !this.wasTapeStopped) {
      this.tapeReadPos = this.writePos;
    }
    this.wasTapeStopped = tapeTarget;

    const decimate = crushAmt > 0.02 ? Math.max(1, Math.floor(1 + crushAmt * 18)) : 1;
    const levels = crushAmt > 0.02 ? Math.max(4, Math.floor(64 * (1 - crushAmt * 0.85))) : 0;

    for (let i = 0; i < n; i++) {
      const sL = inL ? inL[i] : 0;
      const sR = inR ? inR[i] : 0;

      // Always keep recording dry live stream into the ring buffer
      this.ringL[this.writePos] = sL;
      this.ringR[this.writePos] = sR;

      let yL = sL;
      let yR = sR;

      // 1. Beat-repeat / Stutter Roll
      if (rolling) {
        const idx = (this.rollStartPos + this.rollPhase) % len;
        let g = 1.0;
        const fade = Math.min(96, Math.floor(this.rollLen * 0.12));
        if (this.rollPhase < fade) g = this.rollPhase / fade;
        else if (this.rollPhase > this.rollLen - fade) g = (this.rollLen - this.rollPhase) / fade;
        yL = this.ringL[idx] * g + sL * (1.0 - g) * 0.15;
        yR = this.ringR[idx] * g + sR * (1.0 - g) * 0.15;
        this.rollPhase = (this.rollPhase + 1) % this.rollLen;
      }

      // 2. Vinyl Tape-Stop Brake
      if (tapeTarget || this.tapeSpeed < 0.995) {
        const coeff = tapeTarget ? (1.0 / (sampleRate * 0.34)) : (1.0 / (sampleRate * 0.16));
        if (tapeTarget) {
          this.tapeSpeed = Math.max(0.0, this.tapeSpeed - coeff * 1.45);
        } else {
          this.tapeSpeed = Math.min(1.0, this.tapeSpeed + coeff * 2.2);
          // Drift read head toward write head on release
          const livePos = this.writePos;
          let err = livePos - this.tapeReadPos;
          if (err < -len * 0.5) err += len;
          if (err > len * 0.5) err -= len;
          this.tapeReadPos = (this.tapeReadPos + err * 0.015 + len) % len;
        }
        this.tapeReadPos = (this.tapeReadPos + this.tapeSpeed) % len;
        const tL = this.readInterp(this.ringL, this.tapeReadPos) * Math.min(1.0, this.tapeSpeed * 3.5);
        const tR = this.readInterp(this.ringR, this.tapeReadPos) * Math.min(1.0, this.tapeSpeed * 3.5);
        const mix = tapeTarget ? 1.0 : (1.0 - this.tapeSpeed);
        yL = yL * (1.0 - mix) + tL * mix;
        yR = yR * (1.0 - mix) + tR * mix;
      }

      // 3. Bit-Crush & Sample-and-Hold Glitch
      if (decimate > 1 || levels > 0) {
        if (this.crushCount <= 0) {
          let qL = yL;
          let qR = yR;
          if (levels > 0) {
            qL = Math.round(qL * levels) / levels;
            qR = Math.round(qR * levels) / levels;
          }
          this.holdL = qL;
          this.holdR = qR;
          this.crushCount = decimate;
        }
        this.crushCount--;
        yL = yL * (1.0 - crushAmt * 0.85) + this.holdL * (crushAmt * 0.85);
        yR = yR * (1.0 - crushAmt * 0.85) + this.holdR * (crushAmt * 0.85);
      }

      outL[i] = yL;
      outR[i] = yR;
      this.writePos = (this.writePos + 1) % len;
    }
    return true;
  }
}

registerProcessor('wub-dj-fx', WubDjFxProcessor);
`;

export interface DjFxState {
  filterSweep: number; // -1 (LPF) .. 0 (bypass) .. +1 (HPF)
  filterQ: number;     // 0.7 .. 12
  dubEcho: number;     // 0 .. 1
  reverbWash: number;  // 0 .. 1
  acidDrive: number;   // 0 .. 1
  crush: number;       // 0 .. 1
  rollBeats: number;   // 0 | 0.25 | 0.5 | 1.0
  tapeStop: boolean;
}

export const DEFAULT_DJ_FX_STATE: DjFxState = {
  filterSweep: 0,
  filterQ: 2.2,
  dubEcho: 0,
  reverbWash: 0,
  acidDrive: 0,
  crush: 0,
  rollBeats: 0,
  tapeStop: false,
};

let workletBlobUrl: string | null = null;

export async function attachDjFxWorklet(ctx: AudioContext): Promise<AudioWorkletNode | null> {
  if (typeof ctx.audioWorklet === "undefined" || typeof Blob === "undefined") {
    return null;
  }
  try {
    if (!workletBlobUrl) {
      const blob = new Blob([DJ_WORKLET_CODE], { type: "application/javascript" });
      workletBlobUrl = URL.createObjectURL(blob);
    }
    await ctx.audioWorklet.addModule(workletBlobUrl);
    return new AudioWorkletNode(ctx, "wub-dj-fx", {
      numberOfInputs: 1,
      numberOfOutputs: 1,
      outputChannelCount: [2],
    });
  } catch (err) {
    console.warn("[AudioEngine] DJ FX AudioWorklet fallback to direct pass:", err);
    return null;
  }
}
