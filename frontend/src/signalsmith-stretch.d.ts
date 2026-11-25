declare module "signalsmith-stretch" {
  export interface StretchSchedule {
    output?: number;
    outputTime?: number;
    active?: boolean;
    input?: number;
    rate?: number;
    semitones?: number;
    tonalityHz?: number;
    formantSemitones?: number;
    formantCompensation?: boolean;
    formantBaseHz?: number;
    loopStart?: number;
    loopEnd?: number;
  }

  export interface StretchNode extends AudioNode {
    schedule(opts: StretchSchedule): void;
    start(when?: number | StretchSchedule, offset?: number, duration?: number, rate?: number, semitones?: number): void;
    stop(when?: number): void;
    addBuffers(buffers: Float32Array[], transfer?: Transferable[]): Promise<number>;
    dropBuffers(toSeconds?: number): Promise<{ start: number; end: number }>;
    configure(opts: {
      blockMs?: number;
      intervalMs?: number;
      splitComputation?: boolean;
      preset?: "default" | "cheaper";
    }): void;
    latency(): number;
    inputTime: number;
    setUpdateInterval(seconds: number, callback?: (time: number) => void): void;
  }

  export default function SignalsmithStretch(
    audioContext: BaseAudioContext,
    options?: AudioWorkletNodeOptions,
  ): Promise<StretchNode>;
}
