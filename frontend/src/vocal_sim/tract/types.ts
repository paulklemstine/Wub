// Ported from MithrilMan/voice-workshop (Apache-2.0, Copyright 2026 Fabio Angela).
/** Articulator positions (mid-sagittal model, cm and normalised units). */
export interface Articulators {
  /** jaw opening, 0 = incisors together, 1 = about 17 mm incisor gap */
  jaw: number;
  /** tongue body forward (+) / backward (-) displacement, cm */
  tongueX: number;
  /** tongue body up (+) / down (-) displacement, cm */
  tongueY: number;
  /** tongue root advanced (+) / retracted (-), cm: widens or narrows the lower pharynx */
  tongueRoot: number;
  /** tongue tip forward (+) displacement relative to its rest position, cm */
  tipX: number;
  /** tongue tip up (+) displacement, cm */
  tipY: number;
  /** central groove of the tongue blade (sibilant channel), 0..1 */
  groove: number;
  /** margins of the tongue drawn in and lowered from the teeth so that air
   *  flows past its sides (lateral approximants), 0 = braced .. 1 */
  tongueSides: number;
  /** extra lip opening beyond what the jaw gives, cm (negative presses the lips) */
  lipOpen: number;
  /** lip rounding, 0 = spread, 1 = fully rounded */
  lipRound: number;
  /** lip protrusion, 0..1 cm; below zero the lips are spread back against the teeth (as for /s/) */
  lipProtrude: number;
  /** lower lip tucked under the upper incisors (labiodental), 0..1 */
  lipTuck: number;
  /** velum lowering (velopharyngeal port opening), 0 = closed, 1 = wide open */
  velum: number;
  /** larynx height, cm (+ raises the larynx and shortens the tract) */
  larynx: number;
}

/** Speaker anatomy (persistent "person" part of the filter). */
export interface Anatomy {
  /** overall size factor, 1 = adult male (about 17.5 cm tract) */
  scale: number;
  /** relative length of the pharynx (men have relatively longer pharynges) */
  pharynx: number;
  /** relative width of the pharynx (wide = dark/hollow, narrow = bright/thin) */
  pharynxWidth: number;
  /** extra leakage into the nose (adds to the velum), 0..1 */
  nasalLeak: number;
  /** nasal cycle: -1 left passage congested ... 0 equal ... +1 right congested */
  nasalCycle: number;
  /** opening of each nostril, 0 pinched ... 1 open */
  nostrilLeft: number;
  nostrilRight: number;
  /** air in the piriform fossae beside the larynx, 1 = empty (as always) ... 0 = filled (with water, as in Dang & Honda's experiment) */
  piriform: number;
}

/**
 * The incisors and the slit between them. These shape the obstacle that a
 * sibilant jet hits (Shadle 1985) and the narrowest passage at the teeth.
 */
export interface Teeth {
  /** lateral width of the incisal slit, cm */
  slitWidth: number;
  /** gap between the upper central incisors (diastema), cm */
  diastema: number;
  /** upper incisors forward of the lower ones, cm (added to the anatomical 0.1) */
  overjet: number;
  /** both incisors moved forward (+) toward the lips or back (-), cm */
  advance: number;
  /** upper incisor edge lowered (+) or raised (-), cm */
  upperLength: number;
  /** lower incisor edge raised (+) or lowered (-), cm */
  lowerLength: number;
  /** extra vertical opening of the slit independent of the jaw, cm */
  slitOpen: number;
}

export const DEFAULT_ARTICULATORS: Articulators = {
  jaw: 0.25,
  tongueX: 0,
  tongueY: 0,
  tongueRoot: 0,
  tipX: 0,
  tipY: 0,
  groove: 0,
  tongueSides: 0,
  lipOpen: 0.35,
  lipRound: 0,
  lipProtrude: 0,
  lipTuck: 0,
  velum: 0,
  larynx: 0,
};

export const DEFAULT_ANATOMY: Anatomy = { scale: 1, pharynx: 1, pharynxWidth: 1, nasalLeak: 0, nasalCycle: 0, nostrilLeft: 1, nostrilRight: 1, piriform: 1 };

export const DEFAULT_TEETH: Teeth = {
  slitWidth: 2.2,
  diastema: 0,
  overjet: 0,
  advance: 0,
  upperLength: 0,
  lowerLength: 0,
  slitOpen: 0,
};

export const ARTICULATOR_KEYS: (keyof Articulators)[] = [
  'jaw', 'tongueX', 'tongueY', 'tongueRoot', 'tipX', 'tipY', 'groove', 'tongueSides', 'lipOpen', 'lipRound', 'lipProtrude', 'lipTuck', 'velum', 'larynx',
];

/** Allowed ranges of the articulators (used by the UI and the sequencer). */
export const ARTICULATOR_RANGES: Record<keyof Articulators, [number, number]> = {
  jaw: [0, 1],
  tongueX: [-1.6, 1.4],
  tongueY: [-1.6, 1.6],
  tongueRoot: [-1.2, 1.2],
  tipX: [-1.2, 1.4],
  tipY: [-0.6, 2.2],
  groove: [0, 1],
  tongueSides: [0, 1],
  lipOpen: [-0.4, 1.2],
  lipRound: [0, 1],
  lipProtrude: [-0.6, 1],
  lipTuck: [0, 1],
  velum: [0, 1],
  larynx: [-1.5, 1.2],
};
