// Ported from MithrilMan/voice-workshop (Apache-2.0, Copyright 2026 Fabio Angela).
/**
 * Physical constants, CGS units (cm, g, s), as customary in speech acoustics
 * (Flanagan 1972; Stevens 1998). Pressures are in dyn/cm^2, volume velocities
 * in cm^3/s, acoustic impedances in dyn*s/cm^5 ("acoustic ohms").
 */

/** Speed of sound in warm, humid air of the vocal tract (cm/s). 35,000 cm/s makes
 *  the quarter-wave resonances of a 17.5 cm tube fall exactly on 500/1500/2500 Hz. */
export const C_SOUND = 35000;

/** Air density at body temperature (g/cm^3). */
export const RHO = 1.14e-3;

/** Shear viscosity of air (poise = dyn*s/cm^2). */
export const MU = 1.86e-4;

/** Kinematic viscosity (cm^2/s), used for the Reynolds number. */
export const NU = MU / RHO;

/** Ratio of specific heats. */
export const GAMMA_AIR = 1.4;

/** Heat conduction coefficient (cal / (cm s degC)). */
export const LAMBDA_HEAT = 5.5e-5;

/** Specific heat at constant pressure (cal / (g degC)). */
export const CP_AIR = 0.24;

/** Yielding-wall parameters per unit area (Ishizaka, French & Flanagan 1975):
 *  mass (g/cm^2), resistance (dyn s/cm^3), stiffness (dyn/cm^3). */
export const WALL_MASS = 1.5;
export const WALL_RES = 1600;
export const WALL_STIFF = 3.0e5;

/** 1 cm of water column in dyn/cm^2. */
export const CMH2O = 980.665;

/** Characteristic acoustic impedance of a tube of area A (dyn s/cm^5). */
export function z0(area: number): number {
  return (RHO * C_SOUND) / area;
}

/** Smallest area used in the acoustic models (cm^2); a "closed" section. */
export const AREA_CLOSED = 1e-4;
