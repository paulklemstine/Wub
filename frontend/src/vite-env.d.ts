/// <reference types="vite/client" />

interface ImportMetaEnv {
  /**
   * Absolute base URL for kit data (manifest, taste scores, joint ranker,
   * sidecars, and the opus kit bank).  CI sets this to
   * `https://raw.githubusercontent.com/paulklemstine/Wub/<sha>/frontend/public/data/kits`
   * so the 708 MB / 20k+ file bank is served straight from the git repo and
   * never goes through the Pages sync.  Local dev leaves it unset and uses
   * the same-origin `data/kits`.
   */
  readonly VITE_KIT_BASE?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
