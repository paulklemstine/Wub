"""Pydantic schemas for Wub track analysis.

Strictly mirrors Section 4.2 `track_analysis.json`.
"""
from __future__ import annotations

from typing import Dict, List, Literal

from pydantic import BaseModel, Field, field_validator

StemName = Literal["drums", "bass", "vocals", "other"]
STEMS: List[str] = ["drums", "bass", "vocals", "other"]


class Segment(BaseModel):
    id: str
    track_id: str
    stem: StemName
    start: float = Field(ge=0.0)
    duration: float = Field(gt=0.0)
    is_silent: bool = False
    loudness_start: float
    loudness_max: float
    loudness_max_time: float = Field(ge=0.0)
    pitches: List[float] = Field(min_length=12, max_length=12)
    timbre: List[float] = Field(min_length=12, max_length=12)
    spectral_centroid: float = Field(ge=0.0)
    zero_crossing_start_offset: float = Field(ge=0.0)

    @field_validator("pitches", "timbre")
    @classmethod
    def _finite(cls, v: List[float]) -> List[float]:
        import math

        for x in v:
            if not math.isfinite(x):
                raise ValueError("feature vectors must be finite")
        return v


class Tatum(BaseModel):
    id: str
    index: int = Field(ge=0)
    bar_position: int = Field(ge=0)
    beat_position: int = Field(ge=0)
    tatum_position: int = Field(ge=0, le=3)
    start: float = Field(ge=0.0)
    duration: float = Field(gt=0.0)
    confidence: float = Field(ge=0.0, le=1.0)
    stem_segments: Dict[StemName, List[str]] = Field(default_factory=dict)

    @field_validator("stem_segments")
    @classmethod
    def _stems_valid(cls, v: Dict[str, List[str]]) -> Dict[str, List[str]]:
        for k in v:
            if k not in ("drums", "bass", "vocals", "other"):
                raise ValueError(f"unknown stem {k}")
        # ensure all four keys present
        for s in ("drums", "bass", "vocals", "other"):
            v.setdefault(s, [])  # type: ignore[typeddict-item]
        return v


class TrackAnalysis(BaseModel):
    track_id: str
    filename: str
    bpm: float = Field(gt=0.0)
    key: str
    duration: float = Field(gt=0.0)
    sample_rate: int = Field(gt=0)
    stems_available: List[StemName]
    tatums: List[Tatum]
    segments: List[Segment]

    @field_validator("stems_available")
    @classmethod
    def _nonempty(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError("stems_available must be non-empty")
        return v
