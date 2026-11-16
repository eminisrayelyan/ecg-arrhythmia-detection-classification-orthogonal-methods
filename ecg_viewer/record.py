"""The in-memory ECG record that every loader produces and the plotter consumes.

The record is format-independent: a WFDB record, a CSV file and a NumPy array
all end up as the same :class:`ECGRecord`, and it is validated once, here.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import List, Optional, Sequence

import numpy as np


class ECGFormatError(ValueError):
    """Raised when input data cannot be interpreted as an ECG record."""


@dataclass
class ECGRecord:
    """An ECG recording with the metadata needed to draw it honestly.

    Attributes:
        signal: samples, shape ``(N, C)`` (N samples, C leads). A 1-D array is
            accepted and treated as a single lead.
        sampling_rate: samples per second (Hz), or ``None`` if unknown. When it
            is unknown the time axis is shown in samples, never guessed.
        lead_names: one name per lead.
        units: one physical unit per lead (e.g. ``"mV"``), ``None`` if unknown.
        record_id: name of the record, used in titles and file names.
        source: where the data came from (path or database).
        start_sample: index of the first sample relative to the original file,
            so that a cut-out segment keeps its true time position.
        warnings: human-readable notes about anything that was missing or odd.
    """

    signal: np.ndarray
    sampling_rate: Optional[float]
    lead_names: List[str]
    units: List[Optional[str]]
    record_id: str = "record"
    source: str = ""
    start_sample: int = 0
    warnings: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        sig = np.asarray(self.signal)
        if sig.dtype.kind not in "biuf":
            try:
                sig = sig.astype(float)
            except (TypeError, ValueError) as exc:
                raise ECGFormatError(f"signal is not numeric ({sig.dtype})") from exc
        if sig.ndim == 1:
            sig = sig.reshape(-1, 1)
        if sig.ndim != 2:
            raise ECGFormatError(f"signal must be 1-D or 2-D, got shape {sig.shape}")
        if sig.shape[0] == 0 or sig.shape[1] == 0:
            raise ECGFormatError(f"signal is empty (shape {sig.shape})")
        sig = sig.astype(float, copy=False)
        if not np.isfinite(sig).any():
            raise ECGFormatError("signal contains no finite values")
        n_bad = int((~np.isfinite(sig)).sum())
        if n_bad:
            self._warn(f"{n_bad} missing or non-finite sample(s) are left as gaps")
        self.signal = sig

        n_leads = sig.shape[1]
        if self.sampling_rate is None:
            self._warn("sampling rate unknown: time axis is shown in samples")
        else:
            fs = float(self.sampling_rate)
            if not np.isfinite(fs) or fs <= 0:
                raise ECGFormatError(f"sampling rate must be positive, got {self.sampling_rate!r}")
            self.sampling_rate = fs

        names = list(self.lead_names or [])
        if not names:
            names = [f"lead {i + 1}" for i in range(n_leads)]
            self._warn("lead names missing: leads are numbered")
        if len(names) != n_leads:
            raise ECGFormatError(f"{len(names)} lead name(s) for {n_leads} lead(s)")
        self.lead_names = [str(n) for n in names]

        units = list(self.units) if self.units else [None] * n_leads
        if len(units) != n_leads:
            raise ECGFormatError(f"{len(units)} unit(s) for {n_leads} lead(s)")
        units = [u.strip() if isinstance(u, str) and u.strip() else None for u in units]
        missing = [n for n, u in zip(self.lead_names, units) if u is None]
        if missing:
            self._warn("unit unknown for " + ", ".join(missing) + ": amplitude shown as raw values")
        self.units = units

        if self.start_sample < 0:
            raise ECGFormatError("start_sample must be >= 0")

    def _warn(self, msg: str) -> None:
        if msg not in self.warnings:
            self.warnings.append(msg)

    # ------------------------------------------------------------------ properties
    @property
    def n_samples(self) -> int:
        return int(self.signal.shape[0])

    @property
    def n_leads(self) -> int:
        return int(self.signal.shape[1])

    @property
    def duration(self) -> Optional[float]:
        """Length in seconds, or ``None`` if the sampling rate is unknown."""
        if self.sampling_rate is None:
            return None
        return self.n_samples / self.sampling_rate

    def time_axis(self) -> np.ndarray:
        """Seconds from the start of the original recording (samples if fs unknown)."""
        n = np.arange(self.start_sample, self.start_sample + self.n_samples, dtype=float)
        return n if self.sampling_rate is None else n / self.sampling_rate

    # ------------------------------------------------------------------ selection
    def lead_indices(self, leads: Optional[Sequence[str]]) -> List[int]:
        """Map lead names (or 1-based numbers given as text) to column indices."""
        if not leads:
            return list(range(self.n_leads))
        idx = []
        lower = [n.lower() for n in self.lead_names]
        for item in leads:
            key = str(item).strip()
            if key.lower() in lower:
                idx.append(lower.index(key.lower()))
            elif key.isdigit() and 1 <= int(key) <= self.n_leads:
                idx.append(int(key) - 1)
            else:
                raise ECGFormatError(
                    f"unknown lead {key!r}; available: {', '.join(self.lead_names)}")
        return idx

    def select(self, leads: Optional[Sequence[str]] = None, start: float = 0.0,
               duration: Optional[float] = None) -> "ECGRecord":
        """Return a new record with chosen leads and a time window.

        ``start`` and ``duration`` are in seconds relative to the first sample of
        this record (in samples if the sampling rate is unknown).
        """
        cols = self.lead_indices(leads)
        scale = 1.0 if self.sampling_rate is None else self.sampling_rate
        if start < 0:
            raise ECGFormatError("start must be >= 0")
        a = int(round(start * scale))
        if a >= self.n_samples:
            raise ECGFormatError(
                f"start is beyond the end of the record ({self._fmt_len()})")
        if duration is not None and duration <= 0:
            raise ECGFormatError("duration must be > 0")
        b = self.n_samples if duration is None else min(self.n_samples, a + max(1, int(round(duration * scale))))
        return replace(
            self,
            signal=self.signal[a:b, cols],
            lead_names=[self.lead_names[i] for i in cols],
            units=[self.units[i] for i in cols],
            start_sample=self.start_sample + a,
            warnings=list(self.warnings),
        )

    def _fmt_len(self) -> str:
        if self.sampling_rate is None:
            return f"{self.n_samples} samples"
        return f"{self.duration:.2f} s"

    def summary(self) -> str:
        """Plain-text description used by ``ecg-view --info``."""
        fs = "unknown" if self.sampling_rate is None else f"{self.sampling_rate:g} Hz"
        lines = [
            f"record:   {self.record_id}",
            f"source:   {self.source}",
            f"length:   {self.n_samples} samples ({self._fmt_len()})",
            f"sampling: {fs}",
            "leads:    " + ", ".join(
                f"{n} [{u or '?'}]" for n, u in zip(self.lead_names, self.units)),
        ]
        lines += [f"note:     {w}" for w in self.warnings]
        return "\n".join(lines)
