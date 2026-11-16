"""Loaders: WFDB (PhysioNet), CSV/TXT and NumPy files -> :class:`ECGRecord`."""
from __future__ import annotations

import csv
import os
import re
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .record import ECGFormatError, ECGRecord

TIME_NAMES = {"time", "t", "time_s", "time(s)", "seconds", "sec", "s", "timestamp"}
TIME_MS_NAMES = {"time_ms", "time(ms)", "ms", "milliseconds"}
_UNIT_RE = re.compile(r"^\s*(.*?)\s*[\(\[]\s*([^\)\]]+?)\s*[\)\]]\s*$")


def load(path: str, sampling_rate: Optional[float] = None, *, pn_dir: Optional[str] = None,
         time_column: Optional[str] = None, unit: Optional[str] = None,
         lead_names: Optional[Sequence[str]] = None) -> ECGRecord:
    """Load an ECG file, choosing the reader from the file extension.

    * ``.hea`` / ``.dat`` / no extension -> WFDB record (``pn_dir`` reads it
      directly from a PhysioNet database, e.g. ``pn_dir="mitdb"``, ``path="100"``)
    * ``.csv`` / ``.tsv`` / ``.txt`` -> delimited text
    * ``.npy`` -> NumPy array (``sampling_rate`` should be given)
    """
    ext = os.path.splitext(path)[1].lower()
    if pn_dir is not None or ext in ("", ".hea", ".dat"):
        return load_wfdb(path, pn_dir=pn_dir)
    if ext in (".csv", ".tsv", ".txt"):
        return load_csv(path, sampling_rate=sampling_rate, time_column=time_column,
                        unit=unit, lead_names=lead_names)
    if ext == ".npy":
        return load_npy(path, sampling_rate=sampling_rate, unit=unit, lead_names=lead_names)
    raise ECGFormatError(f"unsupported file type {ext!r} (use WFDB, CSV/TXT or NPY)")


# ---------------------------------------------------------------------- WFDB
def load_wfdb(path: str, *, pn_dir: Optional[str] = None, channels: Optional[List[int]] = None,
              sampfrom: int = 0, sampto: Optional[int] = None) -> ECGRecord:
    """Read a WFDB record in physical units (as given by the header's gain/baseline)."""
    try:
        import wfdb
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise ECGFormatError("reading WFDB needs the 'wfdb' package: pip install wfdb") from exc
    base = path[:-4] if path.lower().endswith((".hea", ".dat")) else path
    if pn_dir is None and not os.path.exists(base + ".hea"):
        raise ECGFormatError(f"WFDB header not found: {base}.hea")
    try:
        rec = wfdb.rdrecord(base, sampfrom=sampfrom, sampto=sampto, channels=channels,
                            physical=True, pn_dir=pn_dir)
    except Exception as exc:  # wfdb raises many different types
        raise ECGFormatError(f"cannot read WFDB record {base!r}: {exc}") from exc
    if rec.p_signal is None:
        raise ECGFormatError("WFDB record has no physical signal")
    source = f"PhysioNet {pn_dir}/{os.path.basename(base)}" if pn_dir else os.path.abspath(base)
    return ECGRecord(
        signal=rec.p_signal,
        sampling_rate=rec.fs,
        lead_names=list(rec.sig_name or []),
        units=list(rec.units or []),
        record_id=rec.record_name,
        source=source,
        start_sample=sampfrom,
    )


# ---------------------------------------------------------------------- CSV
def _split_unit(header: str) -> Tuple[str, Optional[str]]:
    """``"MLII (mV)"`` -> ``("MLII", "mV")``; ``"MLII"`` -> ``("MLII", None)``."""
    m = _UNIT_RE.match(header)
    if m:
        return m.group(1), m.group(2)
    return header.strip(), None


def _is_number(text: str) -> bool:
    try:
        float(text)
        return True
    except ValueError:
        return False


def load_csv(path: str, *, sampling_rate: Optional[float] = None, time_column: Optional[str] = None,
             unit: Optional[str] = None, lead_names: Optional[Sequence[str]] = None,
             delimiter: Optional[str] = None) -> ECGRecord:
    """Read a delimited text file with one column per lead.

    The first row is treated as a header if it is not numeric. A time column is
    detected by name (``time``, ``t``, ``time_s``, ``time_ms`` ...) or given via
    ``time_column``; it is used to derive the sampling rate when none is given.
    Units can be written in the header as ``"MLII (mV)"`` or set with ``unit``.
    """
    if not os.path.exists(path):
        raise ECGFormatError(f"file not found: {path}")
    with open(path, newline="", encoding="utf-8-sig") as fh:
        sample = fh.read(4096)
        fh.seek(0)
        if delimiter is None:
            try:
                delimiter = csv.Sniffer().sniff(sample, delimiters=",;\t ").delimiter
            except csv.Error:
                delimiter = ","
        rows = [r for r in csv.reader(fh, delimiter=delimiter, skipinitialspace=True)
                if any(c.strip() for c in r)]
    if not rows:
        raise ECGFormatError(f"{path} is empty")

    header: Optional[List[str]] = None
    if not all(_is_number(c) for c in rows[0] if c.strip()):
        header, rows = [c.strip() for c in rows[0]], rows[1:]
    if not rows:
        raise ECGFormatError(f"{path} has a header but no data rows")

    width = len(header) if header else len(rows[0])
    data = np.full((len(rows), width), np.nan)
    first_data_line = 2 if header else 1
    for i, row in enumerate(rows):
        if len(row) != width:
            raise ECGFormatError(
                f"line {i + first_data_line}: expected {width} values, found {len(row)}")
        for j, cell in enumerate(row):
            cell = cell.strip()
            if cell == "" or cell.lower() in ("nan", "na"):
                continue
            try:
                data[i, j] = float(cell)
            except ValueError:
                raise ECGFormatError(
                    f"line {i + first_data_line}, column {j + 1}: {cell!r} is not a number") from None

    names = header or [f"lead {j + 1}" for j in range(width)]
    units: List[Optional[str]] = []
    clean_names: List[str] = []
    for n in names:
        base, u = _split_unit(n)
        clean_names.append(base)
        units.append(u)

    # ---- time column
    t_idx: Optional[int] = None
    t_scale = 1.0
    lowered = [n.lower().replace(" ", "") for n in names]
    if time_column is not None:
        key = time_column.lower().replace(" ", "")
        if key not in lowered and key not in [c.lower() for c in clean_names]:
            raise ECGFormatError(f"time column {time_column!r} not found")
        t_idx = lowered.index(key) if key in lowered else [c.lower() for c in clean_names].index(key)
    else:
        for j, n in enumerate(lowered):
            if n in TIME_NAMES or n in TIME_MS_NAMES or clean_names[j].lower() in TIME_NAMES:
                t_idx = j
                break
    warnings: List[str] = []
    if t_idx is not None:
        if lowered[t_idx] in TIME_MS_NAMES or (units[t_idx] or "").lower() == "ms":
            t_scale = 1e-3
        t = data[:, t_idx] * t_scale
        data = np.delete(data, t_idx, axis=1)
        del clean_names[t_idx]
        del units[t_idx]
        if data.shape[1] == 0:
            raise ECGFormatError("file has a time column but no signal columns")
        dt = np.diff(t[np.isfinite(t)])
        if sampling_rate is None:
            if dt.size == 0 or np.median(dt) <= 0:
                raise ECGFormatError("time column is not increasing")
            step = float(np.median(dt))
            if np.max(np.abs(dt - step)) > 0.01 * step:
                sampling_rate = round(1.0 / step, 3)
                warnings.append("time column is irregular: a uniform sampling rate "
                                f"of {sampling_rate:g} Hz (median step) is assumed")
            else:
                # whole span / number of steps: robust to time values rounded in the file
                tf = t[np.isfinite(t)]
                sampling_rate = round((tf.size - 1) / float(tf[-1] - tf[0]), 3)

    if lead_names:
        clean_names = list(lead_names)
    if unit:
        units = [u or unit for u in units]
    rec = ECGRecord(signal=data, sampling_rate=sampling_rate, lead_names=clean_names,
                    units=units, record_id=os.path.splitext(os.path.basename(path))[0],
                    source=os.path.abspath(path))
    rec.warnings = warnings + rec.warnings
    return rec


# ---------------------------------------------------------------------- NumPy
def load_npy(path: str, *, sampling_rate: Optional[float] = None, unit: Optional[str] = None,
             lead_names: Optional[Sequence[str]] = None) -> ECGRecord:
    """Read a ``.npy`` array shaped ``(N,)`` or ``(N, C)``."""
    if not os.path.exists(path):
        raise ECGFormatError(f"file not found: {path}")
    try:
        arr = np.load(path, allow_pickle=False)
    except ValueError as exc:
        raise ECGFormatError(f"cannot read {path}: {exc}") from exc
    n_leads = 1 if arr.ndim == 1 else (arr.shape[1] if arr.ndim == 2 else 0)
    return ECGRecord(signal=arr, sampling_rate=sampling_rate,
                     lead_names=list(lead_names or []),
                     units=[unit] * n_leads if (unit and n_leads) else [],
                     record_id=os.path.splitext(os.path.basename(path))[0],
                     source=os.path.abspath(path))
