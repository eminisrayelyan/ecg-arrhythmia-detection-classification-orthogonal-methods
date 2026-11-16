"""Turn an :class:`ECGRecord` into a figure (one row per lead, ECG paper grid)."""
from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter, MaxNLocator, MultipleLocator

from .record import ECGRecord

INK = "#14243A"
ACCENT = "#087F8C"
GRID_MAJOR = "#F0B8B3"
GRID_MINOR = "#FBE3E0"
PAPER = "#FFFBFA"
NOTE = "#6B7784"
DISCLAIMER = "Research prototype for visualisation only - not a diagnostic tool."

# millivolts per one unit of the signal, for drawing the standard grid
_MV_PER_UNIT = {"mv": 1.0, "millivolt": 1.0, "millivolts": 1.0,
                "uv": 1e-3, "µv": 1e-3, "μv": 1e-3, "microvolt": 1e-3, "microvolts": 1e-3,
                "v": 1e3, "volt": 1e3, "volts": 1e3}

MM_PER_INCH = 25.4
PAPER_SPEED_MM_S = 25.0   # standard paper speed
PAPER_GAIN_MM_MV = 10.0   # standard gain


def mv_per_unit(unit: Optional[str]) -> Optional[float]:
    """How many millivolts one unit of ``unit`` is, or ``None`` if not a voltage."""
    if not unit:
        return None
    return _MV_PER_UNIT.get(unit.strip().lower())


def decimate_minmax(x: np.ndarray, y: np.ndarray, max_points: int = 20000) -> Tuple[np.ndarray, np.ndarray]:
    """Reduce a long trace for drawing while keeping every peak.

    Each bucket of samples is replaced by its minimum and maximum (in time
    order), so narrow QRS spikes stay visible even for hour-long records.
    """
    n = len(y)
    if n <= max_points:
        return x, y
    n_buckets = max_points // 2
    edges = np.linspace(0, n, n_buckets + 1).astype(int)
    xs: List[float] = []
    ys: List[float] = []
    for a, b in zip(edges[:-1], edges[1:]):
        seg = y[a:b]
        finite = np.isfinite(seg)
        if not finite.any():
            xs += [x[a], x[b - 1]]
            ys += [np.nan, np.nan]
            continue
        i_min = a + int(np.nanargmin(seg))
        i_max = a + int(np.nanargmax(seg))
        for i in sorted((i_min, i_max)):
            xs.append(x[i])
            ys.append(y[i])
    return np.asarray(xs), np.asarray(ys)


def _ylim(y: np.ndarray, step: Optional[float]) -> Tuple[float, float]:
    finite = y[np.isfinite(y)]
    lo, hi = (float(finite.min()), float(finite.max())) if finite.size else (-1.0, 1.0)
    if hi - lo < 1e-12:
        lo, hi = lo - 1.0, hi + 1.0
    pad = 0.08 * (hi - lo)
    lo, hi = lo - pad, hi + pad
    if step:
        lo, hi = np.floor(lo / step) * step, np.ceil(hi / step) * step
    return lo, hi


def _x_grid_steps(span: float) -> Tuple[float, float, float]:
    """(major, minor, label) spacing in seconds; coarser for long windows."""
    if span <= 30:
        return 0.2, 0.04, 1.0
    if span <= 120:
        return 1.0, 0.2, 5.0
    return 5.0, 1.0, 30.0


def _draw_ecg_grid(ax, mv_unit: float, span: float) -> None:
    """Standard ECG paper: 0.04 s x 0.1 mV small squares, 0.2 s x 0.5 mV large ones.

    Only whole seconds are labelled on the time axis. For windows longer than
    30 s the time grid is coarsened so it stays readable.
    """
    major, minor, label = _x_grid_steps(span)
    ax.set_facecolor(PAPER)
    ax.xaxis.set_major_locator(MultipleLocator(major))
    ax.xaxis.set_minor_locator(MultipleLocator(minor))
    ax.xaxis.set_major_formatter(FuncFormatter(
        lambda v, _pos: f"{v:g}" if abs(v / label - round(v / label)) < 1e-6 else ""))
    ax.yaxis.set_major_locator(MultipleLocator(0.5 / mv_unit))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1 / mv_unit))
    ax.grid(which="major", color=GRID_MAJOR, linewidth=0.8)
    ax.grid(which="minor", color=GRID_MINOR, linewidth=0.5)
    ax.tick_params(which="both", length=0)
    ax.set_axisbelow(True)


def _draw_plain_grid(ax) -> None:
    ax.grid(which="major", color="#E3E8EC", linewidth=0.7)
    ax.set_axisbelow(True)


def plot_record(rec: ECGRecord, *, leads: Optional[Sequence[str]] = None, start: float = 0.0,
                duration: Optional[float] = None, grid: bool = True, paper: bool = False,
                title: Optional[str] = None, width: float = 14.0, lead_height: float = 2.2,
                max_points: int = 20000, notes: bool = True,
                figure_factory: Callable[..., Figure] = Figure) -> Figure:
    """Draw a record (or a window of it) and return the matplotlib Figure.

    Args:
        rec: the record to draw.
        leads: lead names or 1-based numbers to show (default: all).
        start, duration: time window in seconds (samples if fs is unknown).
        grid: draw the standard ECG paper grid when the signal is a voltage and
            the sampling rate is known; otherwise a light neutral grid.
        paper: size the figure so that, printed at 100 %, it is at 25 mm/s and
            10 mm/mV (needs a known sampling rate and a voltage unit).
        title: figure title (default: record name, window and sampling rate).
        width, lead_height: figure size in inches when ``paper`` is False.
        max_points: traces longer than this are min/max-decimated for drawing.
        notes: print warnings about missing metadata and a disclaimer below the plot.
        figure_factory: how to create the figure; pass ``matplotlib.pyplot.figure``
            to get a figure that ``pyplot.show()`` can display.
    """
    seg = rec.select(leads=leads, start=start, duration=duration)
    t = seg.time_axis()
    known_fs = seg.sampling_rate is not None
    mv_units = [mv_per_unit(u) for u in seg.units]

    # ---- layout
    lims = []
    for j in range(seg.n_leads):
        step = (0.5 / mv_units[j]) if (grid and known_fs and mv_units[j]) else None
        lims.append(_ylim(seg.signal[:, j], step))
    note_lines = (seg.warnings + [DISCLAIMER]) if notes else []
    paper_ok = paper and known_fs and all(mv_units)
    if paper and not paper_ok:
        note_lines.insert(0, "paper scale needs a known sampling rate and voltage units: "
                             "drawn at screen size instead")

    top_in, bottom_in = 0.55, 0.55 + 0.17 * len(note_lines)
    left_in, right_in, gap_in = 1.15, 0.3, 0.12
    if paper_ok:
        span_s = t[-1] - t[0] + 1.0 / seg.sampling_rate
        plot_w = span_s * PAPER_SPEED_MM_S / MM_PER_INCH
        heights = [(hi - lo) * mv * PAPER_GAIN_MM_MV / MM_PER_INCH
                   for (lo, hi), mv in zip(lims, mv_units)]
    else:
        plot_w = max(width - left_in - right_in, 2.0)
        heights = [lead_height] * seg.n_leads
    fig_w = left_in + plot_w + right_in
    if fig_w < 8.0:  # keep room for the title and notes; the plot area keeps its exact size
        right_in += 8.0 - fig_w
        fig_w = 8.0
    fig_h = top_in + sum(heights) + gap_in * (seg.n_leads - 1) + bottom_in
    fig = figure_factory(figsize=(fig_w, fig_h), facecolor="white")

    axes = []
    y_cursor = fig_h - top_in
    for j in range(seg.n_leads):
        y_cursor -= heights[j]
        ax = fig.add_axes([left_in / fig_w, y_cursor / fig_h, plot_w / fig_w, heights[j] / fig_h],
                          sharex=axes[0] if axes else None)
        y_cursor -= gap_in
        axes.append(ax)

        x, y = decimate_minmax(t, seg.signal[:, j], max_points=max_points)
        ax.plot(x, y, color=INK, linewidth=0.9, solid_joinstyle="round")
        if known_fs:
            ax.set_xlim(t[0], t[-1] + 1.0 / seg.sampling_rate)
        else:
            ax.set_xlim(t[0], t[-1])
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_ylim(*lims[j])

        use_ecg_grid = grid and known_fs and mv_units[j] is not None
        if use_ecg_grid:
            _draw_ecg_grid(ax, mv_units[j], span=float(t[-1] - t[0]))
        elif grid:
            _draw_plain_grid(ax)

        unit = seg.units[j] or "raw"
        ax.set_ylabel(f"{seg.lead_names[j]}\n({unit})", color=INK, rotation=0,
                      ha="right", va="center", labelpad=8, fontsize=10)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#B8C2CC")
        ax.tick_params(colors=NOTE, labelsize=8)
        if j < seg.n_leads - 1:
            ax.tick_params(labelbottom=False)

    last = axes[-1]
    last.set_xlabel("Time (s)" if known_fs else "Sample index", color=INK, fontsize=10)

    if title is None:
        if known_fs:
            title = (f"{seg.record_id}  ·  {t[0]:.2f}–{t[-1] + 1 / seg.sampling_rate:.2f} s"
                     f"  ·  {seg.sampling_rate:g} Hz")
        else:
            title = f"{seg.record_id}  ·  samples {int(t[0])}–{int(t[-1])}"
        if paper_ok:
            title += "  ·  25 mm/s, 10 mm/mV"
    fig.text(left_in / fig_w, 1 - 0.3 / fig_h, title, color=INK, fontsize=12,
             fontweight="bold", va="center")

    for k, line in enumerate(note_lines):
        fig.text(left_in / fig_w, (0.12 + 0.17 * (len(note_lines) - 1 - k)) / fig_h,
                 ("Note: " if line != DISCLAIMER else "") + line,
                 color=NOTE, fontsize=8, va="bottom")
    return fig


def save_figure(fig: Figure, path: str, dpi: int = 200) -> str:
    """Save as PNG, PDF or SVG (chosen by extension) and return the path."""
    fig.savefig(path, dpi=dpi, facecolor=fig.get_facecolor())
    return path
