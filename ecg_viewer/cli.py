"""Command line: ``ecg-view FILE [options]`` -> image (or an interactive window)."""
from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional

from .io import load
from .record import ECGFormatError


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ecg-view",
        description="Load an ECG recording (WFDB, CSV/TXT, NPY) and draw it on an ECG grid.",
        epilog="Examples:\n"
               "  ecg-view examples/synthetic_ecg.csv -o ecg.png\n"
               "  ecg-view examples/synthetic --leads MLII --start 2 --duration 5 --paper -o ecg.pdf\n"
               "  ecg-view 100 --physionet mitdb --duration 10 -o mitdb100.png\n"
               "  ecg-view signal.npy --fs 500 --unit mV --info",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", help="file path, or a record name when --physionet is used")
    src = p.add_argument_group("input")
    src.add_argument("--physionet", metavar="DB",
                     help="read the record from a PhysioNet database (e.g. mitdb); needs internet")
    src.add_argument("--fs", type=float, help="sampling rate in Hz (CSV/NPY without a time column)")
    src.add_argument("--time-column", help="name of the time column in a CSV file")
    src.add_argument("--unit", help="unit for leads that do not state one (e.g. mV)")
    src.add_argument("--lead-names", help="comma-separated lead names (CSV/NPY)")
    view = p.add_argument_group("view")
    view.add_argument("--leads", help="comma-separated leads to show, by name or number")
    view.add_argument("--start", type=float, default=0.0, help="window start in seconds (default 0)")
    view.add_argument("--duration", type=float, help="window length in seconds (default: whole record)")
    view.add_argument("--no-grid", action="store_true", help="plain axes instead of the ECG grid")
    view.add_argument("--paper", action="store_true",
                      help="true scale: 25 mm/s and 10 mm/mV when printed at 100%%")
    view.add_argument("--title", help="custom figure title")
    out = p.add_argument_group("output")
    out.add_argument("-o", "--out", help="save to PNG, PDF or SVG instead of opening a window")
    out.add_argument("--dpi", type=int, default=200, help="resolution for PNG (default 200)")
    out.add_argument("--info", action="store_true", help="print the record summary and exit")
    return p


def _split(text: Optional[str]) -> Optional[List[str]]:
    return [x.strip() for x in text.split(",") if x.strip()] if text else None


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        rec = load(args.input, sampling_rate=args.fs, pn_dir=args.physionet,
                   time_column=args.time_column, unit=args.unit,
                   lead_names=_split(args.lead_names))
        if args.info:
            print(rec.summary())
            return 0

        if args.out:
            import matplotlib
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from .plot import plot_record, save_figure

        fig = plot_record(rec, leads=_split(args.leads), start=args.start,
                          duration=args.duration, grid=not args.no_grid,
                          paper=args.paper, title=args.title,
                          figure_factory=plt.figure)
    except ECGFormatError as exc:
        print(f"ecg-view: error: {exc}", file=sys.stderr)
        return 2

    for w in rec.warnings:
        print(f"ecg-view: note: {w}", file=sys.stderr)

    if args.out:
        folder = os.path.dirname(os.path.abspath(args.out))
        os.makedirs(folder, exist_ok=True)
        save_figure(fig, args.out, dpi=args.dpi)
        plt.close(fig)
        print(f"saved {args.out}")
        return 0

    plt.show()
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
