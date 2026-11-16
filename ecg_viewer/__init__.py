"""ecg_viewer - load ECG recordings and turn them into figures.

    >>> from ecg_viewer import load, plot_record
    >>> rec = load("examples/synthetic_ecg.csv")
    >>> fig = plot_record(rec, duration=5)
    >>> fig.savefig("ecg.png")
"""
from .io import load, load_csv, load_npy, load_wfdb
from .plot import plot_record, save_figure
from .record import ECGFormatError, ECGRecord

__all__ = ["ECGRecord", "ECGFormatError", "load", "load_csv", "load_npy", "load_wfdb",
           "plot_record", "save_figure"]
__version__ = "0.1.0"
