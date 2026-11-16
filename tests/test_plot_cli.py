import os

import numpy as np
import pytest

from ecg_viewer import ECGRecord, plot_record
from ecg_viewer.cli import main
from ecg_viewer.plot import decimate_minmax

EXAMPLE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "examples", "synthetic_ecg.csv")


def test_one_axis_per_lead_and_axis_labels(two_lead):
    fs, _, sig = two_lead
    fig = plot_record(ECGRecord(sig, fs, ["I", "II"], ["mV", "mV"]))
    assert len(fig.axes) == 2
    assert fig.axes[-1].get_xlabel() == "Time (s)"
    assert "I" in fig.axes[0].get_ylabel() and "mV" in fig.axes[0].get_ylabel()


def test_unknown_rate_uses_samples():
    fig = plot_record(ECGRecord(np.random.default_rng(0).normal(size=500), None, [], []))
    assert fig.axes[-1].get_xlabel() == "Sample index"
    texts = " ".join(t.get_text() for t in fig.texts)
    assert "sampling rate unknown" in texts


def test_paper_scale_is_25mm_per_second(two_lead):
    fs, _, sig = two_lead
    fig = plot_record(ECGRecord(sig, fs, ["I", "II"], ["mV", "mV"]), duration=2, paper=True)
    ax = fig.axes[0]
    width_in = ax.get_position().width * fig.get_figwidth()
    seconds = ax.get_xlim()[1] - ax.get_xlim()[0]
    assert width_in * 25.4 / seconds == pytest.approx(25.0, rel=1e-3)
    lo, hi = ax.get_ylim()
    height_in = ax.get_position().height * fig.get_figheight()
    assert height_in * 25.4 / (hi - lo) == pytest.approx(10.0, rel=1e-3)


def test_decimation_keeps_peaks():
    x = np.arange(100_000, dtype=float)
    y = np.zeros_like(x)
    y[54_321] = 5.0
    y[12_345] = -3.0
    xs, ys = decimate_minmax(x, y, max_points=1000)
    assert len(ys) <= 1000
    assert ys.max() == 5.0 and ys.min() == -3.0
    assert np.all(np.diff(xs) >= 0)


@pytest.mark.skipif(not os.path.exists(EXAMPLE), reason="run examples/make_synthetic.py first")
def test_cli_writes_image(tmp_path, capsys):
    out = tmp_path / "ecg.png"
    assert main([EXAMPLE, "--leads", "MLII", "--duration", "4", "-o", str(out)]) == 0
    assert out.stat().st_size > 10_000
    assert main([EXAMPLE, "--info"]) == 0
    assert "360 Hz" in capsys.readouterr().out


def test_cli_reports_errors(tmp_path, capsys):
    bad = tmp_path / "bad.csv"
    bad.write_text("t,II\n0,x\n")
    assert main([str(bad), "-o", str(tmp_path / "x.png")]) == 2
    assert "not a number" in capsys.readouterr().err
