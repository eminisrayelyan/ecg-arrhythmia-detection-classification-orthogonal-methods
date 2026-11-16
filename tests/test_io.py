import numpy as np
import pytest

from ecg_viewer import ECGFormatError, load, load_csv


def write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text)
    return str(p)


def test_csv_with_time_column_and_units(tmp_path, two_lead):
    fs, t, sig = two_lead
    lines = ["time (s),MLII (mV),V1 (mV)"] + [f"{a:.6f},{b:.6f},{c:.6f}" for a, (b, c) in zip(t, sig)]
    rec = load(write(tmp_path, "x.csv", "\n".join(lines)))
    assert rec.sampling_rate == pytest.approx(fs)
    assert rec.lead_names == ["MLII", "V1"]
    assert rec.units == ["mV", "mV"]
    np.testing.assert_allclose(rec.signal, sig, atol=1e-6)
    assert rec.warnings == []


def test_csv_time_in_milliseconds(tmp_path):
    text = "time_ms;II\n0;0.1\n2;0.2\n4;0.3\n6;0.4\n"
    rec = load_csv(write(tmp_path, "ms.csv", text), unit="mV")
    assert rec.sampling_rate == pytest.approx(500)
    assert rec.units == ["mV"]


def test_csv_without_header_needs_explicit_rate(tmp_path):
    path = write(tmp_path, "raw.txt", "1 2\n3 4\n5 6\n")
    rec = load(path)
    assert rec.sampling_rate is None and rec.n_leads == 2
    assert load(path, sampling_rate=128).sampling_rate == 128


def test_csv_bad_cell_names_the_line(tmp_path):
    path = write(tmp_path, "bad.csv", "time,II\n0,1\n0.01,oops\n")
    with pytest.raises(ECGFormatError, match=r"line 3, column 2"):
        load(path)


def test_csv_ragged_row(tmp_path):
    with pytest.raises(ECGFormatError, match="expected 2 values"):
        load(write(tmp_path, "r.csv", "time,II\n0,1\n0.1\n"))


def test_csv_irregular_time_is_flagged(tmp_path):
    rec = load(write(tmp_path, "irr.csv", "t,II\n0,1\n0.01,1\n0.02,1\n0.05,1\n0.06,1\n"))
    assert rec.sampling_rate == pytest.approx(100)
    assert any("irregular" in w for w in rec.warnings)


def test_npy(tmp_path, two_lead):
    fs, _, sig = two_lead
    p = tmp_path / "a.npy"
    np.save(p, sig)
    rec = load(str(p), sampling_rate=fs, unit="mV")
    assert rec.n_leads == 2 and rec.units == ["mV", "mV"]


def test_unsupported_and_missing(tmp_path):
    with pytest.raises(ECGFormatError, match="unsupported"):
        load(write(tmp_path, "a.xlsx", "x"))
    with pytest.raises(ECGFormatError, match="not found"):
        load(str(tmp_path / "nope.csv"))
    with pytest.raises(ECGFormatError, match="header not found"):
        load(str(tmp_path / "nope.hea"))


def test_wfdb_round_trip(tmp_path, two_lead):
    wfdb = pytest.importorskip("wfdb")
    fs, _, sig = two_lead
    wfdb.wrsamp("rec", fs=fs, units=["mV", "mV"], sig_name=["I", "II"], p_signal=sig,
                fmt=["16", "16"], adc_gain=[1000.0, 1000.0], baseline=[0, 0], write_dir=str(tmp_path))
    rec = load(str(tmp_path / "rec.hea"))
    assert rec.sampling_rate == fs
    assert rec.lead_names == ["I", "II"] and rec.units == ["mV", "mV"]
    np.testing.assert_allclose(rec.signal, sig, atol=1e-3)   # 16-bit ADC quantisation
