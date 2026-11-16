import numpy as np
import pytest

from ecg_viewer import ECGFormatError, ECGRecord


def test_1d_signal_becomes_single_lead():
    rec = ECGRecord(np.arange(10.0), 5, ["II"], ["mV"])
    assert rec.signal.shape == (10, 1)
    assert rec.duration == pytest.approx(2.0)


@pytest.mark.parametrize("bad", [np.array([]), np.zeros((3, 2, 2)), np.array(["a", "b"])])
def test_bad_signals_are_rejected(bad):
    with pytest.raises(ECGFormatError):
        ECGRecord(bad, 100, [], [])


@pytest.mark.parametrize("fs", [0, -5, float("nan")])
def test_sampling_rate_must_be_positive(fs):
    with pytest.raises(ECGFormatError):
        ECGRecord(np.zeros(5), fs, ["II"], ["mV"])


def test_lead_and_unit_counts_must_match():
    with pytest.raises(ECGFormatError):
        ECGRecord(np.zeros((5, 2)), 100, ["I"], ["mV", "mV"])
    with pytest.raises(ECGFormatError):
        ECGRecord(np.zeros((5, 2)), 100, ["I", "II"], ["mV"])


def test_missing_metadata_is_reported_not_guessed():
    rec = ECGRecord(np.zeros((5, 2)), None, [], [])
    assert rec.lead_names == ["lead 1", "lead 2"]
    assert rec.units == [None, None]
    text = " ".join(rec.warnings)
    assert "sampling rate unknown" in text and "unit unknown" in text
    np.testing.assert_array_equal(rec.time_axis(), np.arange(5))


def test_time_axis_and_window(two_lead):
    fs, t, sig = two_lead
    rec = ECGRecord(sig, fs, ["I", "II"], ["mV", "mV"])
    np.testing.assert_allclose(rec.time_axis(), t)
    part = rec.select(leads=["ii"], start=1.0, duration=0.5)
    assert part.lead_names == ["II"]
    assert part.n_samples == 125
    assert part.time_axis()[0] == pytest.approx(1.0)        # keeps its true position
    np.testing.assert_allclose(part.signal[:, 0], sig[250:375, 1])
    assert rec.select(leads=["2"]).lead_names == ["II"]      # 1-based numbers work too


def test_bad_window_requests():
    rec = ECGRecord(np.zeros(100), 100, ["II"], ["mV"])
    with pytest.raises(ECGFormatError):
        rec.select(start=5)
    with pytest.raises(ECGFormatError):
        rec.select(duration=0)
    with pytest.raises(ECGFormatError):
        rec.select(leads=["V9"])


def test_nan_samples_are_kept_as_gaps():
    sig = np.array([0.0, np.nan, 1.0])
    rec = ECGRecord(sig, 10, ["II"], ["mV"])
    assert np.isnan(rec.signal[1, 0])
    assert any("non-finite" in w for w in rec.warnings)
