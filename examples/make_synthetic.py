import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FS = 360          # Hz, as in the MIT-BIH Arrhythmia Database
SECONDS = 10


def gauss(t, mu, sigma, amp):
    return amp * np.exp(-0.5 * ((t - mu) / sigma) ** 2)


def beat(t, r, scale=1.0):
    return scale * (gauss(t, r - 0.20, 0.025, 0.12) + gauss(t, r - 0.035, 0.010, -0.10)
                    + gauss(t, r, 0.011, 1.10) + gauss(t, r + 0.035, 0.011, -0.25)
                    + gauss(t, r + 0.27, 0.045, 0.30))


def make(seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(SECONDS * FS) / FS
    r_peaks = np.cumsum(0.8 + 0.04 * rng.standard_normal(16)) - 0.4
    r_peaks = r_peaks[r_peaks < SECONDS - 0.3]
    lead1 = sum(beat(t, r) for r in r_peaks)
    lead2 = sum(beat(t, r, 0.6) for r in r_peaks) - 0.15 * sum(gauss(t, r, 0.02, 1.0) for r in r_peaks)
    wander = 0.05 * np.sin(2 * np.pi * 0.25 * t)
    lead1 = lead1 + wander + 0.01 * rng.standard_normal(t.size)
    lead2 = lead2 + 0.6 * wander + 0.01 * rng.standard_normal(t.size)
    return t, np.column_stack([lead1, lead2])


def main():
    t, sig = make()
    csv_path = os.path.join(HERE, "synthetic_ecg.csv")
    with open(csv_path, "w") as fh:
        fh.write("time (s),MLII (mV),V1 (mV)\n")
        for ti, (a, b) in zip(t, sig):
            fh.write(f"{ti:.6f},{a:.5f},{b:.5f}\n")
    print("wrote", csv_path)
    try:
        import wfdb
    except ImportError:
        print("wfdb not installed: skipping the WFDB example")
        return
    wfdb.wrsamp("synthetic", fs=FS, units=["mV", "mV"], sig_name=["MLII", "V1"],
                p_signal=sig, fmt=["212", "212"], adc_gain=[200.0, 200.0], baseline=[1024, 1024],
                write_dir=HERE, comments=["SYNTHETIC signal for demonstration, not a real recording"])
    print("wrote", os.path.join(HERE, "synthetic.hea"))


if __name__ == "__main__":
    main()
