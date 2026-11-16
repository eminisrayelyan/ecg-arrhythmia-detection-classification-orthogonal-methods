# ecg-viewer

Loads an ECG recording and turns it into a figure: one row per lead, with the standard
ECG paper grid. That is all it does. There is no filtering, no beat detection and no
diagnosis. It is the first prototype described in the diploma work, covering
**import and visualisation** only.

> Research prototype for visualisation only. It is not a medical device and must not be
> used for diagnosis.

## What it reads

| Format | How | Notes |
|---|---|---|
| WFDB (PhysioNet) `.hea` + `.dat` | `ecg-view examples/synthetic` | Physical units come from the header (gain, baseline). |
| PhysioNet online | `ecg-view 100 --physionet mitdb` | Downloads the record from PhysioNet (internet needed). |
| CSV / TSV / TXT | `ecg-view file.csv` | Header optional. Units are read from `MLII (mV)` or `MLII [mV]`. A time column (`time`, `t`, `time_s`, `time_ms` …) gives the sampling rate. |
| NumPy `.npy` | `ecg-view a.npy --fs 500 --unit mV` | Shape `(N,)` or `(N, leads)`. |

Every loader produces the same `ECGRecord`: `signal [N × C]`, `sampling_rate`,
`lead_names`, `units`, `record_id`, `source` and `start_sample`. Each record is checked
once:

* Non-numeric or empty data, a non-positive sampling rate, a mismatched number of names
  or units, and ragged CSV rows are **rejected**. The error message names the line and
  column.
* A missing sampling rate or unit is **never guessed**:
  * the time axis is shown in samples, or the amplitude as raw values;
  * a note is printed under the figure.
* Missing samples (`NaN`) stay as gaps in the trace.

## Install

```bash
python3 -m venv .venv
```
```bash
.venv/bin/python -m pip install --upgrade pip
```
```bash
.venv/bin/pip install -e ".[test]"
```

The pip upgrade is needed with the Python that ships with macOS: its pip (21.x) is too
old to install a `pyproject.toml` project in editable mode.

## Use

Create the synthetic example data (a generated signal, **not** a real recording):

```bash
.venv/bin/python examples/make_synthetic.py
```

Draw it:

```bash
.venv/bin/ecg-view examples/synthetic_ecg.csv -o ecg.png
```

Lead MLII only, seconds 2 to 7, at true paper scale (25 mm/s, 10 mm/mV) as a PDF:

```bash
.venv/bin/ecg-view examples/synthetic --leads MLII --start 2 --duration 5 --paper -o ecg.pdf
```

The first 10 s of MIT-BIH record 100, straight from PhysioNet:

```bash
.venv/bin/ecg-view 100 --physionet mitdb --duration 10 -o mitdb100.png
```

Print what was loaded without drawing:

```bash
.venv/bin/ecg-view examples/synthetic_ecg.csv --info
```

Without `-o`, the figure opens in a window. Run `ecg-view -h` to see all options.

From Python:

```python
from ecg_viewer import load, plot_record

rec = load("examples/synthetic_ecg.csv")
print(rec.summary())
fig = plot_record(rec, leads=["MLII"], start=0, duration=5)
fig.savefig("ecg.png", dpi=200)
```

## The figure

* **Time axis.** Seconds from the start of the original recording, so a window that
  starts at 2 s is labelled from 2 s. When the sampling rate is unknown, the axis shows
  sample indices instead.
* **Grid.** When the unit is a voltage (mV, µV or V) and the sampling rate is known,
  the standard ECG grid is drawn:
  * small squares are 0.04 s × 0.1 mV;
  * large squares are 0.2 s × 0.5 mV;
  * windows longer than 30 s use a coarser time grid.
* **Long recordings.** These are drawn with min/max decimation, so every QRS peak stays
  visible.
* **`--paper`.** Sizes the figure so that, printed at 100 %, it is exactly 25 mm/s and
  10 mm/mV.

## Tests

```bash
.venv/bin/python -m pytest
```

## Layout

```
ecg_viewer/record.py   ECGRecord: the data contract and its checks
ecg_viewer/io.py       WFDB / CSV / NPY loaders
ecg_viewer/plot.py     figure: lead rows, ECG grid, paper scale, decimation
ecg_viewer/cli.py      ecg-view command
examples/              synthetic example generator
tests/                 pytest suite
```

## Data licences

MIT-BIH Arrhythmia Database and PTB-XL are open PhysioNet datasets. MIT-BIH is under
ODC-By 1.0 and PTB-XL under CC BY 4.0. Cite them when you use them.
