import os
import sys

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def two_lead():
    """3 s of a 2-lead signal at 250 Hz with known values."""
    fs = 250
    t = np.arange(3 * fs) / fs
    return fs, t, np.column_stack([np.sin(2 * np.pi * t), 0.5 * np.cos(2 * np.pi * t)])
