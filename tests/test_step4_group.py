"""
step4_group must report family event times at the detected events, not at the
merge-window lower bound (which put them 1-2 s early).

Run with: python -m pytest tests/test_step4_group.py
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest
from obspy import Trace, UTCDateTime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts_frs", "python_scripts"))
from step4_group import aligned_event_times, extract  # noqa: E402

BASE = UTCDateTime("2025-06-01T03:00:00")
FS = 100.0
# Epoch timestamps are ~1.7e9 s: compare with rtol=0, since the default rtol allows minutes of error


def _fmt(t):
    # Same format as step2_extract_detections (10 ms precision)
    return (BASE + t).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-4]


def _write_detections(path, pairs):
    df = pd.DataFrame({
        "time1": [_fmt(a) for a, _ in pairs],
        "time2": [_fmt(b) for _, b in pairs],
        "mpval": 0.1,
        "prominence": 0.5,
        "peak_std_sec": 0.01,
    })
    df.to_csv(path, index=False)


def test_window_event_times_are_subsecond_detection_times(tmp_path):
    # Three families; event 600.49 is detected twice with 20 ms jitter
    fam_a = [12.37, 140.81, 300.55]
    fam_b = [55.12, 401.96]
    pairs = [
        (fam_a[0], fam_a[1]), (fam_a[0], fam_a[2]), (fam_a[1], fam_a[2]),
        (fam_b[0], fam_b[1]),
        (600.48, 900.73), (600.50, 1200.04),
    ]
    csv = tmp_path / "x_y_STA_detections.csv"
    _write_detections(csv, pairs)

    df, dfref, groups = extract(str(csv))

    true_events = np.array(sorted(fam_a + fam_b + [600.49, 900.73, 1200.04]))
    tevent = np.sort(dfref.tevent.values) - BASE.timestamp
    np.testing.assert_allclose(tevent, true_events, rtol=0, atol=0.011)

    # Windows keep the +/-1 s merge tolerance, without integer truncation
    single = dfref[np.isclose(dfref.ub - dfref.lb, 2.0)]
    np.testing.assert_allclose(single.tevent - single.lb, 1.0, rtol=0, atol=1e-6)

    assert len(groups) == 3


def _shifts_like_align_traces(trace_starts, true_events, master, rng):
    # align_traces: trace k sample n matches master sample n + shift_k
    offset = np.asarray(true_events) - np.asarray(trace_starts)
    shifts = offset[master] - offset
    return np.round(shifts * FS + rng.integers(-1, 2, len(shifts))) / FS  # +/- 1 sample


@pytest.mark.parametrize("master", [0, 3, 5])
def test_aligned_event_times_recover_events(master):
    rng = np.random.default_rng(42)
    true_events = BASE.timestamp + np.sort(rng.uniform(0, 86400, 6))
    # As in step4: window lb = earliest detection - 1 s (or earlier if merged), trace starts at lb - 4 s
    lb = true_events - 1.0 - rng.uniform(0, 0.9, 6)
    trace_starts = np.round((lb - 4) * FS) / FS
    shifts = _shifts_like_align_traces(trace_starts, true_events, master, rng)

    reported = aligned_event_times(trace_starts, true_events, shifts)
    np.testing.assert_allclose(reported - true_events, 0, rtol=0, atol=1.5 / FS)

    # Previous output (tlb - shift) was systematically early
    assert np.all((lb - shifts) - true_events < -0.9)


def test_aligned_event_times_with_eqcorrscan_alignment():
    stacking = pytest.importorskip("eqcorrscan.utils.stacking")
    rng = np.random.default_rng(0)
    n = 6
    true_events = BASE.timestamp + np.sort(rng.uniform(0, 86400, n))
    lb = true_events - 1.0 - rng.uniform(0, 0.9, n)
    ub = lb + 2.0

    t = np.arange(int(11 * FS)) / FS
    traces = []
    for tev, l, u in zip(true_events, lb, ub):
        start = np.round((l - 4) * FS) / FS
        npts = int(round((u + 4 - start) * FS))
        tt = t[:npts] - (tev - start)
        wavelet = np.exp(-(tt / 0.08) ** 2) * np.cos(2 * np.pi * 8 * tt)
        data = wavelet + 0.05 * rng.standard_normal(npts)
        traces.append(Trace(data=data, header={"starttime": UTCDateTime(start), "sampling_rate": FS}))

    shifts, ccs = stacking.align_traces(traces, shift_len=int(4. * FS), master=False, positive=False, plot=False)
    trace_starts = [tr.stats.starttime.timestamp for tr in traces]

    reported = aligned_event_times(trace_starts, true_events, shifts)
    np.testing.assert_allclose(reported - true_events, 0, rtol=0, atol=1.5 / FS)
    assert np.all((lb - np.asarray(shifts)) - true_events < -0.9)
