"""
Cut three-component waveform windows around candidate events.

The daily SAC files of the CaMI.FRS archive are 172 MB per trace at 500 Hz
and ObsPy's SAC reader loads a whole file before trimming, so windows must
never be cut with ``read(file, starttime, endtime)`` once per event (that
pattern made step 4 take hours). :class:`DayArchive` reads every file of a
station-day once, decimates it to a working rate, and keeps the result in
memory; windows are then slices of the cached traces.

Working sampling rate: the families are built in the 5-60 Hz band (the
15-200 Hz band used by the first template scans was noise-dominated on
this array), so 250 Hz keeps a comfortable margin below Nyquist while
shrinking a 500 Hz day from 7.3 GB to 3.6 GB for 42 channels.

All windows of a run share one length, so they stack into a single array
``(n_events, n_stations, 3, n_samples)`` that the similarity and plotting
code consume directly. Missing channels are zero-filled and flagged in a
boolean ``live`` array of shape ``(n_events, n_stations, 3)``.
"""
import os
import warnings
from glob import glob

import numpy as np
from obspy import UTCDateTime, read

from seiscamp.data import day_directory

COMPONENTS = ("Z", "N", "E")   # column order used everywhere downstream


class DayArchive:
    """
    Cache of decimated day-long waveforms, one Stream per station-day.

    :type wf_dir: str
    :param wf_dir: Root of the waveform archive (either layout accepted by
        :func:`seiscamp.data.day_directory`).
    :type fs_work: float
    :param fs_work: Sampling rate (Hz) the day traces are decimated to.
    :type chan_prefix: str
    :param chan_prefix: First two characters of the channel codes (``"DP"``
        for the borehole string).

    .. rubric:: Example

    >>> arc = DayArchive("/home/gilbert_lab/cami_frs/all_daily_symlinks")
    >>> st = arc.get(UTCDateTime("2022-07-19T08:01:18"), "BH001")
    >>> st.select(channel="*Z")[0].stats.sampling_rate
    250.0
    """

    def __init__(self, wf_dir, fs_work=250.0, chan_prefix="DP"):
        self.wf_dir = wf_dir
        self.fs_work = float(fs_work)
        self.chan_prefix = chan_prefix
        self._cache = {}

    def _key(self, t, station):
        return (UTCDateTime(t).strftime("%Y%m%d"), station)

    def get(self, t, station):
        """
        Return the cached day Stream of ``station`` for the day of ``t``,
        reading and decimating the files on first use.

        :type t: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param t: Any time within the wanted day (UTC).
        :type station: str
        :param station: Station code, e.g. ``"BH001"``.
        :rtype: :class:`~obspy.core.stream.Stream`
        :return: One merged trace per component (gaps filled with zeros),
            or an empty Stream if the station has no files that day.
        """
        key = self._key(t, station)
        if key not in self._cache:
            self._cache[key] = self._read_day(UTCDateTime(t), station)
        return self._cache[key]

    def _read_day(self, t, station):
        pattern = os.path.join(day_directory(self.wf_dir, t),
                               "*.%s..%s?.*" % (station, self.chan_prefix))
        files = sorted(glob(pattern))
        st = read(pattern) if files else None
        if st is None:
            warnings.warn("no files for %s on %s" % (station, t.strftime("%Y-%m-%d")))
            from obspy import Stream
            return Stream()
        out = st.__class__()
        for comp in COMPONENTS:
            sub = st.select(channel="*%s" % comp)
            if len(sub) == 0:
                continue
            # Several traces per channel means gaps: merge, zero-filling.
            sub = sub.merge(method=1, fill_value=0)
            tr = sub[0]
            tr.data = np.asarray(tr.data, dtype=np.float32)
            self._decimate(tr)
            out += tr
        return out

    def _decimate(self, tr):
        """
        Bring one trace to ``fs_work`` in place.

        Integer ratios (500 -> 250, 1000 -> 250) use ObsPy's ``decimate``
        (anti-alias low-pass then every k-th sample). The 2020 files are
        nominally 999.99994 Hz; after decimation the rate is set exactly to
        ``fs_work`` (a 6e-8 relative change, far below any timing need).
        Non-integer ratios fall back to the slower Fourier ``resample``.
        """
        ratio = tr.stats.sampling_rate / self.fs_work
        if abs(ratio - round(ratio)) < 1e-3:
            k = int(round(ratio))
            if k > 1:
                tr.decimate(k, no_filter=False, strict_length=False)
            tr.stats.sampling_rate = self.fs_work
        else:
            tr.resample(self.fs_work)

    def clear(self):
        """Drop all cached days (call between days to bound memory)."""
        self._cache.clear()


def cut_windows(events, archive, stations, pre_s=1.0, post_s=3.0,
                fmin=5.0, fmax=60.0, min_fraction=0.98, log=print):
    """
    Cut one filtered three-component window per station around each event.

    Processing per channel: slice ``[t_ref - pre_s, t_ref + post_s]`` from
    the cached day trace (a copy), linear detrend, 5 % cosine taper,
    4-corner zero-phase Butterworth band-pass ``fmin``-``fmax``. Windows
    are kept in physical units; normalisation happens in the correlation.

    :type events: :class:`pandas.DataFrame`
    :param events: Events table with a tz-aware ``t_ref`` column (output of
        :func:`seiscamp.families.associate.associate`).
    :type archive: :class:`DayArchive`
    :param archive: Waveform cache.
    :type stations: list of str
    :param stations: Station order of the output array.
    :type pre_s: float
    :param pre_s: Seconds before ``t_ref``. SCAMP's time is the start of
        the matched 0.5 s window, usually at or just after the onset.
    :type post_s: float
    :param post_s: Seconds after ``t_ref``; 3 s covers P, S and early coda
        of a local microseismic event.
    :type fmin: float
    :param fmin: Band-pass low corner (Hz).
    :type fmax: float
    :param fmax: Band-pass high corner (Hz); must be below ``fs_work / 2``.
    :type min_fraction: float
    :param min_fraction: A slice shorter than this fraction of the nominal
        length (gap or day edge) is treated as missing.
    :return: ``(data, live)``: ``data`` is float32 of shape
        ``(n_events, n_stations, 3, n_samples)`` in the component order
        :data:`COMPONENTS`; ``live`` is bool ``(n_events, n_stations, 3)``.
    """
    fs = archive.fs_work
    if fmax >= fs / 2:
        raise ValueError("fmax=%g Hz is not below Nyquist of fs_work=%g Hz" % (fmax, fs))
    n_samp = int(round((pre_s + post_s) * fs))
    n_ev, n_sta = len(events), len(stations)
    data = np.zeros((n_ev, n_sta, len(COMPONENTS), n_samp), dtype=np.float32)
    live = np.zeros((n_ev, n_sta, len(COMPONENTS)), dtype=bool)

    for ie, t_ref in enumerate(events["t_ref"]):
        t0 = UTCDateTime(t_ref.to_pydatetime())
        for isa, sta in enumerate(stations):
            st = archive.get(t0, sta)
            for ic, comp in enumerate(COMPONENTS):
                sel = st.select(channel="*%s" % comp)
                if len(sel) == 0:
                    continue
                tr = sel[0].slice(t0 - pre_s, t0 + post_s, nearest_sample=True).copy()
                if tr.stats.npts < min_fraction * n_samp:
                    continue
                tr.detrend("linear")
                tr.taper(0.05, type="cosine")
                tr.filter("bandpass", freqmin=fmin, freqmax=fmax, corners=4, zerophase=True)
                m = min(n_samp, tr.stats.npts)
                data[ie, isa, ic, :m] = tr.data[:m]
                live[ie, isa, ic] = True
        if log and (ie + 1) % 100 == 0:
            log("  cut %d / %d events" % (ie + 1, n_ev))
    return data, live


def save_windows(path, data, live, events, stations, fs, pre_s, post_s, fmin, fmax):
    """
    Write the window array and its metadata to one compressed ``.npz``.

    :type path: str
    :param path: Output file (``.npz``).
    :param data, live: Output of :func:`cut_windows`.
    :type events: :class:`pandas.DataFrame`
    :param events: Events table; ``event_id`` and ``t_ref`` are stored.
    :type stations: list of str
    :type fs, pre_s, post_s, fmin, fmax: float
    :param fs, pre_s, post_s, fmin, fmax: Processing parameters, stored so
        downstream code never has to guess them.
    """
    np.savez_compressed(
        path, data=data, live=live,
        event_id=events["event_id"].values.astype(int),
        t_ref=np.array([t.isoformat() for t in events["t_ref"]]),
        stations=np.array(stations), components=np.array(COMPONENTS),
        fs=fs, pre_s=pre_s, post_s=post_s, fmin=fmin, fmax=fmax)


def load_windows(path):
    """
    Read a file written by :func:`save_windows`.

    :rtype: dict
    :return: Keys ``data``, ``live``, ``event_id``, ``t_ref`` (ISO strings),
        ``stations`` (list), ``components`` (list), ``fs``, ``pre_s``,
        ``post_s``, ``fmin``, ``fmax`` (floats).
    """
    with np.load(path, allow_pickle=False) as z:
        out = {k: z[k] for k in z.files}
    for k in ("fs", "pre_s", "post_s", "fmin", "fmax"):
        out[k] = float(out[k])
    out["stations"] = [str(s) for s in out["stations"]]
    out["components"] = [str(c) for c in out["components"]]
    return out
