"""
Network association of SCAMP detections into candidate events.

Step 3 of the pipeline writes, per station and day, a CSV of *pairs* of
mutually similar 0.5 s windows that were detected on all three components
(``time1`` is the anchor window shared by Z, N and E; ``time2``,
``time2_x`` and ``time2_y`` are the partner windows found on Z, N and E).
Those pairs are not events yet:

* one event usually appears several times (as ``time1`` of some pairs and
  as a partner of others, with window starts jittered by a fraction of a
  second), so the window times must first be **collapsed** into
  occurrences, one per station and event;
* a single-station occurrence can be machinery or electronic noise, so an
  **event** is required to have occurrences on several stations within a
  short tolerance.

On the CaMI.FRS borehole string all "stations" BH001-BH024 are levels of
one vertical string at the same surface coordinate, so the travel-time
moveout between them is well below a second; the association tolerance
is set by the jitter of SCAMP window starts, not by wave propagation.

Both steps are greedy and anchored on the earliest member of a group, so a
long run of closely spaced windows cannot chain into one giant group (the
failure mode of step 4's connected components).
"""
import os

import numpy as np
import pandas as pd

# Columns of the combined CSV that hold window start times, and the matrix
# profile value column that belongs to each of them.
TIME_COLUMNS = {
    "time1": "mpval",      # anchor window (Z component pair, shared by N/E)
    "time2": "mpval",      # Z partner
    "time2_x": "mpval_x",  # N partner
    "time2_y": "mpval_y",  # E partner
}


def station_from_filename(path):
    """
    Station code from a pipeline file name.

    The pipeline convention is ``{start}_{end}_{station}_{channel}_...``
    (see :func:`seiscamp.data.get_filename_root`), so the station is the
    third underscore-separated field.

    :type path: str
    :param path: File name or path of a step 2/3/4 output.
    :rtype: str
    """
    return os.path.basename(path).split("_")[2]


def load_occurrences(combined_csv, station=None):
    """
    Read one combined (per-station) detections CSV as a table of window
    occurrences, one row per window start time.

    :type combined_csv: str
    :param combined_csv: Path of a ``*_COMBINED_*_detections.csv`` file.
    :type station: str or None
    :param station: Station code; parsed from the file name if omitted.
    :rtype: :class:`pandas.DataFrame`
    :return: Columns ``station``, ``time`` (tz-aware UTC), ``mpval``
        (Pearson correlation of the window with its partner), ``source``
        (which CSV column the time came from).
    """
    if station is None:
        station = station_from_filename(combined_csv)
    df = pd.read_csv(combined_csv)
    parts = []
    for col, mpcol in TIME_COLUMNS.items():
        if col not in df.columns:
            continue
        parts.append(pd.DataFrame({
            "station": station,
            "time": pd.to_datetime(df[col], utc=True),
            "mpval": df[mpcol].values if mpcol in df.columns else np.nan,
            "source": col,
        }))
    if not parts:
        return pd.DataFrame(columns=["station", "time", "mpval", "source"])
    occ = pd.concat(parts, ignore_index=True).dropna(subset=["time"])
    return occ.sort_values("time").reset_index(drop=True)


def _epoch_seconds(times):
    """tz-aware datetime Series -> float64 array of seconds since 1970."""
    return times.values.astype("datetime64[ns]").astype(np.int64) / 1e9


def collapse_occurrences(occ, tol_s=5.0):
    """
    Merge window occurrences of the same station that fall within ``tol_s``
    of the first member of a group into one occurrence.

    Greedy, anchored on the earliest time of the group: a new group starts
    when a time is more than ``tol_s`` after the group's first time, so the
    longest possible group spans ``tol_s`` regardless of how many windows
    it holds. The representative time is the median of the group.

    :type occ: :class:`pandas.DataFrame`
    :param occ: Output of :func:`load_occurrences` (any number of stations).
    :type tol_s: float
    :param tol_s: Collapse tolerance in seconds. 5 s absorbs the repeated
        windows SCAMP finds along one event's wavetrain.
    :rtype: :class:`pandas.DataFrame`
    :return: Columns ``station``, ``time``, ``n_windows`` (windows merged),
        ``mpval_max``.
    """
    rows = []
    for sta, g in occ.groupby("station", sort=True):
        g = g.sort_values("time")
        ts = _epoch_seconds(g["time"])
        mp = g["mpval"].values.astype(float)
        start = 0
        for i in range(1, len(ts) + 1):
            if i == len(ts) or ts[i] - ts[start] > tol_s:
                sl = slice(start, i)
                mp_sl = mp[sl]
                rows.append({
                    "station": sta,
                    "time": pd.Timestamp(np.median(ts[sl]), unit="s", tz="UTC"),
                    "n_windows": i - start,
                    "mpval_max": np.nanmax(mp_sl) if np.isfinite(mp_sl).any() else np.nan,
                })
                start = i
    out = pd.DataFrame(rows, columns=["station", "time", "n_windows", "mpval_max"])
    return out.sort_values("time").reset_index(drop=True)


def associate(occ, tol_s=1.0, min_stations=3):
    """
    Associate collapsed occurrences across stations into candidate events.

    Occurrences are scanned in time order. The earliest unused occurrence
    opens an event; every later occurrence within ``tol_s`` of that anchor,
    on a station not yet in the event, joins it. The event is kept if it
    reaches ``min_stations`` distinct stations. Occurrences that opened or
    joined a group too small to be kept are consumed as well (they are
    reported in the ``associated`` column of the returned occurrences), so
    every occurrence is used at most once.

    :type occ: :class:`pandas.DataFrame`
    :param occ: Output of :func:`collapse_occurrences`.
    :type tol_s: float
    :param tol_s: Association window in seconds, measured from the anchor.
    :type min_stations: int
    :param min_stations: Minimum number of distinct stations per event.
    :return: ``(events, event_stations, occ)``: one row per event
        (``event_id``, ``t_ref`` = median member time, ``t_first``,
        ``t_last``, ``n_stations``, ``stations`` joined by ``;``,
        ``mpval_max``, ``n_windows``, ``day`` as YYYYMMDD, ``context`` set
        to ``"unknown"`` for the classifier stage to fill); one row per
        event-station member; and the input occurrences with the
        ``event_id`` they went to (-1 if none).
    """
    occ = occ.sort_values("time").reset_index(drop=True)
    n = len(occ)
    ts = _epoch_seconds(occ["time"])
    stations = occ["station"].values
    event_of = np.full(n, -1, dtype=int)
    used = np.zeros(n, dtype=bool)
    events, members_all = [], []

    for i in range(n):
        if used[i]:
            continue
        used[i] = True
        members, seen = [i], {stations[i]}
        j = i + 1
        while j < n and ts[j] - ts[i] <= tol_s:
            if not used[j] and stations[j] not in seen:
                used[j] = True
                members.append(j)
                seen.add(stations[j])
            j += 1
        if len(seen) < min_stations:
            continue
        eid = len(events)
        event_of[members] = eid
        t_members = ts[members]
        t_ref = pd.Timestamp(np.median(t_members), unit="s", tz="UTC")
        events.append({
            "event_id": eid,
            "t_ref": t_ref,
            "t_first": pd.Timestamp(t_members.min(), unit="s", tz="UTC"),
            "t_last": pd.Timestamp(t_members.max(), unit="s", tz="UTC"),
            "n_stations": len(seen),
            "stations": ";".join(sorted(seen)),
            "mpval_max": np.nanmax(occ["mpval_max"].values[members]),
            "n_windows": int(occ["n_windows"].values[members].sum()),
            "day": t_ref.strftime("%Y%m%d"),
            "context": "unknown",
        })
        members_all.extend((eid, m) for m in members)

    events = pd.DataFrame(events, columns=[
        "event_id", "t_ref", "t_first", "t_last", "n_stations", "stations",
        "mpval_max", "n_windows", "day", "context"])
    es_rows = [{"event_id": eid, **occ.iloc[m].to_dict()} for eid, m in members_all]
    event_stations = pd.DataFrame(es_rows, columns=["event_id", "station", "time", "n_windows", "mpval_max"])
    occ = occ.assign(event_id=event_of)
    return events, event_stations, occ
