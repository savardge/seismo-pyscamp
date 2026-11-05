"""
Hierarchical clustering of the correlation matrix into waveform families,
and per-family timing statistics.

Why hierarchical with average linkage, and not connected components: step 4
links two events whenever any matrix-profile peak joins them, which is
single-linkage clustering, and single linkage chains. A -> B -> C with
cc(A, C) = 0.2 ends in one family (the campaign's median within-family
correlation was 0.45). Average linkage merges two clusters only when the
*mean* correlation between all their members passes the threshold, so a
family's members are similar to each other, not just to a neighbour.

Two tiers are cut from the same dendrogram:

* **near** (default 0.7): near-repeating events, the candidates for
  templates;
* **tight** (default 0.9): candidate repeating earthquakes, to be confirmed
  by co-location in stage 3. Thresholds should be revisited against the
  criteria discussed by Uchida & Buergmann (2019, Annu. Rev. Earth Planet.
  Sci. 47, 305-332, doi:10.1146/annurev-earth-053018-060119).
"""
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

DEFAULT_TIERS = {"near": 0.7, "tight": 0.9}


def cluster_families(cc, tiers=None, method="average", min_members=2):
    """
    Cut a hierarchical clustering of ``1 - cc`` at one or more thresholds.

    :type cc: :class:`numpy.ndarray`
    :param cc: Symmetric correlation matrix ``(n, n)``, unit diagonal.
    :type tiers: dict or None
    :param tiers: ``{name: cc_threshold}``; default :data:`DEFAULT_TIERS`.
    :type method: str
    :param method: scipy linkage method (``"average"`` or ``"complete"``;
        never ``"single"``, see module docstring).
    :type min_members: int
    :param min_members: Clusters smaller than this are labelled ``-1``.
    :return: ``(labels, Z)``: ``labels`` is ``{tier: int array (n,)}`` with
        families numbered from 0 in decreasing size and ``-1`` for
        unclustered events; ``Z`` is the scipy linkage matrix.
    """
    tiers = DEFAULT_TIERS if tiers is None else tiers
    n = len(cc)
    if n < 2:
        return {t: np.full(n, -1) for t in tiers}, None
    d = 1.0 - np.clip(cc, -1.0, 1.0)
    np.fill_diagonal(d, 0.0)
    Z = linkage(squareform(d, checks=False), method=method)
    labels = {}
    for name, thr in tiers.items():
        raw = fcluster(Z, t=1.0 - thr, criterion="distance")
        labels[name] = _relabel_by_size(raw, min_members)
    return labels, Z


def _relabel_by_size(raw, min_members):
    """Renumber cluster ids 0..k-1 by decreasing size; small ones -> -1."""
    ids, counts = np.unique(raw, return_counts=True)
    order = ids[np.argsort(-counts, kind="stable")]
    out = np.full(len(raw), -1, dtype=int)
    k = 0
    for cid in order:
        if counts[ids == cid][0] < min_members:
            continue
        out[raw == cid] = k
        k += 1
    return out


def reference_member(cc, members):
    """
    The member most correlated with the rest of its family (the natural
    alignment reference and template seed).

    :type cc: :class:`numpy.ndarray`
    :param cc: Full correlation matrix.
    :type members: sequence of int
    :param members: Indices of the family's events.
    :rtype: int
    """
    members = np.asarray(members)
    sub = cc[np.ix_(members, members)]
    return int(members[np.argmax(sub.mean(axis=1))])


def family_summary(events, labels, cc, periodic_cv=0.25, periodic_min=5):
    """
    One row of statistics per family.

    Inter-event times (IET) are computed from ``t_ref`` of the members in
    time order. A low coefficient of variation of the IET with enough
    members flags a periodic source (machinery, pumps); earthquake-like
    clustering gives CV around or above 1.

    :type events: :class:`pandas.DataFrame`
    :param events: Events table with tz-aware ``t_ref`` and ``n_stations``.
    :type labels: :class:`numpy.ndarray`
    :param labels: Family label per event (one tier), ``-1`` = none.
    :type cc: :class:`numpy.ndarray`
    :param cc: Correlation matrix.
    :type periodic_cv: float
    :param periodic_cv: IET coefficient of variation below which a family
        with at least ``periodic_min`` members is flagged periodic.
    :rtype: :class:`pandas.DataFrame`
    :return: Columns ``family``, ``n``, ``ref_event_id``, ``cc_mean``,
        ``cc_min``, ``t_first``, ``t_last``, ``duration_h``,
        ``iet_median_s``, ``iet_cv``, ``periodic``, ``n_stations_mean``.
    """
    rows = []
    t = events["t_ref"].values.astype("datetime64[ns]").astype(np.int64) / 1e9
    for fam in sorted(set(labels) - {-1}):
        idx = np.where(labels == fam)[0]
        sub = cc[np.ix_(idx, idx)]
        off = sub[~np.eye(len(idx), dtype=bool)]
        ts = np.sort(t[idx])
        iet = np.diff(ts)
        iet_med = float(np.median(iet)) if len(iet) else np.nan
        iet_cv = float(iet.std() / iet.mean()) if len(iet) > 1 and iet.mean() > 0 else np.nan
        rows.append({
            "family": fam,
            "n": len(idx),
            "ref_event_id": int(events["event_id"].values[reference_member(cc, idx)]),
            "cc_mean": float(off.mean()) if len(off) else 1.0,
            "cc_min": float(off.min()) if len(off) else 1.0,
            "t_first": pd.Timestamp(ts[0], unit="s", tz="UTC"),
            "t_last": pd.Timestamp(ts[-1], unit="s", tz="UTC"),
            "duration_h": float((ts[-1] - ts[0]) / 3600.0),
            "iet_median_s": iet_med,
            "iet_cv": iet_cv,
            "periodic": bool(len(idx) >= periodic_min and np.isfinite(iet_cv) and iet_cv < periodic_cv),
            "n_stations_mean": float(events["n_stations"].values[idx].mean()),
        })
    return pd.DataFrame(rows)
