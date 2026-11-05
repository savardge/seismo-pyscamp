"""
Figures for stage 1: the ordered correlation matrix and one waveform
figure per family.

The family figure follows the agreed standard: one figure per family,
columns Z / N / E, one labelled row per station, members in grey, the
aligned stack in colour. (A template, in black, is added in stage 2 when
one exists.)
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STACK_COLOR = "#2a78d6"
MEMBER_COLOR = "#6b6b6b"
INK, MUTED = "#0b0b0b", "#52514e"


def plot_cc_matrix(cc, labels, out, title=""):
    """
    Correlation matrix with events ordered by family, families outlined.

    :type cc: :class:`numpy.ndarray`
    :param cc: Correlation matrix ``(n, n)``.
    :type labels: :class:`numpy.ndarray`
    :param labels: Family label per event (``-1`` = unclustered, drawn last).
    :type out: str
    :param out: Output PNG path.
    :type title: str
    :rtype: str
    :return: ``out``.
    """
    n = len(cc)
    key = np.where(labels < 0, labels.max() + 1, labels)
    order = np.lexsort((-cc.mean(axis=1), key))
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(cc[np.ix_(order, order)], cmap="Blues", vmin=0, vmax=1,
                   interpolation="nearest")
    # Family boundaries.
    sorted_labels = labels[order]
    edges = np.where(np.diff(sorted_labels) != 0)[0] + 0.5
    for e in edges:
        ax.axhline(e, color=INK, lw=0.4)
        ax.axvline(e, color=INK, lw=0.4)
    n_fam = int((labels >= 0).sum())
    ax.set_title("%s%d events, %d in families" % (title + "\n" if title else "", n, n_fam),
                 fontsize=10, color=INK, loc="left")
    ax.set_xlabel("event (ordered by family)", color=MUTED)
    ax.set_ylabel("event", color=MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("network correlation at common lag", color=MUTED)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def plot_family(aligned, live, stations, components, fs, pre_s, out,
                title="", max_members=200):
    """
    Waveform figure of one family: columns = components, rows = stations.

    Every member trace is normalised to its own maximum absolute value
    before plotting, so the figure shows waveform shape, not amplitude;
    the stack is the mean of those normalised traces. Only stations with at
    least one live member are drawn.

    :type aligned: :class:`numpy.ndarray`
    :param aligned: Member windows already aligned to the reference,
        shape ``(n_members, n_stations, 3, n_samples)``.
    :type live: :class:`numpy.ndarray`
    :param live: Bool ``(n_members, n_stations, 3)``.
    :type stations: list of str
    :type components: list of str
    :param components: Component order of axis 2 (plotted in this order).
    :type fs: float
    :param fs: Sampling rate (Hz).
    :type pre_s: float
    :param pre_s: Seconds before the SCAMP time at the window start, so the
        time axis is zero at the detection time.
    :type out: str
    :param out: Output PNG path.
    :type title: str
    :type max_members: int
    :param max_members: Plot at most this many members (the stack uses all).
    :rtype: str
    :return: ``out``.
    """
    n_mem, n_sta, n_comp, m = aligned.shape
    t = np.arange(m) / fs - pre_s
    rows = [i for i in range(n_sta) if live[:, i, :].any()]
    if not rows:
        return None
    fig, axs = plt.subplots(len(rows), n_comp, figsize=(4.2 * n_comp, 1.15 * len(rows) + 1.2),
                            sharex=True, squeeze=False)
    step = max(1, n_mem // max_members)
    for r, isa in enumerate(rows):
        for c in range(n_comp):
            ax = axs[r][c]
            ok = live[:, isa, c]
            if ok.any():
                x = aligned[ok, isa, c, :]
                peak = np.abs(x).max(axis=1, keepdims=True)
                xn = x / np.where(peak > 0, peak, 1.0)
                for tr in xn[::step]:
                    ax.plot(t, tr, color=MEMBER_COLOR, lw=0.4, alpha=0.35)
                ax.plot(t, xn.mean(axis=0), color=STACK_COLOR, lw=1.1)
            ax.axvline(0, color=MUTED, lw=0.6, ls=":")
            ax.set_ylim(-1.1, 1.1)
            ax.set_yticks([])
            for s in ("top", "right", "left"):
                ax.spines[s].set_visible(False)
            ax.tick_params(colors=MUTED, labelsize=8)
            if c == 0:
                ax.set_ylabel(stations[isa], rotation=0, ha="right", va="center",
                              fontsize=9, color=INK)
            if r == 0:
                ax.set_title(components[c], fontsize=10, color=INK)
    for c in range(n_comp):
        axs[-1][c].set_xlabel("time from SCAMP detection (s)", color=MUTED, fontsize=9)
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out
