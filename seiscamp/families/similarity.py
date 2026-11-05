"""
Network cross-correlation between events and alignment to a reference.

For a pair of events the similarity used here is the **network correlation
at a common lag**: the normalised cross-correlation functions of every
channel the two events share are averaged, and the maximum of that average
over lags within ``+/- max_lag`` is the pair's coefficient. One lag for all
channels, rather than the best lag per channel, means two events only score
high when their whole wavefields line up with a single origin-time offset,
which is what distinguishes a repeating source from two different events
with similar spectra. This is the same quantity EQcorrscan's clustering
uses (Chamberlain et al., 2018, SRL 89, 173-181, doi:10.1785/0220170151).

Normalisation uses the full-window energy of both traces, so the
coefficient is exactly the zero-lag Pearson correlation of the aligned
windows when the lag is zero and slightly conservative otherwise (the
shifted-out samples still count in the norm). With windows a few times
longer than ``max_lag`` the effect is a few percent.

Lag sign convention (document once, use everywhere)::

    lag[i, j] = argmax_tau  sum_t  x_i(t) * x_j(t + tau)

so ``x_j`` shifted *earlier* by ``lag[i, j]`` samples lines up with
``x_i``; ``lag[j, i] = -lag[i, j]``.
"""
import numpy as np
from scipy.fft import irfft, next_fast_len, rfft


def network_cc(data, live, max_lag, min_common=6, block=256, log=print):
    """
    Correlation coefficient and lag for every pair of events.

    :type data: :class:`numpy.ndarray`
    :param data: Windows ``(n_events, n_stations, 3, n_samples)`` (see
        :func:`seiscamp.families.windows.cut_windows`).
    :type live: :class:`numpy.ndarray`
    :param live: Bool ``(n_events, n_stations, 3)``; dead channels are
        excluded from the pair's average.
    :type max_lag: int
    :param max_lag: Largest lag searched, in samples, each way.
    :type min_common: int
    :param min_common: Pairs sharing fewer live channels than this get
        ``cc = 0`` (they cannot be judged).
    :type block: int
    :param block: Events per FFT block; only bounds memory.
    :return: ``(cc, lag, n_common)``: ``cc`` float ``(n, n)`` with unit
        diagonal, ``lag`` int ``(n, n)`` in samples (antisymmetric),
        ``n_common`` int ``(n, n)`` live channels shared by each pair.
    """
    n, n_sta, n_comp, m = data.shape
    n_ch = n_sta * n_comp
    X = data.reshape(n, n_ch, m).astype(np.float64)
    L = live.reshape(n, n_ch).copy()

    # Zero-mean, unit-energy channels; channels with no energy are dead.
    X -= X.mean(axis=-1, keepdims=True)
    norm = np.sqrt((X ** 2).sum(axis=-1))
    L &= norm > 0
    X = np.where(L[..., None], X / np.where(norm > 0, norm, 1.0)[..., None], 0.0)

    # Lags up to max_lag are free of wrap-around when nfft >= m + max_lag.
    nfft = next_fast_len(m + max_lag)
    F = rfft(X, nfft, axis=-1)                      # (n, n_ch, nfft//2+1)
    lag_index = np.r_[nfft - max_lag:nfft, 0:max_lag + 1]   # tau = -L..L

    cc = np.eye(n)
    lag = np.zeros((n, n), dtype=int)
    n_common = np.zeros((n, n), dtype=int)
    for i in range(n - 1):
        for j0 in range(i + 1, n, block):
            j1 = min(j0 + block, n)
            r = irfft(np.conj(F[i]) * F[j0:j1], nfft, axis=-1)[..., lag_index]  # (nb, n_ch, 2L+1)
            common = L[i] & L[j0:j1]                                            # (nb, n_ch)
            ncom = common.sum(axis=1)
            S = (r * common[..., None]).sum(axis=1) / np.maximum(ncom, 1)[:, None]
            k = S.argmax(axis=1)
            best = S[np.arange(j1 - j0), k]
            best[ncom < min_common] = 0.0
            cc[i, j0:j1] = best
            lag[i, j0:j1] = k - max_lag
            n_common[i, j0:j1] = ncom
        if log and n > 200 and (i + 1) % 200 == 0:
            log("  correlated %d / %d rows" % (i + 1, n))
    iu = np.triu_indices(n, 1)
    cc[(iu[1], iu[0])] = cc[iu]
    lag[(iu[1], iu[0])] = -lag[iu]
    n_common[(iu[1], iu[0])] = n_common[iu]
    np.fill_diagonal(n_common, L.sum(axis=1))
    return cc, lag, n_common


def shift_zero_fill(x, shift):
    """
    Shift the last axis of ``x`` by ``shift`` samples, filling with zeros
    (positive shift moves samples to later indices).

    :type x: :class:`numpy.ndarray`
    :type shift: int
    :rtype: :class:`numpy.ndarray`
    """
    out = np.zeros_like(x)
    if shift == 0:
        out[...] = x
    elif shift > 0:
        out[..., shift:] = x[..., :-shift]
    else:
        out[..., :shift] = x[..., -shift:]
    return out


def align_to_reference(data, lag, ref, members=None):
    """
    Time-shift events so that they line up with a reference event.

    With the convention of :func:`network_cc`, member ``j`` is moved
    *earlier* by ``lag[ref, j]`` samples.

    :type data: :class:`numpy.ndarray`
    :param data: Windows ``(n_events, ...)``.
    :type lag: :class:`numpy.ndarray`
    :param lag: Lag matrix from :func:`network_cc`.
    :type ref: int
    :param ref: Index of the reference event.
    :type members: sequence of int or None
    :param members: Events to align (default: all).
    :return: ``(aligned, members)`` with ``aligned`` of shape
        ``(len(members), ...)`` in the order of ``members``.
    """
    members = np.arange(len(data)) if members is None else np.asarray(members)
    aligned = np.stack([shift_zero_fill(data[j], -int(lag[ref, j])) for j in members])
    return aligned, members
