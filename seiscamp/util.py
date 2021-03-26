import numpy as np
from statsmodels import robust
import logging

Logger = logging.getLogger(__name__)


def RunningStd(x,N):
    idx = np.arange(N) + np.arange(len(x)-N+1)[:,None]
    #b = [row[row>0] for row in x[idx]] # to exclude 0's
    #return np.array(map(np.std,b)) # to exclude 0's
    #return np.std(x[idx],axis=1) # no padding
    return np.pad(np.std(x[idx],axis=1), pad_width=(0, N-1), mode="edge")
    #return np.array([np.median(c) for c in b])  # This also works

    
def RunningMedian(x,N):
    idx = np.arange(N) + np.arange(len(x)-N+1)[:,None]
    return np.pad(np.median(x[idx],axis=1), pad_width=(0, N-1), mode="edge")


def RunningMean(x,N):
    idx = np.arange(N) + np.arange(len(x)-N+1)[:,None]
    return np.pad(np.mean(x[idx],axis=1), pad_width=(0, N-1), mode="edge")
    

def detect_time_gaps(trace, min_samples=10, epsilon=1e-20, thresh_disc=100):
    """
    Detect time gaps where data is filled with zeros for a given Obspy trace.
    :param trace: obspy.core.trace
    :param min_samples: Minimum number of samples for time gap
    :param epsilon: Machine precision (use to define zero)
    :param thresh_disc: Threshold for discontinuity for multiple gaps
    :return: num_gaps, gap_start_ind, gap_end_ind
    """

    tdata = trace.data

    indz = np.where(abs(tdata) < epsilon)[0]  # indices where we have 0
    diff_indz = indz[min_samples:] - indz[
                                     0:-min_samples]  # Need min_samples consecutive samples with 0's to identify as time gap
    ind_des = np.where(diff_indz == min_samples)[0]  # desired indices: value is equal to min_samples in the time gap
    ind_gap = indz[ind_des]  # indices of the time gaps

    gap_start_ind = []
    gap_end_ind = []

    if 0 == len(ind_gap):  # No time gaps found
        num_gaps = 0
    else:
        # May have more than 1 time gap
        ind_diff = np.diff(ind_gap)  # discontinuities in indices of the time gaps, if there is more than 1 time gap
        ind_disc = np.where(ind_diff > thresh_disc)[0]

        # N-1 time gaps
        # Logger.info(ind_gap)
        curr_ind_start = ind_gap[0]
        # Logger.info(curr_ind_start)
        for igap in range(len(ind_disc)):  # do not enter this loop if ind_disc is empty
            gap_start_ind.append(curr_ind_start)
            last_index = ind_gap[ind_disc[igap]] + min_samples
            gap_end_ind.append(last_index)
            curr_ind_start = ind_gap[ind_disc[igap] + 1]  # update for next iteration

        # Last time gap
        gap_start_ind.append(curr_ind_start)
        gap_end_ind.append(ind_gap[-1] + min_samples)
        num_gaps = len(gap_start_ind)

    return [num_gaps, gap_start_ind, gap_end_ind]


def fill_time_gaps_noise(stream, min_samples=10, epsilon=1e-20, thresh_disc=100, ntest=2000, verbose=False):
    """
    Fill time gaps where data is filled with zeros for a given Obspy stream.
    :param stream: obspy.core.stream
    :param min_samples: Minimum number of samples for time gap
    :param epsilon: Machine precision (use to define zero)
    :param thresh_disc: Threshold for discontinuity for multiple gaps
    :param ntest: Number of test samples in data - assume they are noise
    :param verbose: Show details
    :return: stream filled with noise for time gaps.
    """

    for trace in stream:

        dt = trace.stats.delta

        # 1) Find time gaps
        [num_gaps, gap_start_ind, gap_end_ind] = detect_time_gaps(trace, min_samples, epsilon, thresh_disc)

        if verbose and num_gaps > 0:
            Logger.info('Number of time gaps for %s: %d' % (trace.id, num_gaps))
            for igap in range(num_gaps):
                tstart = trace.stats.starttime + gap_start_ind[igap] * dt
                duration = (gap_end_ind[igap] - gap_start_ind[igap]) * dt
                Logger.info("Time: %s,  Duration: %d s." % (tstart, duration))

        # 2) Fill with uncorrelated noise

        for igap in range(num_gaps):
            ngap = gap_end_ind[igap] - gap_start_ind[igap] + 1

            # Bounds check
            if gap_start_ind[igap] - ntest < 0:  # not enough data on left side
                # median of ntest noise values on right side of gap
                median_gap_right = np.median(trace.data[gap_end_ind[igap] + 1:gap_end_ind[igap] + ntest])
                median_gap = median_gap_right
                # MAD of ntest noise values on right side of gap
                mad_gap_right = robust.mad(trace.data[gap_end_ind[igap] + 1:gap_end_ind[igap] + ntest])
                mad_gap = mad_gap_right
            elif gap_end_ind[igap] + ntest >= len(trace.data):  # not enough data on left side
                # median of ntest noise values on left side of gap
                median_gap_left = np.median(trace.data[gap_start_ind[igap] - ntest:gap_start_ind[igap] - 1])
                median_gap = median_gap_left
                # MAD of ntest noise values on left side of gap
                mad_gap_left = robust.mad(trace.data[gap_start_ind[igap] - ntest:gap_start_ind[igap] - 1])
                mad_gap = mad_gap_left
            else:
                # median of ntest noise values on left side of gap
                median_gap_left = np.median(trace.data[gap_start_ind[igap] - ntest:gap_start_ind[igap] - 1])
                # median of ntest noise values on right side of gap
                median_gap_right = np.median(trace.data[gap_end_ind[igap] + 1:gap_end_ind[igap] + ntest])

                median_gap = 0.5 * (median_gap_left + median_gap_right)
                # MAD of ntest noise values on left side of gap
                mad_gap_left = robust.mad(trace.data[gap_start_ind[igap] - ntest:gap_start_ind[igap] - 1])
                # MAD of ntest noise values on right side of gap
                mad_gap_right = robust.mad(trace.data[gap_end_ind[igap] + 1:gap_end_ind[igap] + ntest])
                mad_gap = 0.5 * (mad_gap_left + mad_gap_right)

            # Fill in gap with uncorrelated white Gaussian noise
            gap_x = mad_gap * np.random.randn(ngap) + median_gap
            trace.data[gap_start_ind[igap]:gap_end_ind[igap] + 1] = gap_x

    return stream


def mccc(seis, dt, twin, ccmin, comp='Z'):
    from scipy.linalg import lstsq
    """ FUNCTION [TDEL,RMEAN,SIGR] = MCCC(SEIS,DT,TWIN);
    Function MCCC determines optimum relative delay times for a set of seismograms based on the
    VanDecar & Crosson multi-channel cross-correlation algorithm. SEIS is the set of seismograms.
    It is assumed that this set includes the window of interest and nothing more since we calculate the
    correlation functions in the Fourier domain. DT is the sample interval and TWIN is the window about
    zero in which the maximum search is performed (if TWIN is not specified, the search is performed over
    the entire correlation interval).
    APP added the ccmin, such that only signals that meet some threshold similarity contribute to the delay times. """

    # Set nt to twice length of seismogram section to avoid
    # spectral contamination/overlap. Note we assume that
    # columns enumerate time samples, and rows enumerate stations.
    # Note in typical application ns is not number of stations...its really number of events
    # all data is from one station
    nt = np.shape(seis)[1] * 2
    ns = np.shape(seis)[0]
    tcc = np.zeros([ns, ns])

    # Copy seis for normalization correction
    seis2 = np.copy(seis)

    # Set width of window around 0 time to search for maximum
    # mask = np.ones([1,nt])
    # if nargin == 3:
    itw = int(np.fix(twin / (2 * dt)))
    mask = np.zeros([1, nt])[0]
    mask[0:itw + 1] = 1.0
    mask[nt - itw:nt] = 1.0

    # Zero array for sigt and list on non-zero channels
    sigt = np.zeros(ns)

    # First remove means, compute autocorrelations, and find non-zeroed stations.
    for iss in range(0, ns):
        seis[iss, :] = seis[iss, :] - np.mean(seis[iss, :])
        ffiss = np.fft.fft(seis[iss, :], nt)
        acf = np.real(np.fft.ifft(ffiss * np.conj(ffiss), nt))
        sigt[iss] = np.sqrt(max(acf))

    # Determine relative delay times between all pairs of traces.
    r = np.zeros([ns, ns])
    tcc = np.zeros([ns, ns])

    # Two-Channel normalization ---------------------------------------------------------

    # This loop gets a correct r by checking how many channels are actually being compared
    if comp == 'NE':

        # First find the zero-channels (the np.any tool will fill in zeroNE)
        # zeroNE ends up with [1,0], [0,1], or [1,1] for each channel, 1 meaning there IS data
        zeroNE = np.zeros([ns, 2])
        dum = np.any(seis2[:, 0:nt / 4], 1, zeroNE[:, 0])
        dum = np.any(seis2[:, nt / 4:nt / 2], 1, zeroNE[:, 1])

        # Now start main (outer) loop
        for iss in range(0, ns - 1):
            ffiss = np.conj(np.fft.fft(seis[iss, :], nt))

            for jss in range(iss + 1, ns):

                ffjss = np.fft.fft(seis[jss, :], nt)
                # ccf  = np.real(np.fft.ifft(ffiss*ffjss,nt))*mask
                ccf = np.fft.fftshift(np.real(np.fft.ifft(ffiss * ffjss, nt)) * mask)
                cmax = np.max(ccf)

                # chcor for channel correction sqrt[ abs( diff[jss] - diff[iss]) + 1]
                # This would be perfect correction if N,E channels always had equal power, but for now is approximate
                chcor = np.sqrt(abs(zeroNE[iss, 0] - zeroNE[jss, 0] - zeroNE[iss, 1] + zeroNE[jss, 1]) + 1)

                # OLD, INCORRECT chcor
                # chcor = np.sqrt( np.sum(zeroNE[iss,:])+np.sum(zeroNE[jss,:]) - (zeroNE[iss,0]*zeroNE[jss,0]+zeroNE[iss,1]*zeroNE[jss,1]) )

                rtemp = cmax * chcor / (sigt[iss] * sigt[jss])

                # Quadratic interpolation for optimal time (only if CC found > ccmin)
                if rtemp > ccmin:

                    ttemp = np.argmax(ccf)

                    x = np.array(ccf[ttemp - 1:ttemp + 2])
                    A = np.array([[1, -1, 1], [0, 0, 1], [1, 1, 1]])

                    [a, b, c] = lstsq(A, x)[0]

                    # Solve dy/dx = 2ax + b = 0 for time (x)
                    tcc[iss, jss] = -b / (2 * a) + ttemp

                    # Estimate cross-correlation coefficient
                    # r[iss,jss] = cmax/(sigt[iss]*sigt[jss])
                    r[iss, jss] = rtemp
                else:
                    tcc[iss, jss] = nt / 2

                    # Reguar Normalization Version -------------------------------------------------------
    elif comp != 'NE':
        for iss in range(0, ns - 1):
            ffiss = np.conj(np.fft.fft(seis[iss, :], nt))
            for jss in range(iss + 1, ns):

                ffjss = np.fft.fft(seis[jss, :], nt)
                # ccf  = np.real(np.fft.ifft(ffiss*ffjss,nt))*mask
                ccf = np.fft.fftshift(np.real(np.fft.ifft(ffiss * ffjss, nt)) * mask)
                cmax = np.max(ccf)

                rtemp = cmax / (sigt[iss] * sigt[jss])

                # Quadratic interpolation for optimal time (only if CC found > ccmin)
                if rtemp > ccmin:

                    ttemp = np.argmax(ccf)

                    x = np.array(ccf[ttemp - 1:ttemp + 2])
                    A = np.array([[1, -1, 1], [0, 0, 1], [1, 1, 1]])

                    [a, b, c] = lstsq(A, x)[0]

                    # Solve dy/dx = 2ax + b = 0 for time (x)
                    tcc[iss, jss] = -b / (2 * a) + ttemp

                    # Estimate cross-correlation coefficient
                    # r[iss,jss] = cmax/(sigt[iss]*sigt[jss])
                    r[iss, jss] = rtemp
                else:
                    tcc[iss, jss] = nt / 2

                    #######################################################

    # Some r could have been made > 1 due to approximation, fix this
    r[r >= 1] = 0.99

    # Fisher's transform of cross-correlation coefficients to produce
    # normally distributed quantity on which Gaussian statistics
    # may be computed and then inverse transformed
    z = 0.5 * np.log((1 + r) / (1 - r))
    zmean = np.zeros(ns)
    for iss in range(0, ns):
        zmean[iss] = (np.sum(z[iss, :]) + np.sum(z[:, iss])) / (ns - 1)
    rmean = (np.exp(2 * zmean) - 1) / (np.exp(2 * zmean) + 1)

    # Correct negative delays (for fftshifted times)
    # ix = np.where( tcc>nt/2);  tcc[ix] = tcc[ix]-nt
    tcc = tcc - nt / 2

    # Subtract 1 to account for sample 1 at 0 lag (Not in python)
    # tcc = tcc-1

    # Multiply by sample rate
    tcc = tcc * dt

    # Use sum rule to assemble optimal delay times with zero mean
    tdel = np.zeros(ns)

    # I changed the tdel calculation to not include zeroed-out waveform pairs in normalization
    for iss in range(0, ns):
        ttemp = np.append(tcc[iss, iss + 1:ns], -tcc[0:iss, iss])
        tdel[iss] = np.sum(ttemp) / (np.count_nonzero(ttemp) + 1)
        # tdel[iss] = ( np.sum(tcc[iss,iss+1:ns])-np.sum(tcc[0:iss,iss]) )/ns

    # Compute associated residuals
    res = np.zeros([ns, ns])
    sigr = np.zeros(ns)
    for iss in range(0, ns - 1):
        for jss in range(iss + 1, ns):
            res[iss, jss] = tcc[iss, jss] - (tdel[iss] - tdel[jss])

    for iss in range(0, ns):
        sigr[iss] = np.sqrt((np.sum(res[iss, iss + 1:ns] ** 2) + np.sum(res[0:iss, iss] ** 2)) / (ns - 2))

    return tdel, rmean, sigr, r, tcc

