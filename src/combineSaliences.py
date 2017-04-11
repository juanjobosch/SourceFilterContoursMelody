
import numpy as np


def combine2(timesHF0, HF0init, timesHSSF, HSSF, G=0, mu=1, doConvolution=True,HF0norm='max'):
    """ Combines HF0 (based on SIMM) and HS (Harmonic summation)
        Parameters
    ----------
    timesHF0: timestamps of each frame in HF0 matrix
    HF0init: Nbins*Nframes
    timesHSSF: timestamps of each frame in HSSF matrix
    HSSF: Nframes2*Nbins2
    Saliences are assumed to have same hopsize and number of bins per semitone
    See the effect of G  (set to 0 to have no effect)
    mu:  HF0 = HF0 ** (1. / mu)  (Set to 1 to have no effect)
    doConvolution: True or False, to perform the convolution with a Gaussian

    Returns
    -------
    times: Timestamps of the frames of the combined salience function
    sal:  Combined Salience function
    """

    # Globally normalise HS

    plotting = False
    HSSF = HSSF.T
    HSSF = HSSF / np.max(HSSF)
    if plotting:
        try:
            import pylab as plt
            import matplotlib.gridspec as gridspec

            f, axarr = plt.subplots(2, 2)
            f.subplots_adjust(wspace=0.00001, hspace=0.00001)
            plt.size([7, 7])

            axarr[0, 0].set_xlim(2800, 3600)
            axarr[0, 0].set_xlim(2800, 3600)
            axarr[0, 1].set_xlim(2800, 3600)
            axarr[0, 1].set_xlim(2800, 3600)
            axarr[1, 0].set_xlim(2800, 3600)
            axarr[1, 0].set_xlim(2800, 3600)
            axarr[1, 1].set_xlim(2800, 3600)
            axarr[1, 1].set_xlim(2800, 3600)

            # plt.setp(axarr,1000
            # locs, labels = plt.xticks(1000*256./44100)
            # print labels
            # labels = labels

            # normalise by the max

            axarr[0, 0].imshow(np.log10(HF0init / (HF0init.max()) + 1e-15), origin='lower')
            axarr[0, 0].set_title('(a) (log)HF0 init')

            axarr[0, 1].imshow(HSSF, origin='lower')
            axarr[0, 1].set_title('(b) HS')
        except:
            print "Error in plotting"

    # Frame-wise normalisation dividing by the max on each frame
    if HF0norm == 'max':
        HF0init = (HF0init / (np.outer(np.ones(HF0init.shape[0]), np.max(HF0init, 0)) + 1e-15))

    # Frame-wise normalisation dividing by the sum on each frame
    if HF0norm == 'sum':
        HF0init = (HF0init / (np.outer(np.ones(HF0init.shape[0]), np.sum(HF0init, 0)) + 1e-15))


    # Gaussian filtering
    if doConvolution:
        sigma = 2
        Gausssize = 5
        x = np.linspace(-Gausssize / 2., Gausssize / 2., Gausssize)
        gaussFilter = np.exp(-x ** 2 / (2 * sigma ** 2))
        gaussFilter = gaussFilter / np.sum(gaussFilter)  # normalize
        HF0 = np.zeros_like(HF0init)
        for i in range(HF0init.shape[1]):
            HF0[:, i] = np.convolve(HF0init[:, i], gaussFilter, mode='same')
    else:
        HF0 = HF0init

    # Global normalisation
    HF0 = HF0 / np.max(HF0)

    # Scaling
    # mu=1 (no scaling) in MIREX (2015,2016), SMC2016 and ISMIR2016
    HF0 = HF0 ** (1. / mu)

    hopSize = np.mean(np.diff(timesHF0))

    # Combining salience functions
    N1Fr = np.argmin(np.abs(timesHF0 - timesHSSF[0]))

    Nf0Mel = HSSF.shape[0]
    NfrMel = HSSF.shape[1]

    Nf0Dur = HF0.shape[0]
    NfrDur = HF0.shape[1]

    NF0 = max(Nf0Dur, Nf0Dur)
    NFr = max(NfrMel, NfrDur) + N1Fr
    times = timesHF0[0] + np.arange(NFr) * hopSize

    # Setting shape of the combination
    salcomb = np.zeros([NF0, NFr])
    salcomb[np.ix_(np.arange(Nf0Dur), (np.arange(NfrDur)))] = (1 - G) * HF0

    # hadamard product
    # G is = 0 in MIREX (2015,2016), SMC2016 and ISMIR2016
    salcomb[np.ix_(np.arange(Nf0Mel), np.arange(N1Fr, NfrMel + N1Fr))] = HSSF * (
        G + salcomb[np.ix_(np.arange(Nf0Mel), np.arange(N1Fr, NfrMel + N1Fr))])

    if plotting:
        try:
            axarr[1, 0].imshow(HF0, origin='lower')
            axarr[1, 0].set_title('(c) HF0-GF-Fn')
            axarr[1, 1].imshow(salcomb, origin='lower')
            axarr[1, 1].set_title('(d) Combination')

            plt.setp([a.get_xticklabels() for a in axarr[0, :]], visible=False)
            axarr[1, 1].set_xlabel('Frame number')
            axarr[1, 0].set_xlabel('Frame number')
            # axarr[0, 1].set_xlabel('Frame number')
            # axarr[0, 0].set_xlabel('Frame number')
            plt.setp([a.get_yticklabels() for a in axarr[:, 1]], visible=False)
            axarr[0, 0].set_ylabel('bins')
            axarr[1, 0].set_ylabel('bins')
            # axarr[0, 1].set_ylabel('bins')
            # axarr[1, 1].set_ylabel('bins')
            axarr[0, 0].tick_params(labelsize=10)
            axarr[0, 1].tick_params(labelsize=10)
            axarr[1, 0].tick_params(labelsize=10)
            axarr[1, 1].tick_params(labelsize=10)
            plt.tight_layout()
            # plt.imshow(HF0init,origin='lower')
            plt.savefig('saliences.pdf', bbox_inches='tight')
            plt.show()
        except:
            print("Error in plotting")

    return times, salcomb / salcomb.max()


def simpleResize(timesHF0, HF0init, timesHSSF, HSSF):
    """ Simple resizing of HF0 (based on SIMM) and HS (Harmonic summation) if necessary
        Parameters.
        Could also be performed with scipy interpolate
    ----------
    timesHF0: timestamps of each frame in HF0 matrix
    HF0init: Nbins*Nframes
    timesHSSF: timestamps of each frame in HSSF matrix
    HSSF: Nframes2*Nbins2

    Returns
    -------
    timesHF0: timestamps of each frame in HF0 matrix
    HF0init: resized HF0
    timesHSSF: timestamps of each frame in HSSF matrix
    HSSF: resized HS   """

    ratio = 1.0 * HSSF.shape[1] / HF0init.shape[0]
    n = round(ratio)
    if n > 1 and abs(n - ratio) < 0.01:
        HF0init = np.repeat(HF0init, n, axis=0)
    else:
        ratio = 1.0 * HF0init.shape[0] / HSSF.shape[1]
        n = round(ratio)
        if n > 1 and abs(n - ratio) < 0.01:
            HSSF = np.repeat(HSSF, n, axis=0)
    ratio = 1.0 * HSSF.shape[0] / HF0init.shape[1]
    n = round(ratio)
    if n > 1 and abs(n - ratio) < 0.01:
        HF0init = np.repeat(HF0init, n, axis=1)
        hop = np.diff(timesHF0)[0] / 2.
        timesHF0 = np.arange(timesHF0[0], timesHF0[-1] + (n - 1) * hop, hop)
    else:
        ratio = 1.0 * HF0init.shape[1] / HSSF.shape[0]
        n = round(ratio)
        if n > 1 and abs(n - ratio) < 0.01:
            HSSF = np.repeat(HSSF, n, axis=1)
            hop = np.diff(timesHSSF)[0] / 2.
            timesHSSF = np.arange(timesHSSF[0], timesHSSF[-1] + (n - 1) * hop, hop)
    return timesHF0, HF0init, timesHSSF, HSSF

def combine14(timesHF0, HF0init, timesHSSF, HSSF, G, mu, doConvolution):

    timesHF0, HF0init, timesHSSF, HSSF = simpleResize(timesHF0, HF0init, timesHSSF, HSSF)

    # if (HSSF.T.shape != HF0init.shape):
    #    HF0init = interpolateSaliences(HSSF.T,HF0init,timesHSSF,timesHF0)

    times, sal = combine2(timesHF0, HF0init, timesHSSF, HSSF, G, mu, doConvolution,HF0norm='sum')
    return times, sal

def combine3MIREX(timesHF0, HF0init, timesHSSF, HSSF, G, mu, doConvolution):
    """ Combines HF0 and HS, used in MIREX (2015,2016), SMC2016 and ISMIR2016
        Parameters
    ----------
    timesHF0: timestamps of each frame in HF0 matrix
    HF0init: Nbins*Nframes
    timesHSSF: timestamps of each frame in HSSF matrix
    HSSF: Nframes2*Nbins2
    Ideally they should have same number of bins
    Simple resizing of matrices if necessary
    See the effect of G and mu in combine2 function
    doConvolution: True or False, to perform the convolution with a Gaussian

    Returns
    -------
    times: Timestamps of the frames of the combined salience function
    sal:  Combined Salience function
    """

    #
    tHF0, HF0in, tHSSF, HSSFin = simpleResize(timesHF0, HF0init, timesHSSF, HSSF)

    # Combine both matrices
    times, sal = combine2(tHF0, HF0in, tHSSF, HSSFin, G, mu, doConvolution)
    return times, sal

