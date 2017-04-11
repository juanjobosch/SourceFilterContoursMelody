# Most original code by J.L. Durrieu, modified by Juan J. Bosch in February, 2015

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import sys

import numpy as np

import SIMM
#import scikits.audiolab
##if (scipy.__version__) < '0.8':
##    raise ImportError('Version of scipy is %s, to read wavfile, one needs >= 0.8' %(scipy.__version__))
import os

from tracking import viterbiTrackingArray


# SOME USEFUL, INSTRUMENTAL, FUNCTIONS

def db(val):
    """
    db(positiveValue)

    Returns the decibel value of the input positiveValue
    """
    return 10 * np.log10(val)

def nextpow2(i):
    """
    Find 2^n that is equal to or greater than.

    code taken from the website:
    http://www.phys.uu.nl/~haque/computing/WPark_recipes_in_python.html
    """
    n = 2
    while n < i:
        n = n * 2
    return n

def ISDistortion(X,Y):
    """
    value = ISDistortion(X, Y)

    Returns the value of the Itakura-Saito (IS) divergence between
    matrix X and matrix Y. X and Y should be two NumPy arrays with
    same dimension.
    """
    return sum((-np.log(X / Y) + (X / Y) - 1))

# DEFINING SOME WINDOW FUNCTIONS

def sinebell(lengthWindow):
    """
    window = sinebell(lengthWindow)

    Computes a "sinebell" window function of length L=lengthWindow

    The formula is:
        window(t) = sin(pi * t / L), t = 0..L-1
    """
    window = np.sin((np.pi * (np.arange(lengthWindow))) \
                    / (1.0 * lengthWindow))
    return window

def hann(args):
    """
    window = hann(args)

    Computes a Hann window, with NumPy's function hanning(args).
    """
    return np.hanning(args)

# FUNCTIONS FOR TIME-FREQUENCY REPRESENTATION

def stft(data, window=sinebell(2048), hopsize=256.0, nfft=2048.0, \
         fs=44100.0):
    """
    X, F, N = stft(data, window=sinebell(2048), hopsize=1024.0,
                   nfft=2048.0, fs=44100)

    Computes the short time Fourier transform (STFT) of data.

    Inputs:
        data                  : one-dimensional time-series to be
                                analyzed
        window=sinebell(2048) : analysis window
        hopsize=1024.0        : hopsize for the analysis
        nfft=2048.0           : number of points for the Fourier
                                computation (the user has to provide an
                                even number)
        fs=44100.0            : sampling rate of the signal

    Outputs:
        X                     : STFT of data
        F                     : values of frequencies at each Fourier
                                bins
        N                     : central time at the middle of each
                                analysis window
    """

    # window defines the size of the analysis windows
    lengthWindow = window.size

    # !!! adding zeros to the beginning of data, such that the first
    # window is centered on the first sample of data
    data = np.concatenate((np.zeros(int(lengthWindow / 2.0)),data))
    lengthData = data.size

    # adding one window for the last frame (same reason as for the
    # first frame)
    numberFrames = int(np.ceil((lengthData - lengthWindow) / hopsize \
                           + 1) + 1)
    newLengthData = (numberFrames - 1) * hopsize + lengthWindow
    # zero-padding data such that it holds an exact number of frames
    data = np.concatenate((data, np.zeros([int(newLengthData - lengthData)])))

    # the output STFT has nfft/2+1 rows. Note that nfft has to be an
    # even number (and a power of 2 for the fft to be fast)
    numberFrequencies = int(nfft / 2.0 + 1)

    STFT = np.zeros([numberFrequencies, numberFrames], dtype=complex)

    for n in np.arange(numberFrames):
        beginFrame = int(n * hopsize)
        endFrame = beginFrame + lengthWindow
        frameToProcess = window * data[beginFrame:endFrame]
        STFT[:,n] = np.fft.rfft(frameToProcess, nfft);

    F = np.arange(numberFrequencies) / nfft * fs
    N = np.arange(numberFrames) * hopsize / fs

    return STFT, F, N

def istft(X, window=sinebell(2048), hopsize=256.0, nfft=2048.0):
    """
    data = istft(X, window=sinebell(2048), hopsize=256.0, nfft=2048.0)

    Computes an inverse of the short time Fourier transform (STFT),
    here, the overlap-add procedure is implemented.

    Inputs:
        X                     : STFT of the signal, to be "inverted"
        window=sinebell(2048) : synthesis window
                                (should be the "complementary" window
                                for the analysis window)
        hopsize=1024.0        : hopsize for the analysis
        nfft=2048.0           : number of points for the Fourier
                                computation
                                (the user has to provide an even number)

    Outputs:
        data                  : time series corresponding to the given
                                STFT the first half-window is removed,
                                complying with the STFT computation
                                given in the function 'stft'
    """
    lengthWindow = np.array(window.size)
    numberFrequencies, numberFrames = np.array(X.shape)
    lengthData = hopsize * (numberFrames - 1) + lengthWindow

    data = np.zeros(lengthData)
    for n in np.arange(numberFrames):
        beginFrame = n * hopsize
        endFrame = beginFrame + lengthWindow
        frameTMP = np.fft.irfft(X[:,n], nfft)
        frameTMP = frameTMP[:lengthWindow]
        data[beginFrame:endFrame] = data[beginFrame:endFrame] \
                                    + window * frameTMP

    # remove the extra bit before data that was - supposedly - added
    # in the stft computation:
    data = data[(lengthWindow / 2.0):]
    return data

# DEFINING THE FUNCTIONS TO CREATE THE 'BASIS' WF0

def generate_WF0_chirped(minF0, maxF0, Fs, Nfft=2048, stepNotes=4, \
                         lengthWindow=2048, Ot=0.5, perF0=2, \
                         depthChirpInSemiTone=0.5, loadWF0=True,
                         analysisWindow='hanning'):
    """
    F0Table, WF0 = generate_WF0_chirped(minF0, maxF0, Fs, Nfft=2048,
                                        stepNotes=4, lengthWindow=2048,
                                        Ot=0.5, perF0=2,
                                        depthChirpInSemiTone=0.5)

    Generates a 'basis' matrix for the source part WF0, using the
    source model KLGLOTT88, with the following I/O arguments:
    Inputs:
        minF0                the minimum value for the fundamental
                             frequency (F0)
        maxF0                the maximum value for F0
        Fs                   the desired sampling rate
        Nfft                 the number of bins to compute the Fourier
                             transform
        stepNotes            the number of F0 per semitone
        lengthWindow         the size of the window for the Fourier
                             transform
        Ot                   the glottal opening coefficient for
                             KLGLOTT88
        perF0                the number of chirps considered per F0
                             value
        depthChirpInSemiTone the maximum value, in semitone, of the
                             allowed chirp per F0

    Outputs:
        F0Table the vector containing the values of the fundamental
                frequencies in Hertz (Hz) corresponding to the
                harmonic combs in WF0, i.e. the columns of WF0
        WF0     the basis matrix, where each column is a harmonic comb
                generated by KLGLOTT88 (with a sinusoidal model, then
                transformed into the spectral domain)
    """
    # generating a filename to keep data:
    filename = str('').join(['wf0_',
                             '_minF0-', str(minF0),
                             '_maxF0-', str(maxF0),
                             '_Fs-', str(Fs),
                             '_Nfft-', str(Nfft),
                             '_stepNotes-', str(stepNotes),
                             '_Ot-', str(Ot),
                             '_perF0-', str(perF0),
                             '_depthChirp-', str(depthChirpInSemiTone),
                             '_analysisWindow-', analysisWindow,
                             '.npz'])

    if os.path.isfile(filename) and loadWF0:
        struc = np.load(filename)
        return struc['F0Table'], struc['WF0']


    # converting to double arrays:
    minF0=np.double(minF0)
    maxF0=np.double(maxF0)
    Fs=np.double(Fs)
    stepNotes=np.double(stepNotes)

    # computing the F0 table:
    numberOfF0 = np.ceil(12.0 * stepNotes * np.log2(maxF0 / minF0)) + 1
    F0Table=minF0 * (2 ** (np.arange(numberOfF0,dtype=np.double) \
                           / (12 * stepNotes)))

    numberElementsInWF0 = numberOfF0 * perF0

    # computing the desired WF0 matrix
    WF0 = np.zeros([Nfft, numberElementsInWF0],dtype=np.double)
    for fundamentalFrequency in np.arange(numberOfF0):
        odgd, odgdSpec = \
              generate_ODGD_spec(F0Table[fundamentalFrequency], Fs, \
                                 Ot=Ot, lengthOdgd=lengthWindow, \
                                 Nfft=Nfft, t0=0.0,\
                                 analysisWindowType=analysisWindow) # 20100924 trying with hann window
        WF0[:,fundamentalFrequency * perF0] = np.abs(odgdSpec) ** 2
        for chirpNumber in np.arange(perF0 - 1):
            F2 = F0Table[fundamentalFrequency] \
                 * (2 ** ((chirpNumber + 1.0) * depthChirpInSemiTone \
                          / (12.0 * (perF0 - 1.0))))
            # F0 is the mean of F1 and F2.
            F1 = 2.0 * F0Table[fundamentalFrequency] - F2
            odgd, odgdSpec = \
                  generate_ODGD_spec_chirped(F1, F2, Fs, \
                                             Ot=Ot, \
                                             lengthOdgd=lengthWindow, \
                                             Nfft=Nfft, t0=0.0)
            WF0[:,fundamentalFrequency * perF0 + chirpNumber + 1] = \
                                       np.abs(odgdSpec) ** 2

    np.savez(filename, F0Table=F0Table, WF0=WF0)

    return F0Table, WF0

def generate_ODGD_spec(F0, Fs, lengthOdgd=2048, Nfft=2048, Ot=0.5, \
                       t0=0.0, analysisWindowType='sinebell'):
    """
    generateODGDspec:

    generates a waveform ODGD and the corresponding spectrum,
    using as analysis window the -optional- window given as
    argument.
    """

    # converting input to double:
    F0 = np.double(F0)
    Fs = np.double(Fs)
    Ot = np.double(Ot)
    t0 = np.double(t0)

    # compute analysis window of given type:
    if analysisWindowType=='sinebell':
        analysisWindow = sinebell(lengthOdgd)
    else:
        if analysisWindowType=='hanning' or \
               analysisWindowType=='hanning':
            analysisWindow = hann(lengthOdgd)

    # maximum number of partials in the spectral comb:
    partialMax = np.floor((Fs / 2) / F0)

    # Frequency numbers of the partials:
    frequency_numbers = np.arange(1,partialMax + 1)

    # intermediate value
    temp_array = 1j * 2.0 * np.pi * frequency_numbers * Ot

    # compute the amplitudes for each of the frequency peaks:
    amplitudes = F0 * 27 / 4 \
                 * (np.exp(-temp_array) \
                    + (2 * (1 + 2 * np.exp(-temp_array)) / temp_array) \
                    - (6 * (1 - np.exp(-temp_array)) \
                       / (temp_array ** 2))) \
                       / temp_array

    # Time stamps for the time domain ODGD
    timeStamps = np.arange(lengthOdgd) / Fs + t0 / F0

    # Time domain odgd:
    odgd = np.exp(np.outer(2.0 * 1j * np.pi * F0 * frequency_numbers, \
                           timeStamps)) \
                           * np.outer(amplitudes, np.ones(lengthOdgd))
    odgd = np.sum(odgd, axis=0)

    # spectrum:
    odgdSpectrum = np.fft.fft(np.real(odgd * analysisWindow), n=Nfft)

    return odgd, odgdSpectrum

def generate_ODGD_spec_chirped(F1, F2, Fs, lengthOdgd=2048, Nfft=2048, \
                               Ot=0.5, t0=0.0, \
                               analysisWindowType='sinebell'):
    """
    generateODGDspecChirped:

    generates a waveform ODGD and the corresponding spectrum,
    using as analysis window the -optional- window given as
    argument.
    """

    # converting input to double:
    F1 = np.double(F1)
    F2 = np.double(F2)
    F0 = np.double(F1 + F2) / 2.0
    Fs = np.double(Fs)
    Ot = np.double(Ot)
    t0 = np.double(t0)

    # compute analysis window of given type:
    if analysisWindowType == 'sinebell':
        analysisWindow = sinebell(lengthOdgd)
    else:
        if analysisWindowType == 'hanning' or \
               analysisWindowType == 'hann':
            analysisWindow = hann(lengthOdgd)

    # maximum number of partials in the spectral comb:
    partialMax = np.floor((Fs / 2) / np.max(F1, F2))

    # Frequency numbers of the partials:
    frequency_numbers = np.arange(1,partialMax + 1)

    # intermediate value
    temp_array = 1j * 2.0 * np.pi * frequency_numbers * Ot

    # compute the amplitudes for each of the frequency peaks:
    amplitudes = F0 * 27 / 4 * \
                 (np.exp(-temp_array) \
                  + (2 * (1 + 2 * np.exp(-temp_array)) / temp_array) \
                  - (6 * (1 - np.exp(-temp_array)) \
                     / (temp_array ** 2))) \
                  / temp_array

    # Time stamps for the time domain ODGD
    timeStamps = np.arange(lengthOdgd) / Fs + t0 / F0

    # Time domain odgd:
    odgd = np.exp(2.0 * 1j * np.pi \
                  * (np.outer(F1 * frequency_numbers,timeStamps) \
                     + np.outer((F2 - F1) \
                                * frequency_numbers,timeStamps ** 2) \
                     / (2 * lengthOdgd / Fs))) \
                     * np.outer(amplitudes,np.ones(lengthOdgd))
    odgd = np.sum(odgd,axis=0)

    # spectrum:
    odgdSpectrum = np.fft.fft(np.real(odgd * analysisWindow), n=Nfft)

    return odgd, odgdSpectrum

def generateHannBasis(numberFrequencyBins, sizeOfFourier, Fs, \
                      frequencyScale='linear', numberOfBasis=20, \
                      overlap=.75):
    isScaleRecognized = False
    if frequencyScale == 'linear':
        # number of windows generated:
        numberOfWindowsForUnit = np.ceil(1.0 / (1.0 - overlap))
        # recomputing the overlap to exactly fit the entire
        # number of windows:
        overlap = 1.0 - 1.0 / np.double(numberOfWindowsForUnit)
        # length of the sine window - that is also to say: bandwidth
        # of the sine window:
        lengthSineWindow = np.ceil(numberFrequencyBins \
                                   / ((1.0 - overlap) \
                                      * (numberOfBasis - 1) + 1 \
                                      - 2.0 * overlap))
        # even window length, for convenience:
        lengthSineWindow = int(2.0 * np.floor(lengthSineWindow / 2.0))

        # for later compatibility with other frequency scales:
        mappingFrequency = np.arange(numberFrequencyBins)

        # size of the "big" window
        sizeBigWindow = int(2.0 * numberFrequencyBins)

        # centers for each window
        ## the first window is centered at, in number of window:
        firstWindowCenter = -numberOfWindowsForUnit + 1
        ## and the last is at
        lastWindowCenter = numberOfBasis - numberOfWindowsForUnit + 1
        ## center positions in number of frequency bins
        sineCenters = np.round(\
            np.arange(firstWindowCenter, lastWindowCenter) \
            * (1 - overlap) * np.double(lengthSineWindow) \
            + lengthSineWindow / 2.0)

        # For future purpose: to use different frequency scales
        isScaleRecognized = True

    # For frequency scale in logarithm (such as ERB scales)
    if frequencyScale == 'log':
        isScaleRecognized = False

    # checking whether the required scale is recognized
    if not(isScaleRecognized):
        print "The desired feature for frequencyScale is not recognized yet..."
        return 0

    # the shape of one window:
    prototypeSineWindow = hann(lengthSineWindow)
    # adding zeroes on both sides, such that we do not need to check
    # for boundaries
    bigWindow = np.zeros([sizeBigWindow * 2, 1])
    bigWindow[int(sizeBigWindow - lengthSineWindow / 2.0):\
              int(sizeBigWindow + lengthSineWindow / 2.0)] \
              = np.vstack(prototypeSineWindow)

    WGAMMA = np.zeros([numberFrequencyBins, numberOfBasis])

    for p in np.arange(numberOfBasis):
        WGAMMA[:, p] = np.hstack(bigWindow[np.int32(mappingFrequency \
                                                    - sineCenters[p] \
                                                    + sizeBigWindow)])

    return WGAMMA

def main(args,options):

    stereoEstimation = True

    # Median filtering in spectrogram
    HPS = False

    displayEvolution = options.displayEvolution
    if displayEvolution:
        import matplotlib.pyplot as plt
        import imageMatlab

        ## plt.rc('text', usetex=True)
        plt.rc('image',cmap='jet') ## gray_r
        plt.ion()

    # Compulsory option: name of the input file:
    inputAudioFile = ''
    if len(args) >= 2:
        inputAudioFile = args[0]
        options.pitch_output_file=args[1]
    if len(args) == 1:
        inputAudioFile = args[0]
    if len(args) == 0:
        inputAudioFile = options.input_file

    if inputAudioFile[-4:] != ".wav":
        raise ValueError("File not WAV file? Only WAV format support, for now...")

    #print "Writing the different following output files:"
    if not(options.vit_pitch_output_file is None):
        print "    estimated pitches in", options.vit_pitch_output_file
    if not(options.sal_output_file is None):
        print "    salience file in ", options.sal_output_file

    if options.pitch_output_file is None:
        options.pitch_output_file = inputAudioFile[:-4]+'_pitches.txt'

    try:
        from essentia.standard import AudioLoader
        loaded = AudioLoader(filename = inputAudioFile)()
        audio = loaded[0]
        Fs = loaded[1]
        nchan = loaded[2]
        loaded = AudioLoader(filename = inputAudioFile)()
        audio = loaded[0]
        if nchan == 1:
            data = audio[:,0].transpose()
        else:
            data = audio.transpose()

        data = np.double(data)/(1.2*abs(data).max())
    except:
        # Using scipy to import wav
        import scipy.io.wavfile as wav
        Fs, data = wav.read(inputAudioFile)
        # data = np.double(data) /  32768.0 # makes data vary from -1 to 1
        scaleData = 1.2 * data.max() # to rescale the data.
        data = np.double(data) / scaleData # makes data vary from -1 to 1
    options.Fs = Fs
    is_stereo = True
    if data.shape[0] == data.size: # data is multi-channel
        #print "The audio file is not stereo."
        #print "The audio file is not stereo. Making stereo out of mono."
        #print "(You could also try the older separateLead.py...)"
        is_stereo = False
        # data = np.vstack([data,data]).T
        # raise ValueError("number of dimensions of the input not 2")
    if is_stereo and data.shape[1] != 2:
        print "The data is multichannel, but not stereo... \n"
        print "Unfortunately this program does not scale well. Data is \n"
        print "reduced to its 2 first channels.\n"
        data = data[:,0:2]

        
    # Processing the options:
    windowSizeInSamples = nextpow2(np.round(options.windowSize * Fs))

    hopsize = np.round(options.hopsize * Fs)
    #if hopsize != windowSizeInSamples/8:
    #    #print "Overriding given hopsize to use 1/8th of window size"
    #    #hopsize = windowSizeInSamples/8
    #    warnings.warn("Chosen hopsize: "+str(hopsize)+\
    #                  ", while windowsize: "+str(windowSizeInSamples))

    options.hopsizeInSamples = hopsize
    if options.fourierSize is None:
        NFT = windowSizeInSamples
    else:
        NFT = options.fourierSize

    # number of iterations for each parameter estimation step:
    niter = options.nbiter
    # number of spectral shapes for the accompaniment
    R = int(options.R)

    eps = 10 ** -9

    if options.verbose:
        print "Some parameter settings:"
        print "    Size of analysis windows: ", windowSizeInSamples
        print "    Hopsize: ", hopsize
        print "    Size of Fourier transforms: ", NFT
        print "    Number of iterations to be done: ", niter
        print "    Number of elements in WM: ", R

    if is_stereo:
        XR, F, N = stft(data[:,0], fs=Fs, hopsize=hopsize,
                        window=sinebell(windowSizeInSamples), nfft=NFT)
        XL, F, N = stft(data[:,1], fs=Fs, hopsize=hopsize,
                        window=sinebell(windowSizeInSamples), nfft=NFT)
        # SX is the power spectrogram:
        ## SXR = np.maximum(np.abs(XR) ** 2, 10 ** -8)
        ## SXL = np.maximum(np.abs(XL) ** 2, 10 ** -8)
        #SXR = np.abs(XR) ** 2
        #SXL = np.abs(XL) ** 2
        SX = np.maximum((0.5*np.abs(XR+XL)) ** 2, eps)
    else: # data is mono
        X, F, N = stft(data, fs=Fs, hopsize=hopsize,
                       window=sinebell(windowSizeInSamples), nfft=NFT)
        SX = np.maximum(np.abs(X) ** 2, eps)

    del data, F, N

    # minimum and maximum F0 in glottal source spectra dictionary
    minF0 = options.minF0
    maxF0 = options.maxF0
    F, N = SX.shape
    stepNotes = options.stepNotes # this is the number of F0s within one semitone

    K = int(options.K_numFilters) # number of spectral shapes for the filter part
    P = int(options.P_numAtomFilters) # number of elements in dictionary of smooth filters
    chirpPerF0 = 1 # number of chirped spectral shapes between each F0
    # this feature should be further studied before
    # we find a good way of doing that.

    # Create the harmonic combs, for each F0 between minF0 and maxF0:
    F0Table, WF0 = \
             generate_WF0_chirped(minF0, maxF0, Fs, Nfft=NFT, \
                                  stepNotes=stepNotes, \
                                  lengthWindow=windowSizeInSamples, Ot=0.25, \
                                  perF0=chirpPerF0, \
                                  depthChirpInSemiTone=.15, loadWF0=True,\
                                  analysisWindow='sinebell')
    WF0 = WF0[0:F, :] # ensure same size as SX
    NF0 = F0Table.size # number of harmonic combs
    # Normalization:
    WF0 = WF0 / np.outer(np.ones(F), np.amax(WF0, axis=0))

    # Create the dictionary of smooth filters, for the filter part of
    # the lead isntrument:
    WGAMMA = generateHannBasis(F, NFT, Fs=Fs, frequencyScale='linear', \
                               numberOfBasis=P, overlap=.75)


    if options.sal_output_file is None or not os.path.exists(options.sal_output_file):
        if displayEvolution:
            plt.figure(1);plt.clf()
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            plt.xlabel(r'Frame number $n$', fontsize=16)
            plt.ylabel(r'Leading source number $u$', fontsize=16)
            plt.ion()
            # plt.show()
            ## the following seems superfluous if mpl's backend is macosx...
            ##        raw_input("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"\
            ##                  "!! Press Return to resume the program. !!\n"\
            ##                  "!! Be sure that the figure has been    !!\n"\
            ##                  "!! already displayed, so that the      !!\n"\
            ##                  "!! evolution of HF0 will be visible.   !!\n"\
            ##                  "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")


        ## section to estimate the melody, on monophonic algo:
        # First round of parameter estimation:
        if(HPS):
            from scipy.signal import medfilt
        if (is_stereo & stereoEstimation):
            SXR = np.maximum(np.abs(XR) ** 2, eps)
            SXL = np.maximum(np.abs(XL) ** 2, eps)
            if(HPS):
                SXR= medfilt(SXR,3)
                SXL= medfilt(SXL,3)

            alphaR, alphaL, HGAMMA, HPHI, HF0, betaR, betaL, HM, WM, recoError1 = SIMM.Stereo_SIMM(
                # the data to be fitted to:
                SXR, SXL,
                # the basis matrices for the spectral combs
                WF0,
                # and for the elementary filters:
                WGAMMA,
                # number of desired filters, accompaniment spectra:
                numberOfFilters=K, numberOfAccompanimentSpectralShapes=R,
                # putting only 2 elements in accompaniment for a start...
                # if any, initial amplitude matrices for
                HGAMMA0=None, HPHI0=None,
                HF00=None,
                WM0=None, HM0=None,
                # Some more optional arguments, to control the "convergence"
                # of the algo
                numberOfIterations=niter, updateRulePower=1.,
                stepNotes=stepNotes,
                lambdaHF0 = 0.0 / (1.0 * SX.max()), alphaHF0=0.9,
                verbose=options.verbose, displayEvolution=displayEvolution)
        else:
            if(HPS):
                SX= medfilt(SX,3)

            HGAMMA, HPHI, HF0, HM, WM, recoError1 = SIMM.SIMM(
                # the data to be fitted to:
                SX,
                # the basis matrices for the spectral combs
                WF0,
                # and for the elementary filters:
                WGAMMA,
                # number of desired filters, accompaniment spectra:
                numberOfFilters=K, numberOfAccompanimentSpectralShapes=R,
                # putting only 2 elements in accompaniment for a start...
                # if any, initial amplitude matrices for
                HGAMMA0=None, HPHI0=None,
                HF00=None,
                WM0=None, HM0=None,
                # Some more optional arguments, to control the "convergence"
                # of the algo
                numberOfIterations=niter, updateRulePower=1.,
                stepNotes=stepNotes,
                lambdaHF0 = 0.0 / (1.0 * SX.max()), alphaHF0=0.9,
                verbose=options.verbose, displayEvolution=displayEvolution)
        if displayEvolution:
            h2 = plt.figure(2);plt.clf();
            imageMatlab.imageM(20 * np.log10(HF0))
            matMax = (20 * np.log10(HF0)).max()
            matMed = np.median(20 * np.log10(HF0))
            plt.clim([matMed - 100, matMax])

    else:
        print "Loading Salience from file to calculate Melody: " + options.sal_output_file
        loaded = np.loadtxt(options.sal_output_file).T
        times = [loaded[0,:]]
        HF0 = loaded[1:,:]


    # If vit_pitch_output_file is not null, do melody extraction with Viterbi
    if not(options.vit_pitch_output_file is None):
        print "Viterbi decoding"
        # Viterbi decoding to estimate the predominant fundamental
        # frequency line
        # create transition probability matrix - adhoc parameter 'scale'
        # TODO: use "learned" parameter scale (NB: after many trials,
        # provided scale and parameterization seems robust)
        scale = 1.0
        transitions = np.exp(-np.floor(np.arange(0,NF0) / stepNotes) * scale)
        cutoffnote = 2 * 5 * stepNotes
        transitions[cutoffnote:] = transitions[cutoffnote - 1]

        transitionMatrixF0 = np.zeros([NF0 + 1, NF0 + 1]) # toeplitz matrix
        b = np.arange(NF0)
        transitionMatrixF0[0:NF0, 0:NF0] = \
                                  transitions[\
            np.array(np.abs(np.outer(np.ones(NF0), b) \
                            - np.outer(b, np.ones(NF0))), dtype=int)]
        pf_0 = transitions[cutoffnote - 1] * 10 ** (-90)
        p0_0 = transitions[cutoffnote - 1] * 10 ** (-100)
        p0_f = transitions[cutoffnote - 1] * 10 ** (-80)
        transitionMatrixF0[0:NF0, NF0] = pf_0
        transitionMatrixF0[NF0, 0:NF0] = p0_f
        transitionMatrixF0[NF0, NF0] = p0_0

        sumTransitionMatrixF0 = np.sum(transitionMatrixF0, axis=1)
        transitionMatrixF0 = transitionMatrixF0 \
                             / np.outer(sumTransitionMatrixF0, \
                                        np.ones(NF0 + 1))

        # prior probabilities, and setting the array for Viterbi tracking:
        priorProbabilities = 1 / (NF0 + 1.0) * np.ones([NF0 + 1])
        logHF0 = np.zeros([NF0 + 1, N])
        normHF0 = np.amax(HF0, axis=0)
        barHF0 = np.array(HF0)

        logHF0[0:NF0, :] = np.log(barHF0)
        logHF0[0:NF0, normHF0==0] = np.amin(logHF0[logHF0>-np.Inf])
        logHF0[NF0, :] = np.maximum(np.amin(logHF0[logHF0>-np.Inf]),-100)

        indexBestPath = viterbiTrackingArray(\
            logHF0, np.log(priorProbabilities),
            np.log(transitionMatrixF0), verbose=options.verbose)

        if displayEvolution:
            h2.hold(True)
            plt.plot(indexBestPath, '-b')
            h2.hold(False)
            plt.axis('tight')

        del logHF0

        # detection of silences:
        # computing the melody restricted F0 amplitude matrix HF00
        # (which will be used as initial HF0 for further algo):
        HF00 = np.zeros([NF0 * chirpPerF0, N])
        scopeAllowedHF0 = 2.0 / 1.0
        # computing indices for and around the melody indices,
        # dim1index are indices along axis 0, and dim2index along axis 1
        # of HF0:
        #     TODO: use numpy broadcasting to make this "clearer" (if possible...)
        dim1index = np.array(\
            np.maximum(\
                np.minimum(\
                    np.outer(chirpPerF0 * indexBestPath,
                             np.ones(chirpPerF0 \
                                     * (2 \
                                        * int(np.floor(stepNotes / scopeAllowedHF0)) \
                                        + 1))) \
                    + np.outer(np.ones(N),
                               np.arange(-chirpPerF0 \
                                         * int(np.floor(stepNotes / scopeAllowedHF0)),
                                         chirpPerF0 \
                                         * int((np.floor(stepNotes / scopeAllowedHF0))) \
                                            + 1)),
                    chirpPerF0 * NF0 - 1),
                0),
            dtype=int).reshape(1, N * chirpPerF0 \
                               * (2 * int(np.floor(stepNotes / scopeAllowedHF0)) \
                                  + 1))
        dim2index = np.outer(np.arange(N),
                             np.ones(chirpPerF0 \
                                     * (2 * int(np.floor(stepNotes \
                                                     / scopeAllowedHF0)) + 1), \
                                     dtype=int)\
                             ).reshape(1, N * chirpPerF0 \
                                       * (2 * int(np.floor(stepNotes \
                                                       / scopeAllowedHF0)) \
                                          + 1))
        HF00[dim1index, dim2index] = HF0[dim1index, dim2index]# HF0.max()

        HF00[:, indexBestPath == (NF0 - 1)] = 0.0
        HF00[:, indexBestPath == 0] = 0.0

        # remove frames with less than (100 thres_energy) % of total energy.
        thres_energy = 0.000584
        SF0 = np.maximum(np.dot(WF0, HF00), eps)
        SPHI = np.maximum(np.dot(WGAMMA, np.dot(HGAMMA, HPHI)), eps)
        SM = np.maximum(np.dot(WM, HM), eps)
        hatSX = np.maximum(SPHI * SF0 + SM, eps)

        energyMel = np.sum((((SPHI * SF0)/hatSX)**2) * SX, axis=0)
        energyMelSorted = np.sort(energyMel)
        energyMelCumul = np.cumsum(energyMelSorted)
        energyMelCumulNorm = energyMelCumul / max(energyMelCumul[-1], eps)
        # normalized to the maximum of energy:
        # expressed in 0.01 times the percentage
        ind_999 = np.nonzero(energyMelCumulNorm>thres_energy)[0][0]
        if ind_999 is None:
            ind_999 = N
        if not os.path.isdir(os.path.dirname((options.vit_pitch_output_file))):
            os.mkdir(os.path.dirname((options.vit_pitch_output_file)))

        np.savetxt(options.vit_pitch_output_file+'.egy',
                   np.array([np.arange(N) * hopsize / np.double(Fs),
                             energyMel]).T,fmt='%10.5f')

        # energyMel <= energyMelCumul[ind_999]?

        melNotPresent = (energyMel <= energyMelCumulNorm[ind_999])

        # edit: frames predicted as unvoiced will be given negative values
        # indexBestPath[melNotPresent] = 0

        freqMelody = F0Table[np.array(np.minimum(indexBestPath,len(F0Table)-1),dtype=int)]
        freqMelody[melNotPresent] = - freqMelody[melNotPresent]

        if not os.path.exists(os.path.dirname(options.vit_pitch_output_file)):
            os.makedirs(os.path.dirname(options.vit_pitch_output_file))

        np.savetxt(options.vit_pitch_output_file,
                   np.array([np.arange(N) * hopsize / np.double(Fs),
                             freqMelody]).T,fmt='%10.7f')

    times = np.array([np.arange(N) * hopsize / np.double(Fs)])

    # Save salience file:
    if not(options.sal_output_file is None):
        if not os.path.exists(os.path.dirname(options.sal_output_file)):
            os.makedirs(os.path.dirname(options.sal_output_file))
        np.savetxt(options.sal_output_file,np.concatenate((times,HF0),axis=0).T,fmt='%10.6f')
        # saveSPHI (timbre related)
        saveSPHI = 0
        if saveSPHI:
            if not os.path.exists(os.path.dirname(options.sal_output_file+'.SPHI')):
                os.makedirs(os.path.dirname(options.sal_output_file))
            WPHI = np.dot(WGAMMA, HGAMMA)
            SPHI = np.dot(WPHI, HPHI)
            np.savetxt(options.sal_output_file+'.SPHI',np.concatenate((times,SPHI),axis=0).T,fmt='%10.4f')
        #np.savetxt(options.sal_output_file+'.WGAMMA',np.concatenate((times,WGAMMA),axis=0).T,fmt='%10.4f')

    # return times[0],freqMelody,HF0
    print "Done!"
    return times[0],HF0,options

if __name__ == '__main__':
    main(sys.argv[1:])



