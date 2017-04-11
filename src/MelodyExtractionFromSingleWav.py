#!/usr/bin/env python
__author__ = "Juan Jose Bosch"
__email__ = "juan.bosch@upf.edu"

import sys

from numpy import savetxt, max, column_stack,tile
import utils
import SourceFilterModelSF
import combineSaliences
import melodyExtractionFromSalienceFunction
from HarmonicSummationSF import calculateSF
from os.path import join,dirname,basename
import parsing

def process(args):
    # options
    mu = 1
    G = 0
    doConvolution = True
    wavfile = args[0]

    (pargs,options) = parsing.parseOptions(args)

    # -------------------------

    if options.extractionMethod == "BG1":
        # Options MIREX 2016:  BG1
        options.pitchContinuity = 27.56
        options.peakDistributionThreshold = 1.3
        options.peakFrameThreshold = 0.7
        options.timeContinuity = 100
        options.minDuration = 100
        options.voicingTolerance = 1
        options.useVibrato = False
        options.decodingMethod = "PCS"
        options.combmode = 13

    if options.extractionMethod == "BG2":
        # Options MIREX 2016:  BG2
        options.pitchContinuity = 27.56
        options.peakDistributionThreshold = 0.9
        options.peakFrameThreshold = 0.9
        options.timeContinuity = 100
        options.minDuration = 100
        options.voicingTolerance = 0
        options.useVibrato = False
        options.decodingMethod = "PCS"
        options.combmode = 13

    if options.extractionMethod == "EWM":
        # Options MIREX 2016:  BG2
        options.combmode = 14
        options.decodingMethod = "PCS"

    if options.extractionMethod == "CBM":
        options.pitchContinuity = 27.56
        options.peakDistributionThreshold = 0.9
        options.peakFrameThreshold = 0.9
        options.timeContinuity = 50
        options.minDuration = 100
        options.voicingTolerance = 0.2
        options.useVibrato = False
        options.decodingMethod = "PCS"
        options.combmode = 13

    combmode = options.combmode

    # Compute salience functions --------------

    # Compute HF0 (SIMM with source-filter model)
    if options.combmode > 0:
        timesHF0, HF0, options = SourceFilterModelSF.main(pargs, options)
        # In order to have the same structure as the Harmonic Summation Salience Function
        HF0 = HF0[1:, :]

    if combmode != 4 and combmode != 5 and combmode != 14:
        # Computing Harmonic Summation salience function
        hopSizeinSamplesHSSF = int(min(options.hopsizeInSamples, 0.01 * options.Fs))
        timesHSSF, HSSF = calculateSF(wavfile, hopSizeinSamplesHSSF)
    else:
        print "Harmonic Summation Salience function not used"

    # Combination mode used in MIREX, ISMIR2016, SMC2016
    if combmode == 13:
        times, combSal = combineSaliences.combine3MIREX(timesHF0, HF0, timesHSSF, HSSF, G, mu, doConvolution)

    # Salience function by Durrieu, multiplying every frame by the estimated energy of the melody, used in SMC2016
    if combmode == 14:
        fileEnergy = options.vit_pitch_output_file+'.egy'
        #fileEnergy = join(dirname(options.sal_output_file),'ME-Viterbi/'+basename(options.sal_output_file)[:-4]+'pitch.egy')
        timesEnergy,energy = utils.loadMEFile(fileEnergy)
        times,combSal = combineSaliences.combine14(timesHF0,HF0, timesEnergy,tile(energy,(HF0.shape[0],1)).T, G,mu,doConvolution)

    combSal = combSal / max(combSal)

    print("Extracting melody from salience function")
    times, pitch = melodyExtractionFromSalienceFunction.MEFromSF(times, combSal, options)

    # Save output file
    savetxt(options.pitch_output_file, column_stack((times.T, pitch.T)), fmt='%-7.5f', delimiter="\t")
    print("Output file written")


def main(args):
    process(args)


if __name__ == '__main__':
    import time

    start_time = time.time()

    main(sys.argv[1:])
    print("Processing time: --- %s seconds ---" % (time.time() - start_time))
