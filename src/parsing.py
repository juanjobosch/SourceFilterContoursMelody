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


import optparse

def parseOptions(argsin,wavfilerequired = False):

    usage = "usage: %prog [options] inputAudioFile"
    usage = "usage: %prog [options]"
    parser = optparse.OptionParser(usage)
    # Name of the output files:
    parser.add_option("-i", "--input-file",
                      dest="input_file", type="string",
                      help="Path of the input file.\n",
                      default=None)
    parser.add_option("-o", "--pitch-output",
                      dest="pitch_output_file", type="string",
                      help="name of the output file for an external algorithm.\n"
                           "If None appends _pitches to the wav",
                      default=None)
    parser.add_option("-s", "--pitch-salience-output-file",
                      dest="sal_output_file", type="string",
                      help="name of the output file for the Salience File.\n"
                           "If None the salience file is not saved.",
                      default=None)

    parser.add_option("-v", "--vit-pitch-output-file",
                      dest="vit_pitch_output_file", type="string",
                      help="name of the output file for the estimated pitches with Viterbi.\n"
                           "If None it does not execute the Viterbi extraction",
                      default=None)

    parser.add_option("-p", "--pitch-output-file",
                      dest="pitch_output_file", type="string",
                      help="name of the output file for an external algorithm.\n"
                           "If None appends _pitches to the wav",
                      default=None)
    # Some more optional options:
    parser.add_option("-d", "--with-display", dest="displayEvolution",
                      action="store_true",help="display the figures",
                      default=False)
    parser.add_option("-q", "--quiet", dest="verbose",
                      action="store_false",
                      help="use to quiet all output verbose",
                      default=False)
    parser.add_option("--nb-iterations", dest="nbiter",
                      help="number of iterations", type="int",
                      default=20)

    parser.add_option("--expandHF0Val", dest="expandHF0Val",
                      help="value for expanding the distribution of the values of HF0", type="float",
                      default=1)

    parser.add_option("--window-size", dest="windowSize", type="float",
                      default=0.04644,help="size of analysis windows, in s.")
    parser.add_option("--Fourier-size", dest="fourierSize", type="int",
                      default=None,
                      help="size of Fourier transforms, "\
                           "in samples.")
    # parser.add_option("--hopsize", dest="hopsize", type="float",
    #                   default=0.0058,
    #                   help="size of the hop between analysis windows, in s.")
    parser.add_option("--hopsize", dest="hopsize", type="float",
                      default=0.01,
                      help="size of the hop between analysis windows, in s.")
    parser.add_option("--nb-accElements", dest="R", type="float",
                      default=40.0,
                      help="number of elements for the accompaniment.")
    parser.add_option("--numAtomFilters", dest="P_numAtomFilters",
                      type="int", default=30,
                      help="Number of atomic filters - in WGAMMA.")
    parser.add_option("--numFilters", dest="K_numFilters", type="int",
                      default=10,
                      help="Number of filters for decomposition - in WPHI")
    parser.add_option("--min-F0-Freq", dest="minF0", type="float",
                      default=55.0,
                      help="Minimum of fundamental frequency F0.")
    parser.add_option("--max-F0-Freq", dest="maxF0", type="float",
                      default=1760.0,
                      help="Maximum of fundamental frequency F0.")
    parser.add_option("--samplingRate", dest="Fs", type="float",
                      default=44100,
                      help="Sampling rate")
    parser.add_option("--step-F0s", dest="stepNotes", type="int",
                      default=10,
                      help="Number of F0s in dictionary for each semitone.")
    # PitchContoursMelody
    parser.add_option("--voicingTolerance", dest="voicingTolerance", type="float",
                      default=0.2,
                      help="Allowed deviation below the average contour mean salience of all contours (fraction of the standard deviation)")

    #PitchContours
    parser.add_option("--peakDistributionThreshold", dest="peakDistributionThreshold", type="float",
                      default=0.9,
                      help="Allowed deviation below the peak salience mean over all frames (fraction of the standard deviation)")

    parser.add_option("--peakFrameThreshold", dest="peakFrameThreshold", type="float",
                      default=0.9,
                      help="Per-frame salience threshold factor (fraction of the highest peak salience in a frame)")

    parser.add_option("--minDuration", dest="minDuration", type="float",
                      default=100,
                      help="the minimum allowed contour duration [ms]")

    parser.add_option("--timeContinuity", dest="timeContinuity", type="float",
                      default=100,
                      help="Time continuity cue (the maximum allowed gap duration for a pitch contour) [ms]")
    parser.add_option("--voiceVibrato",dest = "voiceVibrato",default =False, help="detect voice vibrato for melody estimation")

    parser.add_option("--pitchContinuity", dest="pitchContinuity", type="float",
                      default=27.5625,
                      help="pitch continuity cue (maximum allowed pitch change durig 1 ms time period) [cents]")

    parser.add_option("--extractionMethod", dest="extractionMethod", type="string",
                      help="name of the method to be executed, if None, default is PCS (Pitch Contour Selection)",
                      default="PCS")

    (options, args) = parser.parse_args(argsin)

    options.hopsizeInSamples = int(round(options.hopsize*options.Fs))

    if ((len(args) < 1) & wavfilerequired):
        parser.error("incorrect number of arguments, use option -h for help.")

    return args, options


import optparse

def parseOptionsSS(argsin,wavfilerequired = True):

    usage = "usage: %prog [options] inputAudioFile"
    parser = optparse.OptionParser(usage)
    # Name of the output files:
    parser.add_option("-m", "--melody-output-file",
                      dest="solo_output_file", type="string",
                      help="name of the audio output file for the estimated\n"\
                           "solo (vocal) part",
                      default="estimated_solo.wav")
    parser.add_option("-a", "--accomp-output-file",
                      dest="acc_output_file", type="string",
                      help="name of the audio output file for the estimated\n"\
                           "music part",
                      default="estimated_music.wav")
    parser.add_option("-c", "--melodyPC-output-file",
                      dest="pc_pitch_output_file", type="string",
                      help="name of the output file for the estimated pitches with pitch contours\n",
                      default="pc.pitch")
    parser.add_option("-s", "--pitch-salience-output-file",
                      dest="sal_output_file", type="string",
                      help="name of the output file for the Salience File.\n"
                           "If None the salience file is not saved.",
                      default=None)
    parser.add_option("-v", "--vit-pitch-output-file",
                      dest="vit_pitch_output_file", type="string",
                      help="name of the output file for the estimated pitches with Viterbi.\n"
                           "If None it does not execute the Viterbi extraction",
                      default=None)

    #parser.add_option("-p", "--pitch-output-file",
    #                  dest="pitch_output_file", type="string",
    #                  help="name of the output file for an external algorithm.\n"
    #                       "If None appends _pitches to the wav",
    #                  default=None)
    # Some more optional options:
    parser.add_option("-d", "--with-display", dest="displayEvolution",
                      action="store_true",help="display the figures",
                      default=False)
    parser.add_option("-q", "--quiet", dest="verbose",
                      action="store_false",
                      help="use to quiet all output verbose",
                      default=False)
    parser.add_option("--nb-iterations", dest="nbiter",
                      help="number of iterations", type="int",
                      default=30)

    parser.add_option("--expandHF0Val", dest="expandHF0Val",
                      help="value for expanding the distribution of the values of HF0", type="float",
                      default=1)
    parser.add_option("--voiceVibrato",dest = "voiceVibrato",default =False, help="detect voice vibrato for melody estimation")
    parser.add_option("--window-size", dest="windowSize", type="float",
                      default=0.04644,help="size of analysis windows, in s.")
    parser.add_option("--Fourier-size", dest="fourierSize", type="int",
                      default=None,
                      help="size of Fourier transforms, "\
                           "in samples.")
    parser.add_option("--hopsize", dest="hopsize", type="float",
                      default=0.0058,
                      help="size of the hop between analysis windows, in s.")
    parser.add_option("--nb-accElements", dest="R", type="float",
                      default=40.0,
                      help="number of elements for the accompaniment.")
    parser.add_option("--numAtomFilters", dest="P_numAtomFilters",
                      type="int", default=30,
                      help="Number of atomic filters - in WGAMMA.")
    parser.add_option("--numFilters", dest="K_numFilters", type="int",
                      default=10,
                      help="Number of filters for decomposition - in WPHI")
    parser.add_option("--min-F0-Freq", dest="minF0", type="float",
                      default=100.0,
                      help="Minimum of fundamental frequency F0.")
    parser.add_option("--max-F0-Freq", dest="maxF0", type="float",
                      default=800.0,
                      help="Maximum of fundamental frequency F0.")
    parser.add_option("--samplingRate", dest="Fs", type="float",
                      default=44100,
                      help="Sampling rate")
    parser.add_option("--step-F0s", dest="stepNotes", type="int",
                      default=10,
                      help="Number of F0s in dictionary for each semitone.")
    # PitchContoursMelody
    parser.add_option("--voicingTolerance", dest="voicingTolerance", type="float",
                      default=0.2,
                      help="Allowed deviation below the average contour mean salience of all contours (fraction of the standard deviation)")

    #PitchContours
    parser.add_option("--peakDistributionThreshold", dest="peakDistributionThreshold", type="float",
                      default=0.9,
                      help="Allowed deviation below the peak salience mean over all frames (fraction of the standard deviation)")

    parser.add_option("--peakFrameThreshold", dest="peakFrameThreshold", type="float",
                      default=0.9,
                      help="Per-frame salience threshold factor (fraction of the highest peak salience in a frame)")

    parser.add_option("--minDuration", dest="minDuration", type="float",
                      default=100,
                      help="the minimum allowed contour duration [ms]")

    parser.add_option("--timeContinuity", dest="timeContinuity", type="float",
                      default=100,
                      help="Time continuity cue (the maximum allowed gap duration for a pitch contour) [ms]")

    parser.add_option("--pitchContinuity", dest="pitchContinuity", type="float",
                      default=27.5625,
                      help="pitch continuity cue (maximum allowed pitch change durig 1 ms time period) [cents]")
    (options, args) = parser.parse_args(argsin)

    options.hopsizeInSamples = int(round(options.hopsize*options.Fs))

    if (len(args) != 1 & wavfilerequired):
        parser.error("incorrect number of arguments, use option -h for help.")

    return args, options    