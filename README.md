# SourceFilterContoursMelody
Melody extraction based on source-filter modelling


This repository contains the code of the algorithm evaluated in MIREX 2015 and 2016 (BG).
It also contains the code necessary to run the experiments in the following article (ISMIR2016):

J. J. Bosch, R. M. Bittner, J. Salamon, and E. Gómez, "A Comparison of
Melody Extraction Methods Based on Source-Filter Modelling", in Proc.
17th International Society for Music Information Retrieval Conference
(ISMIR 2016), New York City, USA, Aug. 2016.

Author:
Juan J. Bosch
Music Technology Group, Universitat Pompeu Fabra, Barcelona
Contact: juan.bosch@upf.edu

This repository also contains code by R.M. Bittner (contour_classification folder), and J.L Durrieu, which has been adapted to the needs of the conducted experiments

The code is written in python (version 2.7), and presents the following dependencies:

Essentia 2.0.1 or newer, with python bindings (http://essentia.upf.edu/)
NumPy 1.8.2 (any relatively recent version should work)

For contour classification, the following packages are also used:

pandas
scipy
seaborn
sklearn

In order to execute the algorithm evaluated in MIREX 2016 (BG1 and BG2 submissions), it should be called from the folder which contains the source code, as:

python MelodyExtractionFromSingleWav.py /inputaudiofolder/audio1.wav /estimations/audio1.txt --extractionMethod='BG1' --hopsize=0.01 --nb-iterations=30

where %input is the path to a wav file, and output is the file with the estimated melody.

To run contour classification experiments, you should first compute and save the contours, and adapt the paths, as well as the hopsize.

python run_contour_training_melody_extraction.py
python run_glass_ceiling_experiment.py