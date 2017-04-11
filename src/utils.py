import numpy as np

def loadMEFile(fileName):
    try:
        a = np.loadtxt(fileName)
    except:
        a = np.loadtxt(fileName,delimiter=',')
    if a.shape[1]>2:
        est_freq = a[:, 1:]
    else:
        est_freq = a[:, 1]
    est_time = a[:, 0]
    return est_time,est_freq