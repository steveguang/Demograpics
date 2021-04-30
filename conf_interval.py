import numpy as np
import scipy.stats


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, h

"""
data = [0.7934749505101264, 0.7784249530480687, 0.8185371300949189, 0.7203314552560783, 0.7803030303030303]
print (mean_confidence_interval(data, 0.95))
data = [0.7466117216117215, 0.7271520146520146, 0.7703296703296703, 0.8085622710622712, 0.7788461538461539]
print (mean_confidence_interval(data, confidence=0.95))
data = [0.7272435897435898, 0.7752289377289378, 0.7894688644688644, 0.7752289377289379, 0.798076923076923]
print (mean_confidence_interval(data, confidence=0.95))
"""
data = [0.8357679625795569, 0.7771643278889656, 0.8322981366459627, 0.8043478260869565, 0.8354037267080746]
print (mean_confidence_interval(data, 0.95))
data = [0.8410714285714286, 0.8321428571428571, 0.830357142857143, 0.8372567844342038, 0.8229390681003584]
print (mean_confidence_interval(data, 0.95))
