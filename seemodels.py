# check out models
from cgi import test
import numpy as np
from sklearn.model_selection import RepeatedKFold
from helper_functions import *
from data_formatting import *


tmp = np.load('/home/ubuntu/enigma/results/testrunmodel.npy',allow_pickle=True)
print(tmp)


tmp = np.load('/home/ubuntu/enigma/results/testrunscores.npy',allow_pickle=True)
print(np.mean(tmp[0]))

tmp = np.load('/home/ubuntu/enigma/results/testrunvariable_impts.npy',allow_pickle=True)
print(len(tmp))

