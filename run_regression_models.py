import sys; sys.path
import pandas as pd
import numpy as np 
import seaborn as sns
import scipy.io as sio
import os
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold, StratifiedKFold
from sklearn import preprocessing, linear_model
from sklearn.metrics import explained_variance_score, r2_score, make_scorer
from sklearn.preprocessing import StandardScaler
from joblib import parallel_backend, Parallel, delayed
from helper_functions import *
from data_formatting import *

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
# predictions for SC-  PCA and correlation

import datetime
import pickle 

[X, y, C, site] = create_data_set(atlas='shen268',y_var = 'severity', covariates=['SEX', 'AGE', 'CHRONICITY'], verbose = True, chaco_type = 'chacovol', subset = 'all')
print(X.shape)