import sys; sys.path
import pandas as pd
import numpy as np 
import seaborn as sns
import scipy.io as sio
import os
from sklearn.model_selection import RepeatedKFold, GroupShuffleSplit,ShuffleSplit,GroupKFold, LeaveOneGroupOut, train_test_split, GridSearchCV, cross_val_score, KFold, StratifiedKFold
from sklearn import preprocessing, linear_model
from sklearn.metrics import explained_variance_score, r2_score, make_scorer
from sklearn.preprocessing import StandardScaler
from joblib import parallel_backend, Parallel, delayed
from data_formatting import * 
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import time
from helper_functions import *
import importlib
import data_formatting
import helper_functions 
from imp import reload

reload(helper_functions)
reload(data_formatting)

## all settings:
y_var = 'normed_motor_scores'
subset = 'chronic'
models_tested = ['ensemble_reg']
verbose = True
covariates=['SEX', 'AGE']

lesionload = 'M1'
nperms=1

atlases = ['fs86subj']
chaco_types = ['chacoconn']
crossval_types =[ '1', '5']
null=-1
CSV_PATH = os.path.join("/home/ubuntu/enigma/Behaviour_Information_ALL_April7_2022_sorted_CSTll.csv")  # Set this path accordingly
results_path = '/home/ubuntu/enigma/results' 
print('\n---------------------------------')

print('Starting pipeline..\n')
for atlas in atlases:
        for chaco_type in chaco_types:
            
            for crossval in crossval_types:
                print('Formatting data..')
                [X, Y, C, lesion_load, site] = create_data_set(CSV_PATH,atlas,covariates, verbose, y_var, chaco_type, subset,1,ll= lesionload)
                
                print('Running machine learning model: \n')
                if crossval == '1':
                    print('1. Outer CV: Random partition fixed fold sizes, Inner CV: Random partition fixed fold sizes')
                    # is random when random_state not specified 
                    outer_cv = KFold(n_splits=5, shuffle=True)
                    inner_cv = KFold(n_splits=5, shuffle=True)

                if crossval == '2':
                    print('2. Outer CV: Leave-one-site-out, Inner CV:  Leave-one-site-out')
                    outer_cv = LeaveOneGroupOut()
                    inner_cv = LeaveOneGroupOut()

                if crossval == '3':
                    print('3. Outer CV: Group K-fold, Inner CV: Group K-fold')
                    outer_cv = GroupKFold(n_splits=5)
                    inner_cv = GroupKFold(n_splits=5)
                    
                if crossval == '4':
                    print('4 Outer CV: Shuffle, Inner CV:  Shuffle')
                    outer_cv = ShuffleSplit(n_splits=5)
                    inner_cv = ShuffleSplit(n_splits=5)
                    
                if crossval == '5':
                    print('5 Outer CV: GroupShuffleSplit, Inner CV:  GroupShuffleSplit')
                    outer_cv = GroupShuffleSplit(n_splits=5)
                    inner_cv = GroupShuffleSplit(n_splits=5)
                
                #run_regression(X, Y, site, inner_cv,outer_cv,models_tested, atlas, y_var, chaco_type, subset,1, results_path,crossval, nperms,null)
                #run_regression_ensemble(X, C, Y, site, inner_cv,outer_cv,models_tested, atlas, y_var, chaco_type, subset,1, results_path,crossval, nperms,null)
                #run_regression_lesionload(lesion_load, Y, site, inner_cv,outer_cv,models_tested, atlas, y_var, chaco_type, subset,1, results_path,crossval, nperms,null)
                #run_regression_lesionload_cstonly(lesion_load, Y, site, inner_cv,outer_cv,models_tested, atlas, y_var, chaco_type, subset,1, results_path,crossval, nperms,null)
                    # models_tested = ['ensemble_reg']
                mdl_label = 'ridge'
                filename = results_path + '/{}_{}_{}_{}_{}_crossval{}'.format(atlas, y_var, chaco_type, subset, mdl_label,crossval)
                for k in range(0, nperms):
                    r2scores=np.load(filename +'_'+ str(k) + '_scores.npy',allow_pickle=True)
                    correlation = np.load(filename +'_'+ str(k) +'_correlations.npy',allow_pickle=True)
                    varimpts=np.load(filename +'_'+ str(k) + '_activation_weights.npy',allow_pickle=True)
                    mdl=np.load(filename +'_'+ str(k) + '_model.npy',allow_pickle=True)
                    alphas=[]
                    feats=[]
                    for a in range(0,5):
                            alphas.append(mdl[0][a]['ridge'].alpha)
                            feats.append(mdl[0][a]['featselect'].k)
                


                

                

