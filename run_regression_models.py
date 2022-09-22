import sys; sys.path
import pandas as pd
import numpy as np 
import seaborn as sns
import scipy.io as sio
import os
from sklearn.model_selection import RepeatedKFold, train_test_split, GridSearchCV, cross_val_score, KFold, StratifiedKFold
from sklearn import preprocessing, linear_model
from sklearn.metrics import explained_variance_score, r2_score, make_scorer
from sklearn.preprocessing import StandardScaler
from joblib import parallel_backend, Parallel, delayed
from helper_functions import *
from data_formatting import *

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import time

## all settings:

atlas='fs86subj'
y_var = 'normed_motor_scores'
chaco_type = 'chacoconn'
subset = 'chronic'
models_tested = ['ridge']
verbose = True
covariates=['SEX', 'AGE', 'CHRONICITY']
nperms=1
CSV_PATH = os.path.join("/home/ubuntu/enigma/Behaviour_Information_ALL_April7_2022.csv")  # Set this path accordingly
RESULTS_PATH = '/home/ubuntu/enigma/results'

[X, Y, C, site] = create_data_set(CSV_PATH,atlas,covariates, verbose, y_var, chaco_type, subset)


analysis_ID ='1'

if analysis_ID == '1':
    print('1. Outer CV: Random partition fixed fold sizes, Inner CV: Random partition fixed fold sizes')
    outer_cv = RepeatedKFold(n_splits=5, n_repeats = nperms, random_state=0)
    inner_cv = KFold(n_splits=5, random_state=0, shuffle=True)



def run_regression(inner_cv, outer_cv, models_tested,save_models, atlas, y_var, chaco_type, subset):
    
    outer_cv_splits = outer_cv.get_n_splits(X, Y)

    models = np.zeros((len(models_tested), outer_cv_splits), dtype=object)
    explained_var  = np.zeros((len(models_tested),outer_cv_splits), dtype=object)
    variable_importance  = []
    correlations  = np.zeros((len(models_tested),outer_cv_splits), dtype=object)


    
    for cv_fold, (train_id, test_id) in enumerate(outer_cv.split(X, Y)):

        print("Fold: {}".format(cv_fold + 1))
        X_train, X_test = X[train_id], X[test_id]
        y_train, y_test = Y[train_id], Y[test_id]

        mdls, mdls_labels = get_models('regression', models_tested)
        
        mdl_idx=0
        for mdl, mdl_label in zip(mdls, mdls_labels): 
            
            filename = RESULTS_PATH + '/{}_{}_{}_{}_{}'.format(atlas, y_var, chaco_type, subset, mdl_label)
            
            print('Performing grid search for: {} \n'.format(mdl_label))
            mdl = inner_loop(mdl, mdl_label, X_train, y_train, inner_cv, 10)   
    
            y_pred = mdl.fit(X_train, y_train).predict(X_test)
            
            expl=explained_variance_score(y_test, y_pred)

            variable_importance.append(mdl.named_steps[mdl_label].coef_)
            correlations[mdl_idx, cv_fold] = np_pearson_cor(y_test,y_pred)

            explained_var[mdl_idx, cv_fold]=expl
            if save_models:
                models[mdl_idx, cv_fold] = mdl
                
            mdl_idx += 1

    print("Saving data...")
    np.save(os.path.join(RESULTS_PATH, filename + "_scores.npy"), explained_var)
    np.save(os.path.join(RESULTS_PATH, filename + "_model.npy"), models)
    np.save(os.path.join(RESULTS_PATH, filename + "_model_labels.npy"), mdls_labels)
    np.save(os.path.join(RESULTS_PATH, filename + "_variable_impts.npy"), variable_importance)



run_regression(inner_cv,outer_cv, models_tested=models_tested, atlas=atlas, y_var=y_var, chaco_type=chaco_type, subset=subset, save_models=1)
