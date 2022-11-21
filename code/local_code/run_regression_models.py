import sys; sys.path
import pandas as pd
import numpy as np 
import seaborn as sns
import scipy.io as sio
import os
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
model_tested = ['ridge']
verbose = True
covariates=['SEX', 'AGE']

lesionload_type = 'M1' # options: 
nperms=2

save_models=1
ensemble = 'demog'
atlases = ['fs86subj']
chaco_types = ['chacoconn']
crossval_types =[ '5']
null=-1
CSV_PATH = os.path.join("/home/ubuntu/enigma/Behaviour_Information_ALL_April7_2022_sorted_CSTll.csv")  # Set this path accordingly
results_path = '/home/ubuntu/enigma/results' 
print('\n---------------------------------')

print('Starting pipeline..\n')
for atlas in atlases:
        for chaco_type in chaco_types:
            
            for crossval in crossval_types:
                print('Formatting data..')
                [X, Y, C, lesion_load, site] = create_data_set(CSV_PATH,atlas,covariates, verbose, y_var, chaco_type, subset,1,ll= lesionload_type)
                
                print('Running machine learning model: \n')
                
                set_up_and_run_model(crossval, model_tested,lesion_load, lesionload_type, X, Y, C, site, atlas, y_var, chaco_type, subset, save_models, results_path, nperms, null,ensemble)
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
            


                

                

