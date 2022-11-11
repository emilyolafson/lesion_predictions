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
covariates=['SEX', 'AGE', 'DAYS_POST_STROKE', 'LESIONED_HEMISPHERE']

lesionload_type = 'M1' # options: 
nperms=2

save_models=1
lesionload_types = ['none']
ensembles = ['none']
atlases = ['fs86subj', 'shen268']
chaco_types = ['chacoconn']
crossval_types =['1', '5']
null=-1
CSV_PATH = os.path.join("/home/ubuntu/enigma/Behaviour_Information_ALL_April7_2022_sorted_CSTll.csv")  # Set this path accordingly
results_path = '/home/ubuntu/enigma/results' 
print('\n---------------------------------')


print('Starting pipeline..\n') 
for lesionload_type in lesionload_types:
    for ensemble in ensembles:
        if lesionload_type == 'none' or chaco_types == 'chacoconn' or chaco_types == 'chacovol':
            print('Running ChaCo models.........')
            for atlas in atlases: 
                for chaco_type in chaco_types:
                    for crossval in crossval_types:
                        print('Formatting data..')
                        [X, Y, C, lesion_load, site] = create_data_set(CSV_PATH,atlas,covariates, verbose, y_var, chaco_type, subset,1,ll= lesionload_type)
                        
                        print('Running machine learning model: \n')
                        print('lesionload type: {}'.format(lesionload_type))
                        print('ensemble type: {}'.format(ensemble))
                        print('atlas type: {}'.format(atlas))
                        print('chacotype: {}'.format(chaco_type))
                        print('crossval type: {}'.format(crossval))

                        set_up_and_run_model(crossval, model_tested,lesion_load, lesionload_type, X, Y, C, site, atlas, y_var, chaco_type, subset, save_models, results_path, nperms, null,ensemble)
                        #run_regression_lesionload(lesion_load, Y, site, inner_cv,outer_cv,models_tested, atlas, y_var, chaco_type, subset,1, results_path,crossval, nperms,null)
                        #run_regression_lesionload_cstonly(lesion_load, Y, site, inner_cv,outer_cv,models_tested, atlas, y_var, chaco_type, subset,1, results_path,crossval, nperms,null)
                            # models_tested = ['ensemble_reg']
                        save_model_outputs(results_path, atlas, y_var, chaco_type, subset, model_tested, crossval, nperms,lesionload_type,ensemble)
        else:
            print('Running Basic models.........')
            for crossval in crossval_types:
                print('Formatting data..')
                chaco_type=[]
                atlas=[]
  
                [X, Y, C, lesion_load, site] = create_data_set(CSV_PATH,atlas,covariates, verbose, y_var, chaco_type, subset,1,ll= lesionload_type)
                
                print('Running machine learning model: \n')
                print('lesionload type: {}'.format(lesionload_type))
                print('ensemble type: {}'.format(ensemble))
                print('atlas type: {}'.format(atlas))
                print('chacotype: {}'.format(chaco_type))
                print('crossval type: {}'.format(crossval))

                set_up_and_run_model(crossval, model_tested,lesion_load, lesionload_type, X, Y, C, site, atlas, y_var, chaco_type, subset, save_models, results_path, nperms, null,ensemble)
                save_model_outputs(results_path, atlas, y_var, chaco_type, subset, model_tested, crossval, nperms,lesionload_type,ensemble)


                


                    

                

