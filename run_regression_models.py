import sys; sys.path
import os
from data_formatting import * 
from matplotlib.pyplot import figure
from helper_functions import *
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
nperms=5

save_models=1
lesionload_types = ['M1']
ensembles = ['none']
atlases = ['fs86subj']
chaco_types = ['chacovol']
crossval_types =['1']
null=-1
CSV_PATH = os.path.join("/home/ubuntu/enigma/Behaviour_Information_ALL_April7_2022_sorted_CSTll.csv")  # Set this path accordingly
results_path = '/home/ubuntu/enigma/results' 
print('\n---------------------------------')

import logging

print('Starting pipeline..\n') 
for lesionload_type in lesionload_types:
    for ensemble in ensembles:
        if lesionload_type == 'none' or chaco_types == 'chacoconn' or chaco_types == 'chacovol':
            print('Running ChaCo models.........')
            for atlas in atlases: 
                for chaco_type in chaco_types:
                    for crossval in crossval_types:
                        print('Formatting data..')
                
                        log_file = results_path + '/{}_{}_{}_{}_{}_crossval{}'.format(atlas, y_var, chaco_type, subset, model_tested[0],crossval) + '.log'
                        logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')

                        [X, Y, C, lesion_load, site] = create_data_set(CSV_PATH,atlas,covariates, verbose, y_var, chaco_type, subset,1,ll= lesionload_type)
                        
                        logprint('Running machine learning model: \n')
                        logprint('lesionload type: {}'.format(lesionload_type))
                        logprint('ensemble type: {}'.format(ensemble))
                        logprint('atlas type: {}'.format(atlas))
                        logprint('chacotype: {}'.format(chaco_type))
                        logprint('crossval type: {}'.format(crossval))

                        set_up_and_run_model(crossval, model_tested,lesion_load, lesionload_type, X, Y, C, site, atlas, y_var, chaco_type, subset, save_models, results_path, nperms, null,ensemble)
                        #run_regression_lesionload(lesion_load, Y, site, inner_cv,outer_cv,models_tested, atlas, y_var, chaco_type, subset,1, results_path,crossval, nperms,null)
                        #run_regression_lesionload_cstonly(lesion_load, Y, site, inner_cv,outer_cv,models_tested, atlas, y_var, chaco_type, subset,1, results_path,crossval, nperms,null)
                            # models_tested = ['ensemble_reg']
                        save_model_outputs(results_path, atlas, y_var, chaco_type, subset, model_tested, crossval, nperms,lesionload_type,ensemble)

        else:
            print('Running Basic models.........')
            for crossval in crossval_types:
                print('Formatting data..')
                
                chaco_type='NA'
                atlas=lesionload_type
                
                log_file = results_path + '/{}_{}_{}_{}_{}_crossval{}'.format(atlas, y_var, chaco_type, subset, model_tested[0],crossval) + '.log'
                
                logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
                
                logprint('Started')
                
                #[X, Y, C, lesion_load, site] = create_data_set(CSV_PATH,atlas,covariates, verbose, y_var, chaco_type, subset,1,ll= lesionload_type)
                logprint('~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ \n ')
                logprint('~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ Running machine learning model: ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ \n ')
                logprint('~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ \n ')

                logprint('lesionload type: {}'.format(lesionload_type))
                logprint('ensemble type: {}'.format(ensemble))
                logprint('atlas type: {}'.format(atlas))
                logprint('chacotype: {}'.format(chaco_type))
                logprint('crossval type: {}'.format(crossval))
                
                #set_up_and_run_model(crossval, model_tested,lesion_load, lesionload_type, X, Y, C, site, atlas, y_var, chaco_type, subset, save_models, results_path, nperms, null,ensemble)
                [r2all, corrall] = save_model_outputs(results_path, atlas, y_var, chaco_type, subset, model_tested, crossval, nperms,lesionload_type,ensemble)
                label_plot = model_tested[0] + '_' + atlas +  '_ensemble_' + ensemble + '_chaco_' + chaco_type
                r2mean = np.reshape(np.mean(r2all,axis=1), [1, 5])
                r2input = np.concatenate((r2mean, r2mean),axis=0)
                label_plot = [label_plot, label_plot]
            
                kwargs = {'label': label_plot, 'r2all':r2input, 'corrall':corrall,'results_path': results_path, 'atlas':atlas}
                create_performance_figures(**kwargs)
                
            logprint('Finished')

            logprint('-------------- saving logfile w name: {} -------------'.format(log_file))

            


                    

                

