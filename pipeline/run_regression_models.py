import sys; sys.path
import os
from data_formatting import * 
from matplotlib.pyplot import figure
from helper_functions import *
import data_formatting
import helper_functions 
from imp import reload
import logging
from matplotlib.transforms import Affine2D
from helper_functions_figures import *
import helper_functions_figures
reload(helper_functions)
reload(data_formatting)
reload(helper_functions_figures)


# This function is the main function for running machine learning models with the specified inputs.
# It takes a number of optional parameters, including y_var, subsets, models_tested, verbose, covariates, lesionload_types, nperms, save_models, 
# ensembles, atlases, chaco_types, crossval_types, null, results_path, output_path, figs_only, analysis_id, workbench_vis, scenesdir, wbpath,
# boxplots, and override_rerunmodels. If any of these parameters are not specified, the function uses default values for them. The function then 
# checks if the specified values for models_tested, lesionload_types, and ensembles are valid options, and raises an error if they are not. It then
# loads the necessary data and runs the specified machine learning models, saving the output to the specified directories. The function also has
# options for visualizing the results using the Connectome Workbench software and generating box plots of the results.


def run_models(LESIONMASK_PATH, CSV_PATH, y_var, subsets, model_tested, verbose, covariates, lesionload_types, nperms, save_models, ensembles,hcp_dir, atlases, chaco_types, crossval_types, null, results_path, output_path, figs_only, analysis_id, workbench_vis,scenesdir, wbpath,boxplots, override_rerunmodels, ensemble_atlas):

    model_options= ['ridge', 'lasso', 'elastic_net', 'ridge_nofeatselect', 'linear_regression', 'svm', 'ensemble_reg']
    if not set([model_tested]).issubset(set(model_options)):
        raise RuntimeError('Warning! Unknown model option specified {} \n Only the following options are allowed {} \n'.format(model_tested, model_options))

    lesionload_options = ['M1', 'none', 'all', 'all_2h', 'slnm']
    if not set(lesionload_types).issubset(set(lesionload_options)):
        raise RuntimeError('Warning! Unknown lesion load type specified: {}\n Only the following options are allowed: {} \n'.format(lesionload_types, lesionload_options))

    ensemble_options =['none', 'demog', 'chaco_ll', 'chaco_ll_demog']
    if not set(ensembles).issubset(set(ensemble_options)):
        raise RuntimeError('Warning! Unknown ensemble type specified: {}\n Only the following options are allowed: {} \n'.format(ensembles, ensemble_options))

    atlas_options = ['fs86subj', 'shen268']
    print(atlases)
    print(set(atlases))
    print(set(atlas_options))
    if not set(atlases).issubset(set(atlas_options)):
        raise RuntimeError('Warning! Unknown atlas type specified: {}\n Only the following options are allowed: {} \n'.format(atlases, atlas_options))

    chaco_options = ['chacovol', 'chacoconn']
    if not set(chaco_types).issubset(set(chaco_options)):
        raise RuntimeError('Warning! Unknown atlas type specified: {}\n Only the following options are allowed: {} \n'.format(chaco_types, chaco_options))

    crossval_options = ['1', '2', '3', '4', '5']
    if not set(crossval_types).issubset(set(crossval_options)):
        raise RuntimeError('Warning! Unknown cross validation type specified: {}\n Only the following options are allowed: {} \n'.format(atlases, crossval_options))


    if os.path.exists(os.path.join(results_path, output_path)):
        print('Path {} exists. Potentially overwriting files.'.format(os.path.join(results_path, output_path)))
    else:
        print('Path {} does not exist. Creating it now.'.format(os.path.join(results_path, output_path)))
        os.makedirs(os.path.join(results_path, output_path))

    labels=[]

    r2means=np.empty(shape=(0,nperms))
    corrs=np.empty(shape=(0,nperms))

    for subset in subsets:
        for ensemble in ensembles:
            for lesionload_type in lesionload_types:
                if lesionload_type == 'none' and (('chacovol' in chaco_types) or ('chacoconn' in chaco_types)):
                    
                    print('Running ChaCo models.........')
                    for atlas in atlases: 
                        for chaco_type in chaco_types:
                            for crossval in crossval_types:
                                
                                #create the log file name
                                log_file = os.path.join(results_path + '{}_{}_{}_{}_{}_crossval{}_ensemble-{}'.format(atlas, y_var, chaco_type, subset, model_tested,crossval, ensemble) + '.log')
                                logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')

                                #format the data for the current parameters
                                [X, Y, C, lesion_load, site] = create_data_set(CSV_PATH,LESIONMASK_PATH,atlas,covariates, verbose, y_var, chaco_type, subset,1,ll= lesionload_type)
                                
                                if verbose:
                                    announce_runningmodel(lesionload_type, ensemble, atlas, chaco_type, crossval, override_rerunmodels)
                                
                                if not override_rerunmodels: # if we don't want to override model results
                                    # The code checks if the user wants to override previous model runs. If not, it checks if the model 
                                    # results already exist. If they do, it skips running the model and uses the existing results. If the 
                                    # results don't exist or the user wants to override previous results, the code runs the model and saves the results.

                                    files_exist, folder = check_if_files_exist_already(crossval,model_tested,atlas,chaco_type,results_path, ensemble, y_var, subset,ensemble_atlas)
                                    if files_exist: # we dont want to override but dont recalculate what's already been done
                                        output_fullpath = folder
                                        output_path = output_fullpath.replace(results_path, '').replace('/', '')
                                        print('\n')
                                    else:
                                        if not figs_only: # if figs_only then we just want the output path where the files are located. if not, then actually run the model.
                                            set_up_and_run_model(crossval, model_tested,lesion_load, lesionload_type, X, Y, C, site, atlas, y_var, chaco_type, subset, save_models, results_path, nperms, null,ensemble, output_path,ensemble_atlas)
                                else: # we do want to override previous results
                                    print('Overriding previous model runs!')
                                    set_up_and_run_model(crossval, model_tested,lesion_load, lesionload_type, X, Y, C, site, atlas, y_var, chaco_type, subset, save_models, results_path, nperms, null,ensemble, output_path,ensemble_atlas)

                                n_outer_folds=5
                                
                                [r2all, corrall] = save_model_outputs(results_path, output_path, atlas, y_var, chaco_type, subset, model_tested, crossval, nperms,ensemble,n_outer_folds, ensemble_atlas)
                                r2means=np.append(r2means,np.reshape(np.mean(r2all,axis=1),[-1, nperms]),axis=0)
                                corrs=np.append(corrs,np.reshape(np.mean(corrall,axis=1),[-1, nperms]),axis=0)
                                
                                # Generate labels for figures
                                if ensemble=='demog':
                                    label =atlas + ' ' + chaco_type + ' demog'
                                elif ensemble=='chaco_ll':
                                    label =atlas + ' ' + chaco_type + ' chacoll'
                                elif ensemble=='chaco_ll_demog':
                                    label =atlas + ' ' + chaco_type + ' chacoll+demog'
                                else:
                                    label = atlas + ' ' + chaco_type   
                                if len(crossval_types)>1:
                                    label = label + ' ' + crossval   
                                if len(subsets)>1:
                                    label = label + ' ' + subset
                            
                                labels.append(label)
                                                                
                                # The code generates visualization files for the workbench and creates figures using the workbench.
                                # This is only done if the workbench_vis variable is set to True.
                                if chaco_type == 'chacovol':   
                                    if workbench_vis:
                                        kwargs_workbench_setup = { 'results_path': results_path, 'output_path': output_path, 'analysis_id': analysis_id, \
                                            'y_var':y_var, 'chaco_type':chaco_type, 'subset':subset, 'atlas':atlas,'hcp_dir':hcp_dir,'wbpath':wbpath,  'model_tested':model_tested,  'crossval':crossval, 'scenesdir':scenesdir}
                                        print('\nMaking workbench visualization files..\n')
                                        generate_wb_files(**kwargs_workbench_setup)
                                        kwargs_workbench = { 'results_path': results_path, 'analysis_id': analysis_id, 'scenesdir': scenesdir,\
                                            'y_var':y_var, 'chaco_type':chaco_type, 'subset':subset, 'atlas':atlas,'model_tested':model_tested, 'wbpath':wbpath, 'crossval':crossval}
                                        print('\nMaking workbench visualization files..\n')
                                        print('Making workbench figures..\n')
                                        generate_wb_figures(**kwargs_workbench)

                else: #chaco_ll and chaco_ll_demog run here.
                    
                    for crossval in crossval_types:
                        print('crossval = {}'.format(crossval))

                        atlas, model_tested, chaco_type = set_vars_for_ll(lesionload_type)
                            
                        log_file = os.path.join(results_path + '{}_{}_{}_{}_{}_crossval{}_ensemble-{}'.format(atlas, y_var, chaco_type, subset, model_tested,crossval, ensemble) + '.log')
                        
                        logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
                            
                        [X, Y, C, lesion_load, site] = create_data_set(CSV_PATH,LESIONMASK_PATH,ensemble_atlas,covariates, verbose, y_var, chaco_type,subset,1,ll= lesionload_type)
                        
                        if verbose:
                            announce_runningmodel(lesionload_type, ensemble, atlas, chaco_type, crossval, override_rerunmodels)
                        
                        if not override_rerunmodels: # if we don't want to override model results
                            files_exist, folder = check_if_files_exist_already(crossval,model_tested,atlas,chaco_type,results_path, ensemble, y_var, subset,ensemble_atlas)
                            
                            if files_exist: # we dont want to override but dont recalculate what's already been done
                                print('folder')
                                output_fullpath = folder
                                output_path = output_fullpath.replace(results_path, '')
                            else:
                                if not figs_only: # if figs_only then we just want the output path where the files are located. if not, then actually run the model.
                                    set_up_and_run_model(crossval, model_tested,lesion_load, lesionload_type, X, Y, C, site, atlas, y_var, chaco_type, subset, save_models, results_path, nperms, null,ensemble, output_path,ensemble_atlas)
                        else: # we do want to override previous results
                            print('Overriding previous model runs!')
                            set_up_and_run_model(crossval, model_tested,lesion_load, lesionload_type, X, Y, C, site, atlas, y_var, chaco_type, subset, save_models, results_path, nperms, null,ensemble, output_path,ensemble_atlas)

                        n_outer_folds =5
                                                
                        [r2all, corrall] = save_model_outputs(results_path, output_path, atlas, y_var, chaco_type, subset, model_tested, crossval, nperms,ensemble,n_outer_folds,ensemble_atlas)
                        r2means=np.append(r2means,np.reshape(np.mean(r2all,axis=1),[-1, nperms]),axis=0)
                        corrs=np.append(corrs,np.reshape(np.mean(corrall,axis=1), [-1, nperms]),axis=0)
                        
                        # Generate labels for figures
                        if ensemble == 'chaco_ll':
                            label = atlas + ' chaco_ll' + ensemble_atlas
                        elif ensemble=='demog':
                            label =atlas + ' demog'
                        elif ensemble=='chaco_ll_demog':
                            label =atlas + ' chacoll+demog' + ensemble_atlas
                        else:
                            label = atlas
                        if len(crossval_types)>1:
                            label = label + ' ' + crossval
                        if len(subsets)>1:
                            label = label + ' ' + subset
                            
                        labels.append(label)
                        
                        # The code generates boxplots of the lesion load beta coefficients.
                        if atlas == 'lesionload_all' or atlas=='lesionload_all_2h':
                            kwargs_llfigs = {'results_path':results_path, 'output_path':output_path, 'analysis_id':analysis_id,\
                                'atlas':atlas, 'y_var':y_var, 'chaco_type':chaco_type, 'subset':subset,\
                                    'model_tested':model_tested, 'crossval':crossval}
                            generate_smatt_ll_figures(**kwargs_llfigs)

                            
    subsets = len(ensembles)
    kwargs = {'label': labels, 'r2_scores':r2means, 'correlations':corrs,'results_path': results_path, 'analysis_id': analysis_id, \
            'subsets': subsets}

    print('Making boxplots..')
    create_performance_figures(**kwargs)
        
        
        






