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


def run_models(y_var=None, subsets=None, models_tested=None, verbose=None, covariates=None, lesionload_types=None, nperms=None, save_models=None, ensembles=None, atlases=None, chaco_types=None, crossval_types=None, null =None, results_path=None, output_path = None, figs_only=None, analysis_id=None, workbench_vis=None,scenesdir= None, wbpath =None,boxplots=None, override_rerunmodels=None, ensemble_atlas = None):
    # y_var: str, default ='normed_motor_scores', dependent variable in regression models
    # subsets: str, default = 'chronic', subset of data to use for analysis
    # models_tested: list, default = ['ridge'], machine learning models to run
    # verbose: bool, default = True, whether to print out verbose output
    # covariates: list, default = [], covariates to include in model
    # lesionload_types: list, default = [], lesion load types to use
    # nperms: int, default = 1, number of permutations to run
    # save_models: bool, default = True, whether to save trained models
    # ensembles: list, default = ['none'], what ensemble to run, "demog", "none"
    # atlases: list, default = ['fs86subj'], which atlas to use
    # chaco_types: list, default = ['chacovol'], regional or pairwise chaco type "chacovol", "chacoconn"
    # crossval_types: list, default = ['1'], which cross-validation scheme to use
    # null: int, default= -1, value to use for null entries in data
    # results_path: str, default = '/ubunut/home/enigma/results/', where to save results
    # output_path: str, default = '/test_1', where to save files (npy, not figures)
    # figs_only: bool, default = False, whether to only save figures, without data
    # analysis_id: str, default = 'test_1', identifier for analysis
    # workbench_vis: bool, default = False, whether to generate visualizations using Workbench
    # scenesdir: str, default = '/test_1/scenes', directory where Workbench scenes are saved
    # wbpath: str, default = '/workbench/bin_linux64/wb_command', path to Workbench command-line interface
    # boxplots: bool, default = True, whether to generate boxplots of results
    # override_rerunmodels: bool, default = False, whether to re-run models even if already run with same parameters
        
    # set defaults
    if y_var:
        y_var = y_var   
    else:
        y_var = 'normed_motor_scores'
    
    if subsets:
        subsets= subsets
    else:
        subsets = ['chronic']
        
    if models_tested:
        model_options= ['ridge', 'lasso', 'elastic_net', 'ridge_nofeatselect', 'linear_regression', 'svm', 'ensemble_reg']
        if not set(models_tested).issubset(set(model_options)):
            raise RuntimeError('Warning! Unknown model option specified {} \n Only the following options are allowed {} \n'.format(models_tested, model_options))
        models_tested = models_tested
    else:
        models_tested = ['ridge']
        
    if verbose:
        verbose = verbose
    else:
        verbose = True
    
    if covariates:
        covariates = covariates
    else:
        covariates = []

    if lesionload_types:
        lesionload_options = ['M1', 'none', 'all', 'all_2h', 'slnm']
        lesionload_types = lesionload_types
        if not set(lesionload_types).issubset(set(lesionload_options)):
            raise RuntimeError('Warning! Unknown lesion load type specified: {}\n Only the following options are allowed: {} \n'.format(lesionload_types, lesionload_options))
    else:
        lesionload_types = []
    
    if nperms:
        nperms = nperms
    else:
        nperms = 1
        
    if save_models:
        save_models = save_models
    else:
        save_models = True
    
    if ensembles:
        ensembles = ensembles
        ensemble_options =['none', 'demog', 'chaco_ll', 'chaco_ll_demog']
        if not set(ensembles).issubset(set(ensemble_options)):
            raise RuntimeError('Warning! Unknown ensemble type specified: {}\n Only the following options are allowed: {} \n'.format(ensembles, ensemble_options))
    else:
        ensembles = ['none']

    if atlases:
        atlases=atlases
        atlas_options = ['fs86subj', 'shen268']
        if not set(atlases).issubset(set(atlas_options)):
            raise RuntimeError('Warning! Unknown atlas type specified: {}\n Only the following options are allowed: {} \n'.format(atlases, atlas_options))
    else:
        atlases = ['fs86subj']
    
    if chaco_types:
        chaco_types=chaco_types
        chaco_options = ['chacovol', 'chacoconn']
        if not set(chaco_types).issubset(set(chaco_options)):
            raise RuntimeError('Warning! Unknown atlas type specified: {}\n Only the following options are allowed: {} \n'.format(chaco_types, chaco_options))
    else:
        chaco_types = ['chacovol']
    
    if crossval_types:
        crossval_types=crossval_types
        crossval_options = ['1', '2', '3', '4', '5']
        if not set(crossval_types).issubset(set(crossval_options)):
            raise RuntimeError('Warning! Unknown atlas type specified: {}\n Only the following options are allowed: {} \n'.format(atlases, crossval_options))
    else:
        crossval_types = ['1']
        
    if null:
        null = null
    else:
        null = -1
    
    if results_path:
        results_path = results_path 
    else: 
        results_path =  '/home/ubuntu/enigma/results' 
    if output_path:
        output_path = output_path
    else:
        output_path = '/test_1'
    
    if figs_only:
        figs_only = figs_only
    else:
        figs_only = False
    if analysis_id:
        analysis_id=analysis_id
    else:
        analysis_id=[]
   
    model_tested = models_tested
    if os.path.exists(results_path+output_path + '/'):
        print('Path {} exists. Potentially overwriting files.'.format(results_path +output_path))
    else:
        print('Path {} does not exist. Creating it now.'.format(results_path + output_path))
        os.makedirs(results_path+output_path)
    
    if workbench_vis:
        workbench_vis = workbench_vis
    else:
        workbench_vis = False
    if scenesdir:
        scenesdir=scenesdir
    if wbpath:
        wbpath = wbpath
    if boxplots:
        boxplots = boxplots
    if override_rerunmodels:
        override_rerunmodels=override_rerunmodels
    if ensemble_atlas:
        ensemble_atlas=ensemble_atlas   

    print(override_rerunmodels)
    print('\n---------------------------------')
    LESIONMASK_PATH = os.path.join("/home/ubuntu/enigma/lesionmasks/")  # Set this path accordingly
    CSV_PATH = os.path.join("/home/ubuntu/enigma/Behaviour_Information_ALL_April7_2022_sorted_CST_12_ll_slnm.csv")  # Set this path accordingly

    label_plot_one=[]

    r2means=np.empty(shape=(0,nperms))
    corrs=np.empty(shape=(0,nperms))

    loo_counter = 0#counter for the nested loop
    for subset in subsets:
        print('Starting pipeline..\n') 
        for ensemble in ensembles:
            print('ensemble = {}'.format(ensemble))
            for lesionload_type in lesionload_types:
                print('lesionload = {}'.format(lesionload_type))
                if lesionload_type == 'none' and (('chacovol' in chaco_types) or ('chacoconn' in chaco_types)):
                    print('Running ChaCo models.........')
                    for atlas in atlases: 
                        for chaco_type in chaco_types:
                            for crossval in crossval_types:
                                
                                model_tested = models_tested
                                
                                #create the log file name
                                log_file = results_path + '/{}_{}_{}_{}_{}_crossval{}_ensemble-{}'.format(atlas, y_var, chaco_type, subset, model_tested[0],crossval, ensemble) + '.log'
                                logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')

                                #format the data for the current parameters
                                [X, Y, C, lesion_load, site] = create_data_set(CSV_PATH,LESIONMASK_PATH,atlas,covariates, verbose, y_var, chaco_type, subset,1,ll= lesionload_type)
                                print('X shape: {}'.format(X.shape))
                                if lesion_load:
                                    print('lesionload shape: {}'.format(lesion_load.shape))
                                
                                #number of sites in the dataset
                                nsites = np.unique(site).shape[0]
                                
                                logprint('Running machine learning model: \n')
                                logprint('lesionload type: {}'.format(lesionload_type))
                                logprint('ensemble type: {}'.format(ensemble))
                                logprint('atlas type: {}'.format(atlas))
                                logprint('chacotype: {}'.format(chaco_type))
                                logprint('crossval type: {}'.format(crossval))
                                print('override rerunmodels: {}'.format(override_rerunmodels))
                                
                                if not override_rerunmodels: # if we don't want to override model results
                                    # The code checks if the user wants to override previous model runs. If not, it checks if the model 
                                    # results already exist. If they do, it skips running the model and uses the existing results. If the 
                                    # results don't exist or the user wants to override previous results, the code runs the model and saves the results.

                                    files_exist, folder = check_if_files_exist_already(crossval,model_tested,atlas,chaco_type,results_path, ensemble, y_var, subset,ensemble_atlas)
                                    
                                    if files_exist: # we dont want to override but dont recalculate what's already been done
                                        output_fullpath = folder
                                        output_path = output_fullpath.replace(results_path, '')
                                    else:
                                        if not figs_only: # if figs_only then we just want the output path where the files are located. if not, then actually run the model.
                                            set_up_and_run_model(crossval, model_tested,lesion_load, lesionload_type, X, Y, C, site, atlas, y_var, chaco_type, subset, save_models, results_path, nperms, null,ensemble, output_path,ensemble_atlas)
                                else: # we do want to override previous results
                                    print('Overriding previous model runs!')
                                    set_up_and_run_model(crossval, model_tested,lesion_load, lesionload_type, X, Y, C, site, atlas, y_var, chaco_type, subset, save_models, results_path, nperms, null,ensemble, output_path,ensemble_atlas)

                            
                                if crossval == '2':
                                    n_outer_folds = np.unique(site).shape[0]
                                if crossval == '5' or crossval == '1' or crossval =='3':
                                    n_outer_folds=5
                                
                                [r2all, corrall] = save_model_outputs(results_path, output_path, atlas, y_var, chaco_type, subset, model_tested, crossval, nperms,ensemble,n_outer_folds, ensemble_atlas)

                                if ensemble=='demog':
                                    label =atlas + ' ' + chaco_type + ' ensemble'
                                elif ensemble=='chaco_ll':
                                    label =atlas + ' ' + chaco_type + ' ensemble'

                                else:
                                    label = atlas + ' ' + chaco_type
                                    
                                if len(crossval_types)>1:
                                    label = label + ' ' + crossval
                                    
                                if len(subsets)>1:
                                    label = label + ' ' + subset
                            
                                label_plot_one.append(label)
                                
                                # The code checks if the cross validation type is '2', which corresponds to leave-one-site-out cross validation. 
                                # If it is, it initializes empty arrays for the r2means and corrs and appends the median values of r2all and corrall
                                # to the arrays. If the cross validation type is not '2', it appends the mean values of r2all and corrall to 
                                # the r2means and corrs arrays. It also creates figures of the distributions of r2 scores and correlations, but 
                                # this is commented out in the code.
                                if crossval=='2': # leave-one-site-out
                                    if loo_counter == 0:
                                        # have to initialize these down here so we have nsites available for the size
                                        r2means_loo=np.empty(shape=(0,nsites))
                                        corrs_loo=np.empty(shape=(0,nsites))
                                    r2means_loo=np.append(r2means_loo, np.reshape(np.median(r2all,axis=0), [1, nsites]),axis=0)
                                    corrs_loo=np.append(corrs_loo,np.reshape(np.median(corrall,axis=0), [1, nsites]),axis=0)
                                    #kwargs = {'label': label_plot_one, 'r2_scores':r2means_loo, 'correlations':corrs_loo,'results_path': results_path}
                                    #create_dist_figures(**kwargs)
                                else:
                                    r2means=np.append(r2means,np.reshape(np.mean(r2all,axis=1),[-1, nperms]),axis=0)
                                    corrs=np.append(corrs,np.reshape(np.mean(corrall,axis=1),[-1, nperms]),axis=0)
                                    
                                    
                                # The code generates visualization files for the workbench and creates figures using the workbench.
                                # This is only done if the workbench_vis variable is set to True.
                                if chaco_type == 'chacovol':   
                                    if workbench_vis:
                                        kwargs_workbench_setup = { 'results_path': results_path, 'output_path': output_path, 'analysis_id': analysis_id, \
                                            'y_var':y_var, 'chaco_type':chaco_type, 'subset':subset, 'atlas':atlas,'model_tested':model_tested,  'crossval':crossval}
                                        print('\nMaking workbench visualization files..\n')
                                        generate_wb_files(**kwargs_workbench_setup)
                                        kwargs_workbench = { 'results_path': results_path, 'analysis_id': analysis_id, 'scenesdir': scenesdir,\
                                            'y_var':y_var, 'chaco_type':chaco_type, 'subset':subset, 'atlas':atlas,'model_tested':model_tested, 'wbpath':wbpath, 'crossval':crossval}
                                        print('\nMaking workbench visualization files..\n')
                                        print('Making workbench figures..\n')
                                        generate_wb_figures(**kwargs_workbench)

                else: #chaco_ll and chaco_ll_demog run here.
                    
                    #print('Running Basic models.........')
                    for crossval in crossval_types:
                        print('crossval = {}'.format(crossval))
                        #print('Formatting data..')

                        if lesionload_type =='M1':
                            atlas = 'lesionload_m1'
                            model_tested = ['linear_regression']
                            chaco_type = 'NA'
                        if lesionload_type =='slnm':
                            atlas = 'lesionload_slnm'
                            model_tested = ['linear_regression']
                            chaco_type = 'NA'

                        elif lesionload_type =='all':
                            atlas = 'lesionload_all'
                            model_tested= ['ridge_nofeatselect']
                            chaco_type = 'NA'
                            
                        elif lesionload_type =='all_2h':
                            atlas = 'lesionload_all_2h'
                            model_tested= ['ridge_nofeatselect']
                            chaco_type = 'NA' 
                            
                        
                        log_file = results_path + '/{}_{}_{}_{}_{}_crossval{}_ensemble-{}'.format(atlas, y_var, chaco_type, subset, model_tested[0],crossval, ensemble) + '.log'
                        
                        logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
                        
                        #logprint('Started')
                        if ensemble_atlas:
                            chaco_atlas = ensemble_atlas
                        else:
                            chaco_atlas = atlas
                            
                        [X, Y, C, lesion_load, site] = create_data_set(CSV_PATH,LESIONMASK_PATH,chaco_atlas,covariates, verbose, y_var, chaco_type,subset,1,ll= lesionload_type)

      
                        logprint('~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ \n ')
                        logprint('~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ Running machine learning model: ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ \n ')
                        logprint('~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ \n ')

                        logprint('lesionload type: {}'.format(lesionload_type))
                        logprint('ensemble type: {}'.format(ensemble))
                        logprint('atlas type: {}'.format(atlas))
                        logprint('chacotype: {}'.format(chaco_type))
                        logprint('crossval type: {}'.format(crossval))

                        nsites = np.unique(site).shape[0]

                        if not override_rerunmodels: # if we don't want to override model results
                            files_exist, folder = check_if_files_exist_already(crossval,model_tested,atlas,chaco_type,results_path, ensemble, y_var, subset,ensemble_atlas)
                            
                            if files_exist: # we dont want to override but dont recalculate what's already been done
                                print('folder')
                                output_fullpath = folder
                                output_path = output_fullpath.replace(results_path, '')
                            else:
                                if not figs_only: # if figs_only then we just want the output path where the files are located. if not, then actually run the model.
                                    set_up_and_run_model(crossval, model_tested,lesion_load, lesionload_type, X, Y, C, site, atlas, y_var, chaco_type, subset, save_models, results_path, nperms, null,ensemble, output_path,chaco_atlas)
                        else: # we do want to override previous results
                            print('Overriding previous model runs!')
                            set_up_and_run_model(crossval, model_tested,lesion_load, lesionload_type, X, Y, C, site, atlas, y_var, chaco_type, subset, save_models, results_path, nperms, null,ensemble, output_path,chaco_atlas)

                        if crossval == '2':
                            n_outer_folds = np.unique(site).shape[0]
                        elif crossval == '5' or crossval == '1' or crossval =='3' or crossval == '4':
                            n_outer_folds =5
                                                
                        [r2all, corrall] = save_model_outputs(results_path, output_path, atlas, y_var, chaco_type, subset, model_tested, crossval, nperms,ensemble,n_outer_folds,chaco_atlas)

                        if ensemble == 'chaco_ll':
                            label = atlas + ' chaco_ll_ensemble'
                            
                        elif ensemble=='demog':
                            label =atlas + ' ensemble'
                        else:
                            label = atlas
                        if len(crossval_types)>1:
                            label = label + ' ' + crossval
                        if len(subsets)>1:
                            label = label + ' ' + subset
                            
                            
                        label_plot_one.append(label)


                        if crossval=='2': # leave-one-site-out
                            if loo_counter == 0:
                                # have to initialize these down here so we have nsites available for the size
                                r2means_loo=np.empty(shape=(0,nsites))
                                corrs_loo=np.empty(shape=(0,nsites))
                            r2means_loo=np.append(r2means_loo, np.reshape(np.median(r2all,axis=0), [-1, nsites]),axis=0)
                            corrs_loo=np.append(corrs_loo,np.reshape(np.median(corrall,axis=0), [-1, nsites]),axis=0)
                            loo_counter = loo_counter+1
                        else:
                            r2means=np.append(r2means,np.reshape(np.mean(r2all,axis=1),[-1, nperms]),axis=0)
                            corrs=np.append(corrs,np.reshape(np.mean(corrall,axis=1), [-1, nperms]),axis=0)
                        
                        if atlas == 'lesionload_all' or atlas=='lesionload_all_2h':
                            kwargs_llfigs = {'results_path':results_path, 'output_path':output_path, 'analysis_id':analysis_id,\
                                'atlas':atlas, 'y_var':y_var, 'chaco_type':chaco_type, 'subset':subset,\
                                    'model_tested':model_tested, 'crossval':crossval}
                            generate_smatt_ll_figures(**kwargs_llfigs)
                            kwargs_llfigs = {'results_path':results_path, 'output_path':output_path, 'analysis_id':analysis_id,\
                                'atlas':atlas, 'y_var':y_var, 'chaco_type':chaco_type, 'subset':subset,\
                                    'model_tested':model_tested, 'crossval':crossval,'scenesdir': scenesdir, 'wbpath':wbpath}
                            #generate_smatt_ll_wb_figs(**kwargs_llfigs)
                            
    if 'r2means_loo' in locals():
        print('leave one out')
        r2means = r2means_loo
        corrs = corrs_loo

    if 'chaco_ll' in ensembles:
        subsets = ['one', 'two']
    if 'demog' in ensembles and (len(ensembles) >1):
        subsets = ['one', 'two']
    kwargs = {'label': label_plot_one, 'r2_scores':r2means, 'correlations':corrs,'results_path': results_path, 'analysis_id': analysis_id, \
            'subsets': subsets}
    
    print(subsets)
    print(label_plot_one)
    print('boxplots == {}'.format(boxplots))
    if boxplots:
        # create figures that summarize models' performance
        if crossval=='2':
            # if leave one out, we are taking the mean across the permutations where each 'point' is a site
            create_performance_figures_loo(**kwargs)
        else:
            # if we are not looking at leave one out, we are plotting each permutation separately.
            print('Making boxplots..')
            create_performance_figures(**kwargs)
        
        
        






