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
# ensembles, atlases, chaco_types, crossval_types, null, results_path, output_folder, figs_only, fig_path, workbench_vis, scenesdir, wbpath,
# boxplots, and override_rerunmodels. If any of these parameters are not specified, the function uses default values for them. The function then 
# checks if the specified values for models_tested, lesionload_types, and ensembles are valid options, and raises an error if they are not. It then
# loads the necessary data and runs the specified machine learning models, saving the output to the specified directories. The function also has
# options for visualizing the results using the Connectome Workbench software and generating box plots of the results.


def run_models(site_colname, csv_path, y_var,nemo_path, yvar_colname,subid_colname,chronicity_colname,subsets,nemo_settings, models_tested, verbose, covariates, lesionload_types, nperms, save_models, ensembles,hcp_dir, atlases, chaco_types, crossval_types, null, results_path, output_folder, figs_only, fig_path, workbench_vis,scenesdir, wbpath,boxplots, override_rerunmodels, ensemble_atlas,final_model,generate_figures):
    
    subsetcounter = 0
    labels=[]
    r2means=np.empty(shape=(0,nperms))
    corrs=np.empty(shape=(0,nperms))

    for ensemble in ensembles:
        if ensemble == 'chaco_ll' or ensemble =='chaco_ll_demog':
            models_tested_chaco = []
            models_tested_ll = models_tested
        else:
            models_tested_ll=['one-item list']
            models_tested_chaco= models_tested

        for lesionload_type in lesionload_types:
            if lesionload_type == 'none' and (('chacovol' in chaco_types) or ('chacoconn' in chaco_types)):
                
                print('\nRunning ChaCo models.........')
                for atlas in atlases: 
                    for chaco_type in chaco_types:
                        for crossval in crossval_types:
                            
                            for model_tested in models_tested_chaco:
                                for subset in subsets:
                                    chaco_model_tested= []
                                    print('----- \n ----------- \n ----------- \n ------')

                                    #format the data for the current parameters
                                    [X, Y, C, lesion_load, subIDs] = create_data_set(csv_path,site_colname,nemo_path,yvar_colname,subid_colname,chronicity_colname,atlas,covariates, verbose, y_var, chaco_type, subset,1,nemo_settings=nemo_settings,ll= lesionload_type)
                                    
                                    subIDs = subIDs.index

                                    print(X.shape)
                                    
                                    if subset == 'acutechronic':
                                        # if acutechronic, X (from above) is chronic data.
                                        # load the acute data here:
                                        [acuteX, acuteY,acuteC, acute_lesion_load, acute_subIDs] = create_data_set(csv_path,site_colname,nemo_path,yvar_colname, subid_colname,chronicity_colname,atlas,covariates, verbose, y_var, chaco_type,'acute',1,nemo_settings,ll= lesionload_type)
                                        acute_subIDs = acute_subIDs.index
                                        acute_data = {'acute_X':acuteX,'acute_Y':acuteY, 'acute_LL':acute_lesion_load, 'acute_C':acuteC, 'acute_subIDs':acute_subIDs}
                                    else:
                                        acute_data = []
                                                                        
                                    if verbose:
                                        announce_runningmodel(lesionload_type, ensemble, atlas, chaco_type, crossval, override_rerunmodels)
                                    
                                    if not override_rerunmodels: # if we don't want to override model results
                                        # The code checks if the user wants to override previous model runs. If not, it checks if the model 
                                        # results already exist. If they do, it skips running the model and uses the existing results. If the 
                                        # results don't exist or the user wants to override previous results, the code runs the model and saves the results.
                                        files_exist, folder = check_if_files_exist(crossval,model_tested,atlas,chaco_type,results_path, ensemble, y_var, subset,ensemble_atlas)
                                        if files_exist: # we dont want to override but dont recalculate what's already been done
                                            output_fullpath = folder
                                            output_folder = output_fullpath.replace(results_path, '').replace('/', '')
                                            print('\n')
                                        else:
                                            print('Files dont exist.\n')
                                            print(figs_only)
                                            if figs_only: 
                                                print('running figs only.')
                                            else:# if figs_only then we just want the output path where the files are located. if not, then actually run the model.
                                                set_up_and_run_model(crossval, model_tested,lesion_load, lesionload_type, X, Y, C, subIDs, atlas, y_var, chaco_type, subset, save_models, results_path, nperms, null,ensemble, output_folder,ensemble_atlas,chaco_model_tested,acute_data,final_model)
                                    else: # we do want to override previous results
                                        set_up_and_run_model(crossval, model_tested,lesion_load, lesionload_type, X, Y, C, subIDs, atlas, y_var, chaco_type, subset, save_models, results_path, nperms, null,ensemble, output_folder,ensemble_atlas,chaco_model_tested,acute_data,final_model)

                                    n_outer_folds=5
      
                                    [r2all, corrall] = save_model_outputs(results_path, output_folder, atlas, y_var, chaco_type, subset, model_tested, crossval, nperms,ensemble,n_outer_folds, ensemble_atlas)
                                    r2means=np.append(r2means,np.reshape(np.mean(r2all,axis=1),[-1, nperms]),axis=0)
                                    corrs=np.append(corrs,np.reshape(np.mean(corrall,axis=1),[-1, nperms]),axis=0)

                                    # The code generates visualization files for the workbench and creates figures using the workbench.
                                    # This is only done if the workbench_vis variable is set to True.
                                    if generate_figures:
                                        
                                        if chaco_type == 'chacovol':   
                                            if workbench_vis:
                                                if atlas =='shen268':
                                                    factor = 1
                                                elif atlas=='fs86subj':
                                                    factor=4
                                                kwargs_workbench_setup = { 'results_path': results_path, 'output_folder': output_folder, 'fig_path': fig_path, \
                                                    'y_var':y_var, 'chaco_type':chaco_type, 'subset':subset, 'atlas':atlas,'hcp_dir':hcp_dir,'wbpath':wbpath,  'model_tested':model_tested,  'crossval':crossval, 'scenesdir':scenesdir, 'final_model':final_model,'factor':factor}
                                                print('\nMaking workbench visualization files..\n')
                                                #generate_wb_files(**kwargs_workbench_setup)
        
                                                kwargs_workbench = { 'results_path': results_path, 'fig_path': fig_path, 'scenesdir': scenesdir,\
                                                    'y_var':y_var, 'chaco_type':chaco_type, 'subset':subset, 'atlas':atlas,'model_tested':model_tested, 'wbpath':wbpath, 'crossval':crossval, 'final_model':final_model}
                                                print('\nMaking workbench visualization files..\n')
                                                print('Making workbench figures..\n')
                                                #generate_wb_figures(**kwargs_workbench)
                                
                                        
                                    # Generate labels for boxplots
                                    if atlas == 'fs86subj':
                                        atlaslabel = 'ChaCo (fs86)'
                                    elif atlas == 'shen268':
                                        atlaslabel = 'ChaCo (shen268)'
                                        
                                    if model_tested =='ridge':
                                        model_tested_label=' + feat. select.'
                                        
                                    elif model_tested =='ridge_nofeatselect':
                                        model_tested_label=''
                                        
                                    if ensemble == 'none':
                                        label = atlaslabel
                                    elif ensemble=='demog':
                                        label =atlaslabel + ' + demog.'

                                    if len(crossval_types)>1:
                                        label = label + ' ' + crossval

                                    print('----- \n ----------- \n ----------- \n ------')
                                    print('printing model tested label for model tested: {}'.format(model_tested))
                                    print(model_tested_label)
                                    label = label +  model_tested_label
                                    print(label)
                                    labels.append(label)
                                    subsetcounter = subsetcounter + 1


            elif not lesionload_type == 'none': #chaco_ll and chaco_ll_demog run here.
                
                for crossval in crossval_types:
                    for chaco_model_tested in models_tested_ll:
                        for subset in subsets:

                            
                            atlas, model_tested, chaco_type = set_vars_for_ll(lesionload_type)

                            [X, Y, C, lesion_load,  subIDs] = create_data_set(csv_path,site_colname,nemo_path,yvar_colname, subid_colname,chronicity_colname,ensemble_atlas,covariates, verbose, y_var, chaco_type,subset,1,nemo_settings,ll= lesionload_type)
                            subIDs = subIDs.index
                            if subset == 'acutechronic':
                                # if acutechronic, X (from above) is chronic data.
                                # load the acute data here:
                                [acuteX, acuteY, acuteC, acute_lesion_load,acute_subIDs] = create_data_set(csv_path,site_colname,nemo_path,yvar_colname, subid_colname,chronicity_colname,ensemble_atlas,covariates, verbose, y_var, chaco_type,'acute',1,nemo_settings,ll= lesionload_type)
                                acute_subIDs=acute_subIDs.index
                                acute_data = {'acute_X':acuteX,'acute_Y':acuteY, 'acute_LL':acute_lesion_load, 'acute_C':acuteC, 'acute_site':acute_subIDs}
                            else:
                                acute_data = []
                                
                            if verbose:
                                announce_runningmodel(lesionload_type, ensemble, atlas, chaco_type, crossval, override_rerunmodels,chaco_model_tested)
                                
                            if not override_rerunmodels: # if we don't want to override model results
                                files_exist, folder = check_if_files_exist(crossval,model_tested,atlas,chaco_type,results_path, ensemble, y_var, subset,ensemble_atlas)
                                
                                if files_exist: # we dont want to override but dont recalculate what's already been done
                                    output_fullpath = folder
                                    output_folder = output_fullpath.replace(results_path, '').replace('/', '')
                                else:
                                    if not figs_only: # if figs_only then we just want the output path where the files are located. if not, then actually run the model.
                                        set_up_and_run_model(crossval, model_tested,lesion_load, lesionload_type, X, Y, C, subIDs, atlas, y_var, chaco_type, subset, save_models, results_path, nperms, null,ensemble, output_folder,ensemble_atlas,chaco_model_tested,acute_data,final_model)
                            else: # we do want to override previous results

                                set_up_and_run_model(crossval, model_tested,lesion_load, lesionload_type, X, Y, C, subIDs, atlas, y_var, chaco_type, subset, save_models, results_path, nperms, null,ensemble, output_folder,ensemble_atlas,chaco_model_tested,acute_data,final_model)

                            n_outer_folds =5
                                                    
                            [r2all, corrall] = save_model_outputs(results_path, output_folder, atlas, y_var, chaco_type, subset, model_tested, crossval, nperms,ensemble,n_outer_folds,ensemble_atlas,chaco_model_tested)
                            r2means=np.append(r2means,np.reshape(np.mean(r2all,axis=1),[-1, nperms]),axis=0)
                            corrs=np.append(corrs,np.reshape(np.mean(corrall,axis=1), [-1, nperms]),axis=0)
                            
                            if generate_figures:
                                # The code generates boxplots of the lesion load beta coefficients.
                                if atlas == 'lesionload_all' or atlas=='lesionload_all_2h':
                                    kwargs_llfigs = {'results_path':results_path, 'output_folder':output_folder, 'fig_path':fig_path,\
                                        'atlas':atlas, 'y_var':y_var, 'chaco_type':chaco_type, 'subset':subset,\
                                            'model_tested':model_tested, 'crossval':crossval}
                                    generate_smatt_ll_figures(**kwargs_llfigs)
                                if atlas == 'lesionload_slnm':
                                    kwargs_llfigs = {'results_path':results_path, 'output_folder':output_folder, 'fig_path':fig_path,\
                                        'atlas':atlas, 'y_var':y_var, 'chaco_type':chaco_type, 'subset':subset,\
                                            'model_tested':model_tested, 'crossval':crossval}
                                    generate_slm_figures(**kwargs_llfigs)

                            # Generate labels for boxplots
                            if atlas == 'lesionload_all':
                                atlaslabel = 'Ipsi. SMATT-LL '
                            elif atlas == 'lesionload_all_2h':
                                atlaslabel = 'L/R SMATT-LL '
                            elif atlas == 'lesionload_m1':
                                atlaslabel = 'M1 CST-LL '
                            elif atlas == 'lesionload_slnm':
                                    atlaslabel = 'sNLM-LL ' 

                            if ensemble == 'chaco_ll':
                                label = atlaslabel + ' + ChaCo ' + '('+ensemble_atlas+')'

                            elif ensemble=='demog':
                                label =atlaslabel + ' + demog.'
                            elif ensemble=='chaco_ll_demog':
                                if chaco_model_tested =='ridge':
                                    add = ' + feat. select.'
                                elif chaco_model_tested =='ridge_nofeatselect':
                                    add= ''
                                else:
                                    add= ''
                                label =atlaslabel + ' + ChaCo ' + '('+ensemble_atlas+')' + add + ' + demog.'
                            else:
                                label = atlaslabel
 

                            if len(crossval_types)>1:
                                label = label + ' ' + crossval
                            #if len(subsets)>1:
                            #    if subset == 'acutechronic':
                            #        subset_label = 'Acute & chronic data: '
                            #    else:
                            #        subset_label = 'Chronic data: '
                            #    label = subset_label + label 
                            labels.append(label)
                            subsetcounter = subsetcounter+1


    if len(ensembles)>1:              
        subsets = len(ensembles)

    if type(subsets)==int:
        print('sfjsa')
        acutechronic=True
    else:
        if len(subsets)==2:
            acutechronic = True
            subsets = 2
        else:
            acutechronic=[]
            subsets = 1
    kwargs = {'label': labels, 'r2_scores':r2means, 'correlations':corrs,'results_path': results_path, 'fig_path': fig_path, \
            'subsets': subsets, 'acutechronic':acutechronic}
    if generate_figures:
        if boxplots:
            print('Making boxplots..')
            create_performance_figures(**kwargs)
        if boxplots:
            print('Making paired plots..')
            create_matrix_figures(**kwargs)








