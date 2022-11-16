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
reload(helper_functions)
reload(data_formatting)


def run_models(y_var=None, subset=None, models_tested=None, verbose=None, covariates=None, lesionload_types=None, nperms=None, save_models=None, ensembles=None, atlases=None, chaco_types=None, crossval_types=None, null =None):
    """
    Runs regression models with specified inputs. Saves figures to specified directories.
    :param y_var: str, default ='normed_motor_scores'
    :param subset: str, default = 'chronic
    :param models_tested: list, default = ['ridge']
    :param verbose: bool, default = True
    :param covariates: list, default = []
    :param lesionload_type: list, default = []
    :param perms: int, default = 1
    :param save_models: bool, default = True
    :param ensembles: list, default = ['none']
    :param atlases: list, default = ['fs86subj']
    :param chaco_types: list, default = ['chacovol']
    :param crossval_types: list, default = ['1']
    :param null: int, default= -1
    :param results_path: str, default = '/ubunut/home/enigma/results/"""
    if y_var:
        y_var = y_var   
    else:
        y_var = 'normed_motor_scores'
        
    if subset:
        subset= subset
    else:
        subset = 'chronic'
        
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
        lesionload_options = ['M1', 'none', 'all']
        print(lesionload_types)
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
        ensemble_options =['none', 'demog']
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
        

print('\n---------------------------------')
LESIONMASK_PATH = os.path.join("/home/ubuntu/enigma/lesionmasks/")  # Set this path accordingly
CSV_PATH = os.path.join("/home/ubuntu/enigma/Behaviour_Information_ALL_April7_2022_sorted_CSTll.csv")  # Set this path accordingly

label_plot_one=[]
label_plot=[]
r2means=np.empty(shape=(0,5))
corrs=np.empty(shape=(0,5))
r2means_loo=np.empty(shape=(0,22))
corrs_loo=np.empty(shape=(0,22))

print('Starting pipeline..\n') 
for lesionload_type in lesionload_types:
    print('lesionload = {}'.format(lesionload_type))
    for ensemble in ensembles:
        print('ensemble = {}'.format(ensemble))
        if lesionload_type == 'none' and (('chacovol' in chaco_types) or ('chacoconn' in chaco_types)):
            print('Running ChaCo models.........')
            for atlas in atlases: 
                for chaco_type in chaco_types:
                    for crossval in crossval_types:
                        print('Formatting data..')
                
                        log_file = results_path + '/{}_{}_{}_{}_{}_crossval{}_ensemble-{}'.format(atlas, y_var, chaco_type, subset, model_tested[0],crossval, ensemble) + '.log'
                        logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')

                        [X, Y, C, lesion_load, site] = create_data_set(CSV_PATH,LESIONMASK_PATH,atlas,covariates, verbose, y_var, chaco_type, subset,1,ll= lesionload_type)
                        
                        nsites = np.count_nonzero(np.unique(site))
                        
                        logprint('Running machine learning model: \n')
                        logprint('lesionload type: {}'.format(lesionload_type))
                        logprint('ensemble type: {}'.format(ensemble))
                        logprint('atlas type: {}'.format(atlas))
                        logprint('chacotype: {}'.format(chaco_type))
                        logprint('crossval type: {}'.format(crossval))

                        #set_up_and_run_model(crossval, model_tested,lesion_load, lesionload_type, X, Y, C, site, atlas, y_var, chaco_type, subset, save_models, results_path, nperms, null,ensemble)
                        if crossval == '2':
                            shape_2 = np.unique(site).shape[0]
                        if crossval == '5' or crossval == '1' or crossval =='3':
                            shape_2==5
                            
                        [r2all, corrall] = save_model_outputs(results_path, atlas, y_var, chaco_type, subset, model_tested, crossval, nperms,lesionload_type,ensemble,shape_2)

                        label_plot_one.append(model_tested[0]+ '_crossval_' + crossval + '_' + atlas +  '_ensemble_' + ensemble + '_chaco_' + chaco_type)
                        
                        if crossval=='2': # leave-one-site-out
                            r2means_loo=np.append(r2means_loo, np.reshape(np.median(r2all,axis=0), [1, nsites]),axis=0)
                            corrs_loo=np.append(corrs_loo,np.reshape(np.median(corrall,axis=0), [1, nsites]),axis=0)
                            #kwargs = {'label': label_plot_one, 'r2_scores':r2means_loo, 'correlations':corrs_loo,'results_path': results_path}
                            #create_dist_figures(**kwargs)
                        else:
                            r2means=np.append(r2means,np.reshape(np.mean(r2all,axis=1),[5]),axis=0)
                            corrs=np.append(corrs,np.reshape(np.mean(corrall,axis=1),[-1, 5]),axis=0)
        else:
            #print('Running Basic models.........')
            for crossval in crossval_types:
                print('crossval = {}'.format(crossval))
                #print('Formatting data..')
                    
                if lesionload_type =='M1':
                    atlas = 'lesionload_m1'
                    model_tested = ['linear_regression']
                    chaco_type = 'NA'

                elif lesionload_type =='all':
                    atlas = 'lesionload_all'
                    model_tested= ['ridge_nofeatselect']
                    chaco_type = 'NA'
                
                log_file = results_path + '/{}_{}_{}_{}_{}_crossval{}_ensemble-{}'.format(atlas, y_var, chaco_type, subset, model_tested[0],crossval, ensemble) + '.log'
                
                logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
                
                #logprint('Started')
                
                [X, Y, C, lesion_load, site] = create_data_set(CSV_PATH,LESIONMASK_PATH,atlas,covariates, verbose, y_var, chaco_type, subset,1,ll= lesionload_type)
                #logprint('~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ \n ')
                #logprint('~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ Running machine learning model: ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ \n ')
                #logprint('~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ \n ')

                #logprint('lesionload type: {}'.format(lesionload_type))
                #logprint('ensemble type: {}'.format(ensemble))
                #logprint('atlas type: {}'.format(atlas))
                ##logprint('chacotype: {}'.format(chaco_type))
                #logprint('crossval type: {}'.format(crossval))
                
                
                #set_up_and_run_model(crossval, model_tested,lesion_load, lesionload_type, X, Y, C, site, atlas, y_var, chaco_type, subset, save_models, results_path, nperms, null,ensemble)
                if crossval == '2':
                    shape_2 = np.unique(site).shape[0]
                if crossval == '5' or crossval == '1' or crossval =='3':
                    shape_2==5
                    
                [r2all, corrall] = save_model_outputs(results_path, atlas, y_var, chaco_type, subset, model_tested, crossval, nperms,lesionload_type,ensemble,shape_2)
               
                label_plot_one.append(model_tested[0]+ '_crossval_' + crossval + '_' + atlas +  '_ensemble_' + ensemble + '_chaco_' + chaco_type)
                                
                if crossval=='2': # leave-one-site-out
                    r2means_loo=np.append(r2means_loo, np.reshape(np.median(r2all,axis=0), [1, nsites]),axis=0)
                    corrs_loo=np.append(corrs_loo,np.reshape(np.median(corrall,axis=0), [1, nsites]),axis=0)
                    #kwargs = {'label': label_plot_one, 'r2_scores':r2means_loo, 'correlations':corrs_loo,'results_path': results_path}
                    #create_dist_figures(**kwargs)
                else:
                    r2means=np.append(r2means,np.reshape(np.mean(r2all,axis=1),[5]),axis=0)
                    corrs=np.append(corrs,np.reshape(np.mean(corrall,axis=1),[-1, 5]),axis=0)
                    
    if  r2means_loo.shape[0] != 0:
        print('loo')

        r2means = r2means_loo
        corrs = corrs_loo


    kwargs = {'label': label_plot_one, 'r2_scores':r2means, 'correlations':corrs,'results_path': results_path, 'filename':fname}
    create_performance_figures_loo(**kwargs)

    site_specific_plots=[]
    if site_specific_plots:
        for site in range(0,shape_2): # for each site, create a plot.
            predarrays=[]
            truearrays=[]
            labels=[]
            for lesionload_type in lesionload_types:
                for ensemble in ensembles:
                    for chaco_type in chaco_types:
                        if crossval == '2': 
                            if lesionload_type == 'none' and (('chacovol' in chaco_types) or ('chacoconn' in chaco_types)):
                                for atlas in atlases:
                                    mdl_label = 'ridge'
                                    rootname_truepred = results_path + '/figures/true_pred/{}_{}_{}_{}_{}_crossval{}'.format(atlas, y_var, chaco_type, subset, mdl_label,crossval)
                                    truearray = np.loadtxt(rootname_truepred+ '_outerfold'+str(site) +'_true_allperms.txt')
                                    predarray = np.loadtxt(rootname_truepred+ '_outerfold'+str(site) +'_pred_allperms.txt')
                                    predarrays.append(predarray)
                                    truearrays.append(truearray)
                                labels.append(lesionload_type)
                            else: # basic model
                                if lesionload_type =='M1':
                                    atlas = 'lesionload_m1'
                                    mdl_label = 'linear_regression'
                                    chaco_type ='NA'
                                elif lesionload_type =='all':
                                    atlas = 'lesionload_all'
                                    mdl_label= 'ridge_nofeatselect'
                                    chaco_type ='NA'
                                rootname_truepred = results_path + '/figures/true_pred/{}_{}_{}_{}_{}_crossval{}'.format(atlas, y_var, chaco_type, subset, mdl_label,crossval)
                                
                                truearray = np.loadtxt(rootname_truepred+ '_outerfold'+str(site) +'_true_allperms.txt')
                                predarray = np.loadtxt(rootname_truepred+ '_outerfold'+str(site) +'_pred_allperms.txt')
                                predarrays.append(predarray)
                                truearrays.append(truearray)
                            labels.append(lesionload_type)

            fig, ax = plt.subplots(figsize =(10, 10))

            if site == 9:
                continue
            listlabels = ['ChaCovol - fs86subj', 'ChaCovol - Shen268',   'M1-CST', 'all-CST']
            trans1 = Affine2D().translate(-0.005, 0.0) + ax.transData
            trans2 = Affine2D().translate(+0.005, 0.0) + ax.transData
            trans3 = Affine2D().translate(+0.001, 0.0) + ax.transData
            trans4 = Affine2D().translate(+0.001, 0.0) + ax.transData

            transforms = [trans1, trans2, trans3, trans4]
            font = {'family' : 'normal','size'   : 10}
            print(len(predarrays))
            
            for i in range(0, len(predarrays)):
                site_size = predarrays[i].shape[0]
                site_size = predarrays[i].shape[0]
                print(predarrays[i].shape)
                
                meanpred = np.mean(predarrays[i],axis=1)
                meantrue = np.mean(truearrays[i],axis=1)
                meanpred_rep = np.repeat(np.reshape(meanpred, [site_size, 1]), 8, axis=1)
                if i==0:
                    color = 'tab:blue'
                elif i==1:
                    color = 'tab:orange'
                elif i==2:
                    color ='tab:green'
                elif i==3:
                    color ='tab:cyan'
                err = np.mean(np.abs(predarrays[i] - meanpred_rep), axis=1) # MAE
                ax.errorbar(meantrue, meanpred, yerr = err,marker='o', color=color, linestyle="none", transform=transforms[i],ms=6,label='$test$').set_label(listlabels[i])
            
            ax.plot([0,1],[0,1],'k--')
            plt.ylim([0, 1])
            plt.xlim([0, 1])
            plt.xlabel('True normed motor scores')
            plt.ylabel('Predicted normed motor scores')
            plt.title('Site {}'.format(site))
            plt.legend()
            plt.savefig(results_path + '/figures/' + 'scatter_ChaCovol_vs_BaseModels_Site' + str(site)+'.png')


        logprint('Finished')

        logprint('-------------- saving logfile w name: {} -------------'.format(log_file))

            


                            

                    

