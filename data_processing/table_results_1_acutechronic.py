# make table with all results

import helper_functions 
from helper_functions import *
from imp import reload
import pandas as pd
reload(helper_functions)
from matplotlib.colors import Normalize
def round_with_padding(value, round_digits):
    return format(round(value,round_digits), "."+str(round_digits)+"f")
r2scores = []
correlations = []
labels = []

nperms = 100
n_outer_folds = 5

crossval = '1'
models_tested = [ 'ridge_nofeatselect', 'ridge']
chaco_types = ['chacovol']
results_path = '/home/ubuntu/enigma/results'
ensembles = ['none','demog', 'chaco_ll','chaco_ll_demog']
y_var = 'normed_motor_scores'
subset = 'acutechronic'
ensemble_atlases = ['fs86subj','shen268']
atlases=['fs86subj', 'shen268']
lesionload_types = ['M1', 'all', 'all_2h', 'slnm', 'none']
output_folder = '/home/ubuntu/enigma/results/analysis_1'

ensemble_labels=[]
r2scores=[]
corrs=[]
counter=0
for ensemble in ensembles:
    if ensemble == 'chaco_ll' or ensemble =='chaco_ll_demog':
        models_tested_chaco = []
        models_tested_ll = models_tested
    else:
        models_tested_ll=['run one']
        models_tested_chaco= models_tested

   

    for chaco_model_tested in models_tested_ll:
        for lesionload_type in lesionload_types:

            if not lesionload_type == 'none': #chaco_ll and chaco_ll_demog run here.
                if (ensemble == 'none') or (ensemble == 'demog'):
                    ensemble_atlases_tmp = ['fs86subj']
                else:
                    ensemble_atlases_tmp = ensemble_atlases
                    
                for ensemble_atlas in ensemble_atlases_tmp:
                    counter = counter +1
                    atlas, model_tested, chaco_type = set_vars_for_ll(lesionload_type)
                    
                    [r2all, corrall] = save_model_outputs(results_path, output_folder, atlas, y_var, chaco_type, subset, model_tested, crossval, nperms,ensemble,n_outer_folds,ensemble_atlas,chaco_model_tested)
                    r2scores.append('{:.3f}'.format(np.median(np.reshape(np.mean(r2all,axis=1),[-1, nperms]))))
                    correlations.append('{:.3f}'.format(np.median(np.reshape(np.mean(corrall,axis=1), [-1, nperms]))))

                    # Generate labels for boxplots
                    if atlas == 'lesionload_all':
                        atlaslabel = 'Ipsi. SMATT-LL'
                    elif atlas == 'lesionload_all_2h':
                        atlaslabel = 'L/R SMATT-LL'
                    elif atlas == 'lesionload_m1':
                        atlaslabel = 'M1 CST-LL'
                    elif atlas == 'lesionload_slnm':
                        atlaslabel = 'sLNM-LL'
                    if ensemble == 'chaco_ll':
                        label = atlaslabel + ' + ChaCo ' + '('+ensemble_atlas+')'
                    elif ensemble=='demog':
                        label =atlaslabel 
                    elif ensemble=='chaco_ll_demog':
                        label =atlaslabel + ' + ChaCo ' + '('+ensemble_atlas+')'
                    else:
                        label = atlaslabel
                    if chaco_model_tested =='ridge':
                        model_tested_label=' (feat. select.)'
                    elif chaco_model_tested =='ridge_nofeatselect':
                        model_tested_label=''
                    else:
                        model_tested_label=''

                    label = label +  model_tested_label     
                    if ensemble=='demog':
                        label =label + ' + demog.' 
                    elif ensemble=='chaco_ll_demog':
                        label =label + ' + demog.' 
                    else:
                        label=label
                    labels.append(label)
                    ensemble_labels.append(ensemble.replace('_', ' '))
                        
    for model_tested in models_tested_chaco:

        for lesionload_type in lesionload_types:
            if lesionload_type == 'none' and (('chacovol' in chaco_types) or ('chacoconn' in chaco_types)):
                
                print('\nRunning ChaCo models.........')
                for atlas in atlases: 
                    for chaco_type in chaco_types:

                            ensemble_atlas=atlas
                            files_exist, folder = check_if_files_exist(crossval,model_tested,atlas,chaco_type,results_path, ensemble, y_var, subset,ensemble_atlas)
                            if files_exist: # we dont want to override but dont recalculate what's already been done
                                output_fullpath = folder
                                output_folder = output_fullpath.replace(results_path, '').replace('/', '')

                            n_outer_folds=5
    
                            [r2all, corrall] = save_model_outputs(results_path, output_folder, atlas, y_var, chaco_type, subset, model_tested, crossval, nperms,ensemble,n_outer_folds, ensemble_atlas)
                            r2scores.append('{:.3f}'.format(np.median(np.reshape(np.mean(r2all,axis=1),[-1, nperms]))))
                            correlations.append('{:.3f}'.format(np.median(np.reshape(np.mean(corrall,axis=1), [-1, nperms]))))
                            
                            
                            # Generate labels for boxplots
                            if atlas == 'fs86subj':
                                atlaslabel = 'ChaCo (fs86)'
                            elif atlas == 'shen268':
                                atlaslabel = 'ChaCo (shen268)'
                                
                            if model_tested =='ridge':
                                model_tested_label=' (feat. select.)'
                                
                            elif model_tested =='ridge_nofeatselect':
                                model_tested_label=''

                            print('----- \n ----------- \n ----------- \n ------')
                            print('printing model tested label for model tested: {}'.format(model_tested))
                            print(model_tested_label)
                            label = atlaslabel +  model_tested_label
                            if ensemble == 'chaco_ll_demog':
                                label = label + ' + demog'
                            elif ensemble=='demog':
                                label =label + ' + demog'
                            else:
                                label=label

                            labels.append(label)
                            ensemble_labels.append(ensemble.replace('_', ' '))

data_table = pd.DataFrame({'Corr. (Std. dev.)': correlations,'$R^2$ (Std. dev.)': r2scores})
#data_table.index = ensemble_labels
print(data_table)

#ensemb = np.array(ensemble_labels, dtype=str)
data_table.index  = pd.MultiIndex.from_arrays([ensemble_labels,labels])
    
def colorize_corr(v, props=''):
    if float(v) < 0.4 :
        return 'color:--rwrapBlack;'
    elif float(v) < 0.42:
        return 'color:--rwrapBlack;'
    elif float(v) < 0.44:
        return 'color:--rwrapBlue;'
    elif float(v) < 0.46:
        return 'color:--rwrapBlue;'
    elif float(v) < 0.49:
        return 'color:--rwrapNavyBlue;'
    elif float(v) < 0.5:
        return 'color:--rwrapProcessBlue;'
    
def colorize_r2(v, props=''):
    if float(v) < 0.2 :
        return 'color:--rwrapBlack;'
    elif float(v) < 0.22:
        return 'color:--rwrapBlue;'

    elif float(v) < 0.23:
        return 'color:--rwrapNavyBlue;'
    elif float(v) < 0.24:
        return 'color:--rwrapProcessBlue;'

print(data_table.style)
s = data_table.style.applymap(colorize_corr, subset='Corr. (Std. dev.)')\
                    .applymap(colorize_r2, subset='$R^2$ (Std. dev.)')


print(s.to_latex(
    column_format="rrrrr", position="h", position_float="centering",
    hrules=True, label="table:5", caption="Styled LaTeX Table",
    multirow_align="t",multicol_align="c"
)  )

print(counter)