
from imp import reload
import run_regression_models
reload(run_regression_models)
from run_regression_models import run_models


run_analyses = ['1','2','3'] # list of analyses to run (corresponds to analysis folders "analysis_X")
generate_figs_only = True # whether to run ML models or just generate figures


if set(['1']).issubset(set(run_analyses)):  
        kwargs = {'covariates':['AGE', 'SEX', 'DAYS_POST_STROKE'], \
                'results_path':'/home/ubuntu/enigma/results',\
                'output_path': '/analysis_1',\
                'models_tested': ['ridge'],\
                'lesionload_types': [ 'M1', 'all', 'none'], \
                'crossval_types':['1'],\
                'atlases':['fs86subj', 'shen268'],\
                'chaco_types':['chacovol'],\
                'nperms':100, \
                'figs_only':generate_figs_only}
        run_models(**kwargs)

if set(['2']).issubset(set(run_analyses)):  
        kwargs = {'covariates':['AGE', 'SEX', 'DAYS_POST_STROKE'], \
                'results_path':'/home/ubuntu/enigma/results',\
                'output_path': '/analysis_2',\
                'models_tested': ['ridge'],\
                'lesionload_types': ['M1', 'all', 'none'], \
                'crossval_types':['5'],\
                'atlases':['fs86subj', 'shen268'],\
                'nperms':100, \
                'figs_only':generate_figs_only}
        run_models(**kwargs)

if set(['3']).issubset(set(run_analyses)):  
        kwargs = {'covariates':['AGE', 'SEX', 'DAYS_POST_STROKE'], \
                'results_path':'/home/ubuntu/enigma/results',\
                'output_path': '/analysis_3',\
                'models_tested': ['ridge'],\
                'lesionload_types': ['M1', 'all', 'none'], \
                'crossval_types':['4'],\
                'atlases':['fs86subj', 'shen268'],\
                'nperms':100, \
                'figs_only':generate_figs_only}
        run_models(**kwargs)

if set(['4']).issubset(set(run_analyses)):  
        kwargs = {'covariates':['AGE', 'SEX', 'DAYS_POST_STROKE'], \
                'results_path':'/home/ubuntu/enigma/results',\
                'output_path': '/analysis_4',\
                'models_tested': ['ridge'],\
                'lesionload_types': [ 'all', 'none'], \
                'crossval_types':['2'],\
                'atlases':['fs86subj', 'shen268'],\
                'nperms':100, \
                'figs_only':generate_figs_only}

        run_models(**kwargs)

