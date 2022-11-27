
from imp import reload
import run_regression_models
reload(run_regression_models)
from run_regression_models import run_models
from helper_functions_figures import *

run_analyses = [] # list of analyses to run (corresponds to analysis folders "analysis_X")
generate_figs_only = False # whether to run ML models or just generate figures
workbench_vis = False # 

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
                'figs_only':generate_figs_only,\
                'analysis_id':'analysis_1'}
        run_models(**kwargs)
        
scenesdir = '/home/ubuntu/enigma/motor_predictions/wb_files/'
hcpdir ='/home/ubuntu/enigma/motor_predictions/wb_files/HCP_S1200_GroupAvg_v1/'
wbpath = '/home/ubuntu/enigma/motor_predictions/wb_files/workbench_ubuntu/bin_linux64'
run_analyses = [] # list of analyses to run (corresponds to analysis folders "analysis_X")
generate_figs_only = False # whether to run ML models or just generate figures
workbench_vis = True # 
boxplots = True
generate_wb_figures_setup(hcpdir, scenesdir)
if set(['2']).issubset(set(run_analyses)):  
        kwargs = {'covariates':['AGE', 'SEX', 'DAYS_POST_STROKE'], \
                'results_path':'/home/ubuntu/enigma/results',\
                'output_path': '/analysis_2',\
                'models_tested': ['ridge'],\
                'lesionload_types': ['M1', 'all', 'none'], \
                'crossval_types':['5'],\
                'ensembles':['none'],\
                'atlases':['fs86subj', 'shen268'],\
                'nperms':100, \
                'figs_only':generate_figs_only,\
                'analysis_id':'analysis_2'}
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
                'figs_only':generate_figs_only,\
                'analysis_id':'analysis_3'}
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
                'figs_only':generate_figs_only,\
                'analysis_id':'analysis_4'}
        run_models(**kwargs)

if set(['5']).issubset(set(run_analyses)):  
        kwargs = {'covariates':['AGE', 'SEX', 'DAYS_POST_STROKE'], \
                'results_path':'/home/ubuntu/enigma/results',\
                'output_path': '/analysis_5',\
                'models_tested': ['ridge'],
                'lesionload_types': ['M1', 'all'], \
                'crossval_types':['1'],\
                'atlases':['fs86subj', 'shen268'],\
                'nperms':100, \
                'figs_only':generate_figs_only,\
                'ensembles':['none', 'demog'],\
                'analysis_id':'analysis_5'}
        run_models(**kwargs)
        
if set(['6']).issubset(set(run_analyses)):  
        kwargs = {'covariates':['AGE', 'SEX', 'DAYS_POST_STROKE'], \
                'results_path':'/home/ubuntu/enigma/results',\
                'output_path': '/analysis_6',\
                'models_tested': ['ridge'],
                'lesionload_types': ['none', 'M1', 'all'], \
                'crossval_types':['1', '2','3', '4', '5'],\
                'atlases':['fs86subj', 'shen268'],\
                'nperms':100, \
                'figs_only':generate_figs_only,\
                'ensembles':['none', 'demog'],\
                'analysis_id':'analysis_6'}

        run_models(**kwargs)
        
scenesdir = '/home/ubuntu/enigma/motor_predictions/wb_files/'
run_analyses=['7']
generate_figs_only = False # whether to run ML models or just generate figures
workbench_vis = True # 
boxplots = True
if set(['7']).issubset(set(run_analyses)):  
        kwargs = {'covariates':['AGE', 'SEX', 'DAYS_POST_STROKE'], \
                'results_path':'/home/ubuntu/enigma/results',\
                'output_path': '/analysis_7',\
                'models_tested': ['ridge'],
                'lesionload_types': ['none'], \
                'crossval_types':['1'],\
                'atlases':['shen268'],\
                'nperms':100, \
                'figs_only':generate_figs_only,\
                'ensembles':['none'],\
                'analysis_id':'analysis_7',\
                'workbench_vis':workbench_vis,\
                'scenesdir': scenesdir,\
                'wbpath': wbpath}

        run_models(**kwargs)
        
run_analyses = [] # list of analyses to run (corresponds to analysis folders "analysis_X")

if set(['8']).issubset(set(run_analyses)):  
        kwargs = {'covariates':['AGE', 'SEX', 'DAYS_POST_STROKE'], \
                'results_path':'/home/ubuntu/enigma/results',\
                'output_path': '/analysis_8',\
                'models_tested': ['ridge'],
                'lesionload_types': ['none'], \
                'crossval_types':[ '5'],\
                'atlases':['fs86subj', 'shen268'],\
                'chaco_types':['chacoconn'],\
                'nperms':100, \
                'figs_only':generate_figs_only,\
                'ensembles':['none'],\
                'analysis_id':'analysis_8',\
                'workbench_vis':workbench_vis}

        run_models(**kwargs)


scenesdir = '/home/ubuntu/enigma/motor_predictions/wb_files/'
hcpdir ='/home/ubuntu/enigma/motor_predictions/wb_files/HCP_S1200_GroupAvg_v1/'
wbpath = '/home/ubuntu/enigma/motor_predictions/wb_files/workbench_ubuntu/bin_linux64'
run_analyses = [] # list of analyses to run (corresponds to analysis folders "analysis_X")
generate_figs_only = False # whether to run ML models or just generate figures
workbench_vis = True # 
boxplots = False
generate_wb_figures_setup(hcpdir, scenesdir)

if set(['9']).issubset(set(run_analyses)):  
        kwargs = {'covariates':['AGE', 'SEX', 'DAYS_POST_STROKE'], \
                'results_path':'/home/ubuntu/enigma/results',\
                'output_path': '/analysis_9',\
                'models_tested': ['ridge'],
                'lesionload_types': ['none'], \
                'crossval_types':['1'],\
                'atlases':['fs86subj'],\
                'chaco_types':['chacovol'],\
                'nperms':100, \
                'figs_only':generate_figs_only,\
                'ensembles':['none'],\
                'analysis_id':'analysis_9',\
                'workbench_vis':workbench_vis,\
                'scenesdir': scenesdir,\
                'wbpath': wbpath}

        run_models(**kwargs)
        
        
#run_analyses = ['10'] # list of analyses to run (corresponds to analysis folders "analysis_X")
generate_figs_only = False # whether to run ML models or just generate figures
workbench_vis = False # 
boxplots = False
generate_wb_figures_setup(hcpdir, scenesdir)

if set(['10']).issubset(set(run_analyses)):  
        kwargs = {'covariates':['AGE', 'SEX', 'DAYS_POST_STROKE'], \
                'results_path':'/home/ubuntu/enigma/results',\
                'output_path': '/analysis_10',\
                'models_tested': ['ridge'],
                'lesionload_types': ['M1', 'all','none'], \
                'crossval_types':['1'],\
                'atlases':['fs86subj', 'shen268'],\
                'chaco_types':['chacovol'],\
                'nperms':100, \
                'figs_only':generate_figs_only,\
                'ensembles':['none'],\
                'analysis_id':'analysis_10',\
                'workbench_vis':workbench_vis,\
                'scenesdir': scenesdir,\
                'wbpath': wbpath, \
                'boxplots': boxplots}

        run_models(**kwargs)
        
#run_analyses = ['11'] # list of analyses to run (corresponds to analysis folders "analysis_X")
generate_figs_only = False # whether to run ML models or just generate figures
workbench_vis = False # 
boxplots = False

if set(['11']).issubset(set(run_analyses)):  
        kwargs = {'covariates':['AGE', 'SEX', 'DAYS_POST_STROKE'], \
                'results_path':'/home/ubuntu/enigma/results',\
                'output_path': '/analysis_11',\
                'models_tested': ['ridge'],
                'lesionload_types': ['all'], \
                'crossval_types':['1'],\
                'nperms':100, \
                'figs_only':generate_figs_only,\
                'ensembles':['none'],\
                'analysis_id':'analysis_11',\
                'workbench_vis':workbench_vis,\
                'scenesdir': scenesdir,\
                'wbpath': wbpath, \
                'boxplots': boxplots}

        run_models(**kwargs)