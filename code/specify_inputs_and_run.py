
from imp import reload
import run_regression_models
reload(run_regression_models)
from run_regression_models import run_models
from helper_functions_figures import *

# This code sets up and runs a machine learning analysis on brain imaging data. 
# The analysis uses regression models and ridge regression models to study the relationship between certain covariates
# (lesion load, structural disconnection, age, sex, and days post-stroke) and motor scores 
# It will use two atlases (fs86subj and shen268) and will be run with 100 permutations. The results of the analysis will be saved in the specified
# results and output paths, and figures may be generated based on the value of the generate_figs_only and boxplots arguments.
# he workbench_vis variable controls whether to generate visualizations using Workbench, and the boxplots variable 
# controls whether to generate boxplots of the results. 
# The override_rerunmodels variable controls whether to re-run the models even if they have already been run with the same parameters.
# - ChatGPT


# The code above is defining a list of analyses to run, and setting parameters for the run_models function. 
# The run_analyses list specifies the analysis folders to run, and the scenesdir variable specifies the directory
# where Workbench scenes should be saved. The wbpath variable specifies the path to the Workbench command-line interface.
# The generate_figs_only variable controls whether to run the machine learning models or just generate figures from previously
# saved data. The workbench_vis variable controls whether to generate visualizations using Workbench, and the boxplots variable 
# controls whether to generate boxplots of the results. The override_rerunmodels variable controls whether to re-run the 
# models even if they have already been run with the same parameters.

run_analyses = [] # list of analyses to run (corresponds to analysis folders "analysis_X")
scenesdir = '/home/ubuntu/enigma/motor_predictions/wb_files/'
hcpdir ='/home/ubuntu/enigma/motor_predictions/wb_files/HCP_S1200_GroupAvg_v1/'
wbpath = '/home/ubuntu/enigma/motor_predictions/wb_files/workbench_ubuntu/bin_linux64'
run_analyses = [] # list of analyses to run (corresponds to analysis folders "analysis_X")
generate_figs_only = True # whether to run ML models or just generate figures
workbench_vis = False # 
boxplots = True
override_rerunmodels = False

if set(['1']).issubset(set(run_analyses)):  
        kwargs = {'covariates':['AGE', 'SEX', 'DAYS_POST_STROKE'], \
                'results_path':'/home/ubuntu/enigma/results',\
                'output_path': '/analysis_1',\
                'models_tested': ['ridge'],\
                'lesionload_types': [ 'M1', 'all', 'all_2h','none'], \
                'crossval_types':['1'],\
                'atlases':['fs86subj', 'shen268'],\
                'chaco_types':['chacovol'],\
                'nperms':100, \
                'figs_only':generate_figs_only,\
                'analysis_id':'analysis_1', \
                'boxplots': boxplots, \
                'override_rerunmodels': override_rerunmodels,\
                'workbench_vis':workbench_vis,\
                'scenesdir': scenesdir,\
                'wbpath': wbpath}
        run_models(**kwargs)
        
scenesdir = '/home/ubuntu/enigma/motor_predictions/wb_files/'
hcpdir ='/home/ubuntu/enigma/motor_predictions/wb_files/HCP_S1200_GroupAvg_v1/'
wbpath = '/home/ubuntu/enigma/motor_predictions/wb_files/workbench_ubuntu/bin_linux64'
run_analyses = [] # list of analyses to run (corresponds to analysis folders "analysis_X")
generate_figs_only = True # whether to run ML models or just generate figures
workbench_vis = False # 
boxplots = True
override_rerunmodels = False

generate_wb_figures_setup(hcpdir, scenesdir)
if set(['2']).issubset(set(run_analyses)):  
        kwargs = {'covariates':['AGE', 'SEX', 'DAYS_POST_STROKE'], \
                'results_path':'/home/ubuntu/enigma/results',\
                'output_path': '/analysis_2',\
                'models_tested': ['ridge'],\
                'lesionload_types':  [ 'M1', 'all', 'all_2h','none'], \
                'crossval_types':['5'],\
                'ensembles':['none'],\
                'atlases':['fs86subj', 'shen268'],\
                'nperms':100, \
                'figs_only':generate_figs_only,\
                'analysis_id':'analysis_2', \
                'boxplots':boxplots,\
                'override_rerunmodels': override_rerunmodels,\
                'workbench_vis':workbench_vis,\
                'scenesdir': scenesdir,\
                'wbpath': wbpath}
        run_models(**kwargs)


scenesdir = '/home/ubuntu/enigma/motor_predictions/wb_files/'
hcpdir ='/home/ubuntu/enigma/motor_predictions/wb_files/HCP_S1200_GroupAvg_v1/'
wbpath = '/home/ubuntu/enigma/motor_predictions/wb_files/workbench_ubuntu/bin_linux64'
run_analyses = ['3'] # list of analyses to run (corresponds to analysis folders "analysis_X")
generate_figs_only = False # whether to run ML models or just generate figures
workbench_vis = False # 
boxplots = False
override_rerunmodels = False

generate_wb_figures_setup(hcpdir, scenesdir)
if set(['3']).issubset(set(run_analyses)):  
        kwargs = {'covariates':['AGE', 'SEX', 'DAYS_POST_STROKE'], \
                'results_path':'/home/ubuntu/enigma/results',\
                'output_path': '/analysis_3',\
                'models_tested': ['ridge'],\
                'lesionload_types':  ['M1', 'all','all_2h', 'none'], \
                'crossval_types':['5'],\
                'subsets': ['acute', 'chronic'], \
                'ensembles':['demog'],\
                'atlases':['fs86subj', 'shen268'],\
                'nperms':100, \
                'figs_only':generate_figs_only,\
                'analysis_id':'analysis_3', \
                'boxplots':boxplots,\
                'override_rerunmodels': override_rerunmodels,\
                'workbench_vis':workbench_vis,\
                'scenesdir': scenesdir,\
                'wbpath': wbpath}
        run_models(**kwargs)
        
scenesdir = '/home/ubuntu/enigma/motor_predictions/wb_files/'
hcpdir ='/home/ubuntu/enigma/motor_predictions/wb_files/HCP_S1200_GroupAvg_v1/'
wbpath = '/home/ubuntu/enigma/motor_predictions/wb_files/workbench_ubuntu/bin_linux64'
run_analyses = [] # list of analyses to run (corresponds to analysis folders "analysis_X")
generate_figs_only = False # whether to run ML models or just generate figures
workbench_vis = False # 
boxplots = True
override_rerunmodels = False

generate_wb_figures_setup(hcpdir, scenesdir)
if set(['14']).issubset(set(run_analyses)):  
        kwargs = {'covariates':['AGE', 'SEX', 'DAYS_POST_STROKE'], \
                'results_path':'/home/ubuntu/enigma/results',\
                'output_path': '/analysis_14',\
                'models_tested': ['ridge'],\
                'lesionload_types':  ['M1', 'all','all_2h', 'none'], \
                'crossval_types':['5'],\
                'subsets': ['acute', 'chronic'], \
                'ensembles':['demog'],\
                'atlases':['fs86subj', 'shen268'],\
                'nperms':100, \
                'figs_only':generate_figs_only,\
                'analysis_id':'analysis_14', \
                'boxplots':boxplots,\
                'override_rerunmodels': override_rerunmodels,\
                'workbench_vis':workbench_vis,\
                'scenesdir': scenesdir,\
                'wbpath': wbpath}
        run_models(**kwargs)
        
if set(['6']).issubset(set(run_analyses)):  
        kwargs = {'covariates':['AGE', 'SEX', 'DAYS_POST_STROKE'], \
                'results_path':'/home/ubuntu/enigma/results',\
                'output_path': '/analysis_6',\
                'models_tested': ['ridge'],
                'lesionload_types': ['none', 'M1', 'all'], \
                'crossval_types':['1'],\
                'atlases':['fs86subj', 'shen268'],\
                'nperms':100, \
                'figs_only':generate_figs_only,\
                'ensembles':['none', 'demog'],\
                'analysis_id':'analysis_6'}

        run_models(**kwargs)
        

scenesdir = '/home/ubuntu/enigma/motor_predictions/wb_files/'
hcpdir ='/home/ubuntu/enigma/motor_predictions/wb_files/HCP_S1200_GroupAvg_v1/'
wbpath = '/home/ubuntu/enigma/motor_predictions/wb_files/workbench_ubuntu/bin_linux64'
run_analyses = [] # list of analyses to run (corresponds to analysis folders "analysis_X")
generate_figs_only = False # whether to run ML models or just generate figures
workbench_vis = False # 
boxplots = True
override_rerunmodels = False

generate_wb_figures_setup(hcpdir, scenesdir)
if set(['14']).issubset(set(run_analyses)):  
        kwargs = {'covariates':['AGE', 'SEX', 'DAYS_POST_STROKE'], \
                'results_path':'/home/ubuntu/enigma/results',\
                'output_path': '/analysis_14',\
                'models_tested': ['ridge'],\
                'lesionload_types':  ['M1', 'all','all_2h', 'none'], \
                'crossval_types':['5'],\
                'subsets': ['acute', 'chronic'], \
                'ensembles':['none'],\
                'atlases':['fs86subj', 'shen268'],\
                'nperms':100, \
                'figs_only':generate_figs_only,\
                'analysis_id':'analysis_14', \
                'boxplots':boxplots,\
                'override_rerunmodels': override_rerunmodels,\
                'workbench_vis':workbench_vis,\
                'scenesdir': scenesdir,\
                'wbpath': wbpath}
        run_models(**kwargs)
        
        
        
scenesdir = '/home/ubuntu/enigma/motor_predictions/wb_files/'
hcpdir ='/home/ubuntu/enigma/motor_predictions/wb_files/HCP_S1200_GroupAvg_v1/'
wbpath = '/home/ubuntu/enigma/motor_predictions/wb_files/workbench_ubuntu/bin_linux64'
#run_analyses = ['12'] # list of analyses to run (corresponds to analysis folders "analysis_X")
generate_figs_only = True # whether to run ML models or just generate figures
workbench_vis = False # 
boxplots = True
override_rerunmodels = False

generate_wb_figures_setup(hcpdir, scenesdir)
if set(['12']).issubset(set(run_analyses)):  
        kwargs = {'covariates':['AGE', 'SEX', 'DAYS_POST_STROKE'], \
                'results_path':'/home/ubuntu/enigma/results',\
                'output_path': '/analysis_12',\
                'models_tested': ['ridge'],\
                'lesionload_types':  ['M1', 'all', 'all_2h', 'none'], \
                'crossval_types':[ '5'],\
                'subsets': ['acute', 'chronic'], \
                'ensembles':['none'],\
                'atlases':['fs86subj', 'shen268'],\
                'nperms':100, \
                'figs_only':generate_figs_only,\
                'analysis_id':'analysis_12', \
                'boxplots':boxplots,\
                'override_rerunmodels': override_rerunmodels,\
                'workbench_vis':workbench_vis,\
                'scenesdir': scenesdir,\
                'wbpath': wbpath}
        run_models(**kwargs)


scenesdir = '/home/ubuntu/enigma/motor_predictions/wb_files/'
hcpdir ='/home/ubuntu/enigma/motor_predictions/wb_files/HCP_S1200_GroupAvg_v1/'
wbpath = '/home/ubuntu/enigma/motor_predictions/wb_files/workbench_ubuntu/bin_linux64'
run_analyses = [] # list of analyses to run (corresponds to analysis folders "analysis_X")
generate_figs_only = True # whether to run ML models or just generate figures
workbench_vis = False # 
boxplots = True
override_rerunmodels = False

generate_wb_figures_setup(hcpdir, scenesdir)
if set(['13']).issubset(set(run_analyses)):  
        kwargs = {'covariates':['AGE', 'SEX', 'DAYS_POST_STROKE'], \
                'results_path':'/home/ubuntu/enigma/results',\
                'output_path': '/analysis_13',\
                'models_tested': ['ridge'],\
                'lesionload_types':  ['M1', 'all','all_2h', 'none'], \
                'crossval_types':[ '1'],\
                'subsets': ['acute', 'chronic'], \
                'ensembles':['none'],\
                'atlases':['fs86subj', 'shen268'],\
                'nperms':100, \
                'figs_only':generate_figs_only,\
                'analysis_id':'analysis_13', \
                'boxplots':boxplots,\
                'override_rerunmodels': override_rerunmodels,\
                'workbench_vis':workbench_vis,\
                'scenesdir': scenesdir,\
                'wbpath': wbpath}
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
        
        


        
scenesdir = '/home/ubuntu/enigma/motor_predictions/wb_files/'
#run_analyses=['7']
generate_figs_only = False # whether to run ML models or just generate figures
workbench_vis = True # 
boxplots = True
if set(['7']).issubset(set(run_analyses)):  
        kwargs = {'covariates':['AGE', 'SEX', 'DAYS_POST_STROKE'], \
                'results_path':'/home/ubuntu/enigma/results',\
                'output_path': '/analysis_7',\
                'models_tested': ['ridge'],
                'lesionload_types': ['none'], \
                'crossval_types':['5'],\
                'atlases':['fs86subj', 'shen268'],\
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