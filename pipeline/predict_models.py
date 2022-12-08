import argparse
from imp import reload
import run_regression_models
reload(run_regression_models)
from run_regression_models import run_models
from helper_functions_figures import *
import pprint

def main(args):
    kwargs = vars(args)
    pprint.pprint(kwargs)
    run_models(**kwargs)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a neural net")

    # LESIONMASK_PATH: str, default ='/home/ubuntu/enigma/lesionmasks/', path to niftis
    parser.add_argument("--LESIONMASK_PATH", default='/home/ubuntu/enigma/lesionmasks/',
      help="Absolute path where lesion masks are located, default='/home/ubuntu/enigma/lesionmasks/'")
    
    # CSV_PATH: str, default ='/home/ubuntu/enigma/Behaviour_Information_ALL_April7_2022_sorted_CST_12_ll_slnm.csv', dependent variable in regression models
    parser.add_argument("--CSV_PATH", default='/home/ubuntu/enigma/Behaviour_Information_ALL_April7_2022_sorted_CST_12_ll_slnm.csv',
      help="Absolute path where demographic/motor data is stored, default='/home/ubuntu/enigma/Behaviour_Information_ALL_April7_2022_sorted_CST_12_ll_slnm.csv'")
    
    # y_var: str, default ='normed_motor_scores', dependent variable in regression models
    parser.add_argument("--y_var", default='normed_motor_scores',
      help="Dependent variable in regression models, default='normed_motor_scores'")

    # subsets: str, default = 'chronic', subset of data to use for analysis
    parser.add_argument("--subsets", default=['chronic'], type=lambda s: [item.replace(" ", "") for item in s.split(',')],
      help="Subset of data to use for analysis, default=['chronic']")

    # model_tested: list, default = ['ridge'], machine learning models to run
    parser.add_argument("--model_tested", default='ridge',
      help="Machine learning models to run, default='ridge'")

    # verbose: bool, default = True, whether to print out verbose output
    parser.add_argument("--verbose", default=True,
      help="Whether to print out verbose output, default=True")

    # covariates: list, default = [], covariates to include in model
    parser.add_argument("--covariates", default=[],
      help="Covariates to include in model, default=[]")

    # lesionload_types: list, default = [], lesion load types to use
    parser.add_argument("--lesionload_types", default=['none'], type=lambda s: [item.replace(" ", "") for item in s.split(',')],
      help="Lesion load types to use, Options, ['none', 'M1', 'all', 'all_2h'], default=['none]")

    # nperms: int, default = 1, number of permutations to run
    parser.add_argument("--nperms", type=int, default=1,
      help="Number of permutations to run, default=1")

    # save_models: bool, default = True, whether to save trained models
    parser.add_argument("--save_models", default=True,
      help="Whether to save trained models,  default=True")

    # ensembles: list, default = ['none'], what ensemble to run, "demog", "none", "chaco_ll", "chaco_ll_demog"
    parser.add_argument("--ensembles", default=['none'],
      help="What ensemble to run. Options: 'demog', 'none', 'chaco_ll', 'chaco_ll_demog', \n default=['none']")

    # atlases: list, default = ['fs86subj'], which atlas to use
    parser.add_argument("--atlases", default=['fs86subj'], type=lambda s: [item.replace(" ", "") for item in s.split(',')],
      help="Which atlas to use, default=['fs86subj']")

    # chaco_types: list, default = ['chacovol'], regional or pairwise chaco type "chacovol", "chacoconn"
    parser.add_argument("--chaco_types", default=['chacovol'], type=lambda s: [item.replace(" ", "") for item in s.split(',')],
      help="Regional or pairwise chaco type, Options: 'chacovol', 'chacoconn', default=['chacovol']")

    # crossval_types: list, default = ['1'], which cross-validation scheme to use
    parser.add_argument("--crossval_types", default=['1'], type=lambda s: [item.replace(" ", "") for item in s.split(',')],
      help="Which cross-validation scheme to use, Options = ['1', '2', '3', '4', '6'], default=['1']")
    
    # null: int, default= -1, value to use for null entries in data
    parser.add_argument("--null", default=-1,
      help="Value to use for null entries in data, default=-1 (no null model)")
    
    # results_path: str, default = '/ubuntu/home/enigma/results/', where to save results
    parser.add_argument("--results_path", default='/home/ubuntu/enigma/results',
      help="Absolute path to save results (model outputs in .npy format), default='/home/ubuntu/enigma/results'")
    
    # output_path: str, default = '/analysis_1', where to save files (npy, not figures)
    parser.add_argument("--output_path", default='analysis_1',
      help="Directory below results_path to save outputs, for multiple analyses, default='/analysis_1'")
    
    # figs_only: bool, default = False, whether to only save figures, without data
    parser.add_argument("--figs_only", default=False,
      help="Whether to only save figures, without running models, default=False")
    
    # analysis_id: str, default = 'analysis_1', identifier for analysis
    parser.add_argument("--analysis_id", default='analysis_1',
      help="Identifier for analysis. Saves figures to this directory, default='analysis_1'")
    
    # workbench_vis: bool, default = False, whether to generate visualizations using Workbench
    parser.add_argument("--workbench_vis", default=True,
      help="Whether to generate visualizations using Workbench, default=False")
    
    # scenesdir: str, default = '/wb_files', directory where Workbench scenes are saved
    parser.add_argument("--scenesdir", default='/home/ubuntu/enigma/motor_predictions/wb_files',
      help="Directory where Workbench scenes are saved, default='/home/ubuntu/enigma/motor_predictions/wb_files'")
    
    parser.add_argument("--hcp_dir", default= '/home/ubuntu/enigma/motor_predictions/wb_files/HCP_S1200_GroupAvg_v1',
      help="Directory where HCP defaults are saved, default='/home/ubuntu/enigma/motor_predictions/wb_files/HCP_S1200_GroupAvg_v1'")
    
    # wbpath: str, default = '/workbench/bin_linux64/wb_command', 
    parser.add_argument("--wbpath", default='/home/ubuntu/enigma/motor_predictions/wb_files/workbench_ubuntu/bin_linux64',
      help="path to Workbench command-line interface, default='/home/ubuntu/enigma/motor_predictions/wb_files/workbench/bin_linux64'")
    
    # boxplots: bool, default = True, whether to generate boxplots of results
    parser.add_argument("--boxplots", default=False,
      help="Whether to generate boxplots of results, default=False")
    
    # ensemble_atlas: bool, default = 'fs86subj',Which ChaCo atlas to use, if running ensemble models with lesion data + ChaCo score
    parser.add_argument("--ensemble_atlas", default='fs86subj',
      help="Which ChaCo atlas to use, if running ensemble models with lesion data + ChaCo scores, default='fs86subj")
    
    # override_rerunmodels: bool, default = False, whether to re-run models even if already run with same parameters
    parser.add_argument("--override_rerunmodels", default=False,
      help="Whether to re-run models even if already run with same parameters, default=False") 
        
    args = parser.parse_args()
    main(args)