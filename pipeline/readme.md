![Pipeline](pipeline.png)

usage: parse_args.py [-h] [--nemo_path NEMO_PATH]
                     [--nemo_settings NEMO_SETTINGS]
                     [--motor_colname MOTOR_COLNAME]
                     [--subid_colname SUBID_COLNAME]
                     [--site_colname SITE_COLNAME]
                     [--chronicity_colname CHRONICITY_COLNAME]
                     [--csv_path CSV_PATH] [--y_var Y_VAR] [--subsets SUBSETS]
                     [--model_specified MODEL_SPECIFIED] [--verbose VERBOSE]
                     [--covariates COVARIATES]
                     [--lesionload_types LESIONLOAD_TYPES] [--nperms NPERMS]
                     [--save_models SAVE_MODELS] [--ensembles ENSEMBLES]
                     [--atlases ATLASES] [--chaco_types CHACO_TYPES]
                     [--crossval_types CROSSVAL_TYPES] [--null NULL]
                     [--results_path RESULTS_PATH] [--output_path OUTPUT_PATH]
                     [--figs_only FIGS_ONLY] [--analysis_id ANALYSIS_ID]
                     [--workbench_vis WORKBENCH_VIS] [--scenesdir SCENESDIR]
                     [--hcp_dir HCP_DIR] [--wbpath WBPATH]
                     [--boxplots BOXPLOTS] [--ensemble_atlas ENSEMBLE_ATLAS]
                     [--override_rerunmodels OVERRIDE_RERUNMODELS]

Set up and run machine learning pipeline for lesion biomarker data.

optional arguments:
  -h, --help            show this help message and exit
  --nemo_path NEMO_PATH
                        Absolute path where NeMo outputs (.pkl files) are
                        located, default='/home/ubuntu/enigma/lesionmasks/'
  --nemo_settings NEMO_SETTINGS
                        Nemo settings, default=['1mm','sdstream']
  --motor_colname MOTOR_COLNAME
                        Column name of motor scores in .csv file,
                        default='NORMED_MOTOR'
  --subid_colname SUBID_COLNAME
                        Column name of subject IDs in .csv file,
                        default='BIDS_ID'
  --site_colname SITE_COLNAME
                        Column name of sites variriable in .csv file, options:
                        'site', 'none', 'SITE', etc. default='SITE'
  --chronicity_colname CHRONICITY_COLNAME
                        Column name of chronicity variable .csv file. Chronic
                        subjects have value 180, acute subjects have value 90,
                        default='CHRONICITY'
  --csv_path CSV_PATH   Absolute path where .csv file containing
                        demographic/motor scores is stored, default='/home/ubu
                        ntu/enigma/Behaviour_Information_ALL_April7_2022_sorte
                        d_CST_12_ll_slnm.csv'
  --y_var Y_VAR         Dependent variable in regression models (column name
                        in .csv), default='normed_motor_scores'
  --subsets SUBSETS     Subset of data to use for analysis, options: 'acute',
                        'chronic', 'none' default=['chronic']
  --model_specified MODEL_SPECIFIED
                        Machine learning models to run, default='ridge'
  --verbose VERBOSE     Whether to print out verbose output, default=True
  --covariates COVARIATES
                        Covariates to include in model, default=[]
  --lesionload_types LESIONLOAD_TYPES
                        Lesion load types to use, Options, ['none', 'M1',
                        'all', 'all_2h'], default=['none]
  --nperms NPERMS       Number of permutations to run, default=1
  --save_models SAVE_MODELS
                        Whether to save trained models, default=True
  --ensembles ENSEMBLES
                        What ensemble to run. Options: 'demog', 'none',
                        'chaco_ll', 'chaco_ll_demog', default=['none']
  --atlases ATLASES     Which atlas to use, 'none', 'fs86subj', 'shen268',
                        default=['fs86subj']
  --chaco_types CHACO_TYPES
                        Regional or pairwise chaco type, Options: 'none',
                        'chacovol', 'chacoconn', default=['none']
  --crossval_types CROSSVAL_TYPES
                        Which cross-validation scheme to use, Options = ['1',
                        '2', '3', '4', '6'], default=['1']
  --null NULL           Value to use for null entries in data, default=-1 (no
                        null model)
  --results_path RESULTS_PATH
                        Absolute path to save results (model outputs in .npy
                        format), default='/home/ubuntu/enigma/results'
  --output_path OUTPUT_PATH
                        Directory below results_path to save outputs, for
                        multiple analyses, default='/analysis_1'
  --figs_only FIGS_ONLY
                        Whether to only save figures, without running models,
                        default=False
  --analysis_id ANALYSIS_ID
                        Identifier for analysis. Saves figures to this
                        directory, default='analysis_1'
  --workbench_vis WORKBENCH_VIS
                        Whether to generate visualizations using Workbench,
                        default=False
  --scenesdir SCENESDIR
                        Directory where Workbench scenes are saved, default='/
                        home/ubuntu/enigma/motor_predictions/wb_files'
  --hcp_dir HCP_DIR     Directory where HCP defaults are saved, default='/home
                        /ubuntu/enigma/motor_predictions/wb_files/HCP_S1200_Gr
                        oupAvg_v1'
  --wbpath WBPATH       path to Workbench command-line interface, default='/ho
                        me/ubuntu/enigma/motor_predictions/wb_files/workbench/
                        bin_linux64'
  --boxplots BOXPLOTS   Whether to generate boxplots of results, default=False
  --ensemble_atlas ENSEMBLE_ATLAS
                        Which ChaCo atlas to use, if running ensemble models
                        with lesion data + ChaCo scores, options
                        'none','fs86subj', 'shen268', default='fs86subj
  --override_rerunmodels OVERRIDE_RERUNMODELS
                        Whether to re-run models even if already run with same
                        parameters, default=False
