![Pipeline](pipeline.png)

## Some assumptions of the pipeline:

- Your data is stored in a .csv file that contains subjects in rows, with named columns containing 
1) subject IDs (e.g. BIDS_ID)
2) A singular outcome variable

and optionally,

3) covariates of interest (e.g. sex, age)
4) lesion load values 
5) chronicity

Column names can be entered into the pipeline according to the documentation below.

- Subject IDs correspond to NeMo outputs
- 


## Outputs:

### Basic outputs you may care about:

- {fileprefix}_scores.py
-- R^2 scores for all test folds in the outer loop
- {fileprefix}_correlations.npy
-- Correlations between true & predicted outcomes for all test folds in outer loop
- {fileprefix}_beta_coeffs.npy
-- Beta coefficients for features
- {fileprefix}_model.py 
-- The final trained model for all training folds
- {fileprefix}_test_group_sizes.py 
-- Size of subjects in test folds.

The information entered into parse_args.py will be used to save results into files, producing prefixes according to the structure:

```
fileprefix = {atlas}_{y_var}_{chaco_type}_{subset}_{model_specified}_crossval{crossval_type}_{n}_
```
where n is the permutation #.
e.g.,

```
shen268_normed_motor_scores_chacovol_chronic_ridge_crossval1_perm0_
```
would be the prefix using the atlas "shen268", where y_var is "normed_motor_scores", chaco_types is "chacovol", subsets is "chronic", model_specified is "ridge", crossval_type is "1"

Files are saved as .npy pickled objects in the folder results_path.

See analysis_1.sh for an example of how to call parse_args.py and run the model.


## Documentation of inputs
```
usage: parse_args.py [-h] [--nemo_path NEMO_PATH] [--nemo_settings NEMO_SETTINGS] [--yvar_colname YVAR_COLNAME] [--subid_colname SUBID_COLNAME] [--site_colname SITE_COLNAME] [--chronicity_colname CHRONICITY_COLNAME] [--csv_path CSV_PATH] [--y_var Y_VAR] [--subsets SUBSETS]
                     [--model_specified MODEL_SPECIFIED] [--verbose VERBOSE] [--covariates COVARIATES] [--lesionload_types LESIONLOAD_TYPES] [--nperms NPERMS] [--save_models SAVE_MODELS] [--ensembles ENSEMBLES] [--atlases ATLASES] [--chaco_types CHACO_TYPES] [--crossval_types CROSSVAL_TYPES]
                     [--null NULL] [--results_path RESULTS_PATH] [--output_path OUTPUT_PATH] [--figs_only FIGS_ONLY] [--analysis_id ANALYSIS_ID] [--workbench_vis WORKBENCH_VIS] [--scenesdir SCENESDIR] [--hcp_dir HCP_DIR] [--wbpath WBPATH] [--boxplots BOXPLOTS] [--ensemble_atlas ENSEMBLE_ATLAS]
                     [--override_rerunmodels OVERRIDE_RERUNMODELS]

Set up and run machine learning pipeline for lesion biomarker data.

optional arguments:
  -h, --help            show this help message and exit
  --nemo_path NEMO_PATH
                        Absolute path where NeMo outputs (subject_*_mean.pkl files) are located, default='/home/ubuntu/enigma/lesionmasks/'
  --nemo_settings NEMO_SETTINGS
                        Settings used in Network Modification Tool (used to access output files). Default=['1mm','sdstream']
  --yvar_colname YVAR_COLNAME
                        Column name of motor scores (to be predicted) in .csv file, default='NORMED_MOTOR'
  --subid_colname SUBID_COLNAME
                        Column name of subject IDs in .csv file, default='BIDS_ID'
  --site_colname SITE_COLNAME
                        Column name of the sites variable in .csv file. If subjects are all from the same site, specify 'none' (default). Options: 'site', 'none', 'SITE', etc. default='none'
  --chronicity_colname CHRONICITY_COLNAME
                        Column name of chronicity variable .csv file. Chronic subjects should have a value of 180, acute subjects should have value 90. If subjects are all of one type, specify 'none' (default). Default='none'
  --csv_path CSV_PATH   Absolute path where .csv file containing demographic/motor scores is stored, default='/home/ubuntu/enigma/Behaviour_Information_ALL_April7_2022_sorted_CST_12_ll_slnm.csv'
  --y_var Y_VAR         Dependent variable in regression models (column name in .csv), default='normed_motor_scores'
  --subsets SUBSETS     Subset of data to use for analysis, options: 'acute', 'chronic', 'none' default=['chronic']
  --model_specified MODEL_SPECIFIED
                        Machine learning model used for ChaCo score-based prediction. Note that models for lesion load predictions are hard-coded. If running only lesion-load based predictions, specify 'none', Default='none'
  --verbose VERBOSE     Whether to print out verbose output, default=True
  --covariates COVARIATES
                        Covariates to include in model, default=[]
  --lesionload_types LESIONLOAD_TYPES
                        Lesion load types to use, Options, ['none', 'M1', 'all', 'all_2h'], default=['none]
  --nperms NPERMS       Number of permutations to run, default=1
  --save_models SAVE_MODELS
                        Whether to save trained models, default=True
  --ensembles ENSEMBLES
                        What ensemble to run. Options: 'demog', 'none', 'chaco_ll', 'chaco_ll_demog', default=['none']
  --atlases ATLASES     Which atlas to use, 'none', 'fs86subj', 'shen268', default=['fs86subj']
  --chaco_types CHACO_TYPES
                        Regional or pairwise chaco type, Options: 'none', 'chacovol', 'chacoconn', default=['none']
  --crossval_types CROSSVAL_TYPES
                        Which cross-validation scheme to use, Options = ['1', '2', '3', '4', '6'], default=['1']
  --null NULL           Value to use for null entries in data, default=-1 (no null model)
  --results_path RESULTS_PATH
                        Absolute path to save results (model outputs in .npy format), default='/home/ubuntu/enigma/results'
  --output_path OUTPUT_PATH
                        Directory below results_path to save outputs, for multiple analyses, default='/analysis_1'
  --figs_only FIGS_ONLY
                        Whether to only save figures, without running models, default=False
  --analysis_id ANALYSIS_ID
                        Identifier for analysis. Saves figures to this directory, default='analysis_1'
  --workbench_vis WORKBENCH_VIS
                        Whether to generate visualizations using Workbench, default=False
  --scenesdir SCENESDIR
                        Directory where Workbench scenes are saved, default='/home/ubuntu/enigma/motor_predictions/wb_files'
  --hcp_dir HCP_DIR     Directory where HCP defaults are saved, default='/home/ubuntu/enigma/motor_predictions/wb_files/HCP_S1200_GroupAvg_v1'
  --wbpath WBPATH       path to Workbench command-line interface, default='/home/ubuntu/enigma/motor_predictions/wb_files/workbench/bin_linux64'
  --boxplots BOXPLOTS   Whether to generate boxplots of results, default=False
  --ensemble_atlas ENSEMBLE_ATLAS
                        Which ChaCo atlas to use, if running ensemble models with lesion data + ChaCo scores, options 'none','fs86subj', 'shen268', default='fs86subj
  --override_rerunmodels OVERRIDE_RERUNMODELS
                        Whether to re-run models even if already run with same parameters, default=False
```
