#!/bin/bash

# python3 parse_args.py --help

# analysis 1:
# chronic motor score prediction comparing m1-cst lesion load, ipsilesional and bihemispheric lesion load, and regional chaco scores (freesurfer and shen atlases)
# cross validation type: shuffled
# scores predicted: normed motor scores
# data subset: chronic subjects
# ensemble/demographic variables: none


csv_path='/home/ubuntu/enigma/Behaviour_Information_ALL_April7_2022_sorted_CST_12_ll_slnm.csv'
nemo_settings='1mm,sdstream'
nemo_path='/home/ubuntu/enigma/lesionmasks/'
results_path='/home/ubuntu/enigma/results/'
analysis_id='analysis_1'
workbench_vis=true
scenesdir='/home/ubuntu/enigma/motor_predictions/wb_files'
hcp_dir='/home/ubuntu/enigma/motor_predictions/wb_files/HCP_S1200_GroupAvg_v1'
wbpath='/home/ubuntu/enigma/motor_predictions/wb_files/workbench_ubuntu/bin_linux64'
chaco_types='chacovol'
atlases='shen268,fs86subj'
boxplots=true
lesionload_types='none'
crossval_types='1'
verbose=0
nperms=100
save_models=true
ensembles='none'
subid_colname='BIDS_ID'
site_colname='SITE'
chronicity_colname='CHRONICITY'
yvar_colname='NORMED_MOTOR'
subsets='chronic'
y_var='normed_motor_scores'
model_specified='ridge'
override_rerunmodels=true

python3 parse_args.py --atlases $atlases --nemo_settings $nemo_settings --results_path $results_path --workbench_vis $workbench_vis --scenesdir $scenesdir --hcp_dir $hcp_dir --wbpath $wbpath --subid_colname $subid_colname --override_rerunmodels $override_rerunmodels --subset $subsets --analysis_id $analysis_id --y_var $y_var --boxplots $boxplots --csv_path $csv_path --nemo_path $nemo_path --crossval_types $crossval_types --nperms $nperms --chaco_types $chaco_types  --lesionload_types $lesionload_types --verbose $verbose --site_colname $site_colname --chronicity_colname $chronicity_colname --model_specified $model_specified --yvar_colname $yvar_colname


