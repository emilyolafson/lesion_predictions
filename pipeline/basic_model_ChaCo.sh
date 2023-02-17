#!/bin/bash

# python3 parse_args.py --help



csv_path='/home/ubuntu/enigma/motor_predictions/Behaviour_Information_ALL_April7_2022_sorted_CST_12_ll_slnm.csv'
results_path='/home/ubuntu/enigma/results/'
fig_path='figures'
output_folder='analysis_1'
boxplots=true
lesionload_types='none'
crossval_types='1'
chaco_types='chacovol'
atlases='shen268'
verbose=0
nperms=2
save_models=true
subid_colname='BIDS_ID'
yvar_colname='NORMED_MOTOR'
y_var='normed_motor_scores'
models_tested='ridge'
override_rerunmodels=True

python3 parse_args.py --results_path $results_path --chaco_types $chaco_types --atlases $atlases --override_rerunmodels $override_rerunmodels --output_folder $output_folder --subid_colname $subid_colname --fig_path $fig_path --y_var $y_var --csv_path $csv_path --crossval_types $crossval_types --nperms $nperms  --lesionload_types $lesionload_types --verbose $verbose --models_tested $models_tested --yvar_colname $yvar_colname
