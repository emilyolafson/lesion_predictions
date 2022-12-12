#!/bin/bash

# python3 parse_args.py --help

csv_path='/home/ubuntu/enigma/Behaviour_Information_ALL_April7_2022_sorted_CST_12_ll_slnm.csv'
nemo_path='/home/ubuntu/enigma/lesionmasks/'
analysis_id='analysis_2'
chaco_types='chacovol'
atlases='shen268,fs86subj'
lesionload_types='M1,all,all_2h'
crossval_types='1'
verbose=0
nperms=100
site_colname='SITE'
chronicity_colname='CHRONICITY'
yvar_colname='NORMED_MOTOR'
model_specified='ridge'


python3 parse_args.py  --analysis_id $analysis_id --boxplots True --csv_path $csv_path --nemo_path $nemo_path --crossval_types $crossval_types --nperms $nperms --chaco_types $chaco_types --atlases $atlases --lesionload_types $lesionload_types --verbose $verbose --site_colname $site_colname --chronicity_colname $chronicity_colname --model_specified $model_specified --yvar_colname $yvar_colname

