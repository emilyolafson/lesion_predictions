#!/bin/bash

# python3 parse_args.py --help

csv_path='/home/ubuntu/enigma/Behaviour_Information_ALL_April7_2022_sorted_CST_12_ll_slnm.csv'
nemo_path='/home/ubuntu/enigma/lesionmasks/'

python3 parse_args.py --csv_path $csv_path --nemo_path $nemo_path --chaco_types 'chacovol' --atlases 'shen268' --lesionload_types 'M1' --verbose False

