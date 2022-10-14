#!/bin/bash

export FREESURFER_HOME=/mnt/tmp/kwj2001/freesurfer6 #keith install
source $FREESURFER_HOME/SetUpFreeSurfer.sh


freesurfer_dir=/home/emo4002/colossus_shared3/enigma

for site in $freesurfer_dir/FREESURFER/*/*
do
	subj=${site: -12}
	FILE=$freesurfer_dir/t1_fs/"$subj"_T1.nii.gz
	if [[ -f "$FILE" ]]
	then
		echo "file exists"
	else
		mri_convert $site/mri/T1.mgz $freesurfer_dir/t1_fs/"$subj"_T1.nii.gz
	fi
done

for site in $freesurfer_dir/FREESURFER/*/*
do
        subj=${site: -12}
        FILE=$freesurfer_dir/voxel_parc/"$subj"_aparc+aseg.nii.gz
        if [[ -f "$FILE" ]]
        then
                echo "file exists"
        else
                mri_convert $site/mri/aparc+aseg.mgz $freesurfer_dir/voxel_parc/"$subj"_aparc+aseg.nii.gz
        fi
done


