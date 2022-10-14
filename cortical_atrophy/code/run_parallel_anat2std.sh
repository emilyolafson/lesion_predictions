#!/bin/bash

#  get_anat2std_warps.sh <subject_ID>
# This script will take T1.mgz from a freesurfer recon_all output directory, register it to MNI, and output warp files for anat2mni and mni2anat

# Set dirs on lines 25 and 27
#
# This script uses fsl_anat, which performs more steps than is necessary to obtain the warp files (e.g. tissue segmentation). This is only necessary if you are in the pecu$
#
#  Created by Parker Singleton && Keith Jamison on 3/9/22.
#

set -e #exit on error
#set -x #print each command as it runs

i=$1 #subject ID

echo ${i}

fsoutputDIR='/home/emo4002/colossus_shared3/enigma/FREESURFER'
enigmaDIR='/home/emo4002/colossus_shared3/enigma'
fsDIR='/home/emo4002/colossus_shared3/enigma/t1_fs'
#folder containing each subject's freesurfer outputs

outDIR='/home/emo4002/colossus_shared3/enigma/mni_fsoutputs'
#where to store outputs - ultimately a lot of the outputs from fsl_anat can be deleted if you do not need FSL tissue segmentation, etc. This could be skipped howev$

parcDIR='/home/emo4002/colossus_shared3/enigma/voxel_parc'

for subj in ${fsDIR}/*T1.nii.gz
do
	subj_short=${subj: -22}
	subj_short=$(basename $subj_short _T1.nii.gz)
	if ! [[ -f ${outDIR}/"$subj_short".anat/"$subj_short"_aseg_aparc_ro_toMNI.nii.gz ]]
	then
		echo $subj_short
	fi
done > ${enigmaDIR}/sub2run.txt

echo 'running subjects now'
nsubs=$(cat ${enigmaDIR}/sub2run.txt)
echo $nsubs

for F in $(cat ${enigmaDIR}/sub2run.txt) ; do
  echo ./get_anat2std_warps.sh $F
done | parallel -j 3
