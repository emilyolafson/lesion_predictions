
# Predicting chronic motor scores from lesion data.


## Getting Started
The machine learning pipeline, which includes data formatting and model evaluation, can be found in [pipeline](pipeline)

Requirements:
```
matplotlib==3.3.4
numpy==1.20.1
pandas==1.2.4
scikit_learn==1.1.3
scipy==1.6.2
```
# Contents

1. [Predicting motor scores from M1-CST-LL](#m1-corticospinal-tract-lesion-load)
2. [Predicting motor scores from CST-LL](#sensorimotor-area-tract-template-(smatt)-lesion-load)
3. [Predicting motor scores from ChaCo scores](#change-in-connectivity-(chaco)-scores)

# M1 corticospinal tract lesion load
![M1_pic](figures/M1.png)
Calculate the lesion load on the corticospinal tract originating from ipsilesional M1. 
Template: [Sensorimotor Area Tract Template (SMATT)](http://lrnlab.org/)
Calculated as the number of lesioned voxels that intersect with the ipsilesional M1-CST.

# Sensorimotor Area Tract Template (SMATT) lesion load
![SMATT_pic](figures/all_SMATT_stacked.png)
Calcualte the lesion load on all ipsilesional corticospinal tracts originating from M1 (primary motor cortex), S1 (sensorimotor cortex), SMA (supplementary motor area), pre-SMA (pre-supplementary motor area), ventral premotor cortex (PMv), and dorsal premotor cortex (PMd).
Calculated as the proportion of lesioned voxels that intersect with each ipsilesional tract.

Template: [Sensorimotor Area Tract Template (SMATT)](http://lrnlab.org/) 

Code:

- MATLAB: SMATT_lesion_load.m (requires FSL)
- python: SMATT_lesion_load.ipynb (uses nibabel)

# Change in Connectivity (ChaCo) scores
![nemo_pic](figures/chaco-git.png)
The Network Modification Tool [NeMo 2.1](https://kuceyeski-wcm-web.s3.us-east-1.amazonaws.com/upload.html) can be used to estimate regional or pairwise change in connectivity (ChaCo) scores, given a binary lesion mask.
