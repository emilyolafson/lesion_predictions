import nibabel as nib



surface_fileR ='/home/ubuntu/enigma/results/analysis_1/shen268_normed_motor_scores_chacovol_chronic_ridge_crossval1_meanbetas_allperms_50_surfacefile_betasR.shape.gii'

giftimetaR = nib.load('/home/ubuntu/enigma/motor_predictions/wb_files/surfmetadataR.shape.gii') # reference gifti
giftimetaR.darrays[0].data = nib.load(surface_fileR).darrays[0].data
print(giftimetaR.header)