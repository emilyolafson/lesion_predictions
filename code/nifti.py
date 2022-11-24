import nibabel as nib
scenesdir = '/home/ubuntu/enigma/motor_predictions/wb_files/'

giftimetaR = nib.load(scenesdir + 'surfmetadataR.shape.gii')

testfile = nib.load('/home/ubuntu/enigma/results/analysis_9/shen268_normed_motor_scores_chacovol_chronic_ridge_crossval1_meanfeatureweight_allperms_50_surfacefileR.shape.gii')
print(vars(giftimetaR))

#workbench changes the dataarray metadata.


giftimetaR.darrays[0].data = testfile.darrays[0].data

newgifti = nib.gifti.gifti.GiftiImage(header=giftimetaR.header, extra=None, file_map = giftimetaR.file_map, labeltable=giftimetaR.labeltable, darrays=giftimetaR.darrays, meta = giftimetaR.meta, version='1.0')

nib.save(newgifti, scenesdir+'surfmetadataR_shenlabels.shape.gii')
