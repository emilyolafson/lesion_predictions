% view all lesion maps


fid = fopen('/Users/emilyolafson/GIT/ENIGMA/data/lesionmasks/1mm/filenames.csv');
tline = fgetl(fid);
alllesions=zeros(182, 218, 182);

for i=1:1068
    tline = fgetl(fid);
    lesion = read_avw(sprintf('/Users/emilyolafson/GIT/ENIGMA/data/lesionmasks/1mm/%s', tline));
    alllesions =alllesions+lesion;
    clear lesion
end
fclose(fid);

save('/Users/emilyolafson/GIT/ENIGMA/data/lesionmasks/1068_sum.mat', 'alllesions')


save_avw(alllesions,'/Users/emilyolafson/GIT/ENIGMA/data/lesionmasks/lesion.nii.gz', 'd', [1 1 1 1]);



fslmodel = read_avw('/usr/local/fsl/data/standard/MNI152_T1_1mm_brain.nii.gz'); 
save_avw(fslmodel, '/Users/emilyolafson/GIT/ENIGMA/data/lesionmasks/fsl.nii.gz', 'd', [1 1 1 1]);
