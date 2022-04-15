
%% SDC 
% Load NeMo voxelwise data 
files=dir('/Users/emilyolafson/GIT/ENIGMA/data/NeMo_2mm_deterministic_voxelwise/')
esc=[];

leng=length(files)

% load NeMo nifti files to n x p matrix (902629 x nsubjects)
for i=8:leng
    filename=files(i).name;
    esc=cat(2,esc,reshape(read_avw(filename), 902629, []));
end

%principle components analysis of voxelwise data
[coeff, score, ~, ~, explained]=pca(esc);

%save components
save '/Users/emilyolafson/GIT/ENIGMA/PCA/PCA_coeff.mat' 'coeff'
save '/Users/emilyolafson/GIT/ENIGMA/PCA/PCA_explained.mat' 'explained'

%plot variance explained 
bar(explained);

% find # components needed to explain >90% of the variance in the data (N)
sum(explained(1:35))

% save only first N components 
score_subset=score(:,1:35);
save '/Users/emilyolafson/GIT/ENIGMA/PCA/PCA_score_first35.mat' 'score_subset'

coeff_subset=coeff(:,1:35);
save '/Users/emilyolafson/GIT/ENIGMA/PCA/PCA_coeff_first35.mat' 'coeff_subset'

explained_subset=explained(1:35);
save '/Users/emilyolafson/GIT/ENIGMA/PCA/PCA_explained_first35.mat' 'explained_subset'

%save components as .nii files for visualization.
ncomp=35
for j=1:ncomp
    comp=reshape(score(:,j), [91 109 91]);
    save_avw(comp, sprintf('/Users/emilyolafson/GIT/ENIGMA/PCA/comp%i.nii', j),'f', [2 2 2 2])
end

%% Lesion data
% downsample to 2mm voxels (currently 1mm)

% Load lesion voxelwise data 
files=dir('/Users/emilyolafson/GIT/ENIGMA/data/lesionmasks/')
esc=[];

leng=length(files)
% load lesion nifti files to n x p matrix (902629 x nsubjects)
for i=3:leng
    filename=files(i).name;
    esc=cat(2,esc,reshape(read_avw(filename), 902629, []));
end

%principle components analysis of voxelwise data
[coeff, score, ~, ~, explained]=pca(esc);

%save components
save '/Users/emilyolafson/GIT/ENIGMA/PCA/lesionPCA_coeff.mat' 'coeff'
save '/Users/emilyolafson/GIT/ENIGMA/PCA/lesionPCA_explained.mat' 'explained'

%plot variance explained 
bar(explained);

% find # components needed to explain >90% of the variance in the data (N)
sum(explained(1:180))

% save only first N components 
score_subset=score(:,1:180);
save '/Users/emilyolafson/GIT/ENIGMA/PCA/lesionPCA_score_first180.mat' 'score_subset'

coeff_subset=coeff(:,1:180);
save '/Users/emilyolafson/GIT/ENIGMA/PCA/lesionPCA_coeff_first180.mat' 'coeff_subset'

explained_subset=explained(1:180);
save '/Users/emilyolafson/GIT/ENIGMA/PCA/lesionPCA_explained_first180.mat' 'explained_subset'

%save components as .nii files for visualization.
ncomp=180
for j=1:ncomp
    comp=reshape(score(:,j), [91 109 91]);
    save_avw(comp, sprintf('/Users/emilyolafson/GIT/ENIGMA/PCA/lesioncomp%i.nii', j),'f', [2 2 2 2])
end