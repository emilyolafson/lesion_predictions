
%% add FSL to path
% If you have installed FSL somewhere
% other than  /usr/local/fsl/, change
% this first line accordingly.
fsldir = '/usr/local/fsl/'; 
fsldirmpath = sprintf('%s/etc/matlab',fsldir);
setenv('FSLDIR', fsldir);
setenv('FSLOUTPUTTYPE', 'NIFTI_GZ');
path(path, fsldirmpath);
clear fsldir fsldirmpath;

%% load lesion network/behavior maps

% calculate lesion load
df = readtable('~/GIT/ENIGMA/data/Behaviour_Information_ALL_April7_2022_sorted_CST_12_ll.csv')
sub_left=df.LESIONED_HEMISPHERE

lesion_folder = '/Users/emilyolafson/GIT/ENIGMA/data/lesionmasks/all_lesionmasks_2009tov6_usingSym_rename/';

% LBM: FM (L+R)
fmLBM = read_avw('/Users/emilyolafson/GIT/ENIGMA/data/lnm/rawWeights_LBM.nii.gz');
% structural LNM - continuous (R only)
sLNM_R = read_avw('/Users/emilyolafson/GIT/ENIGMA/data/lnm/fm_pc1_sLNM.nii');
sLNM_L = read_avw('/Users/emilyolafson/GIT/ENIGMA/data/lnm/fm_pc1_sLNM_L.nii.gz');

sLNM_R_above0 = sLNM_R.*(sLNM_R>0);
sLNM_L = (sLNM_L).*(sLNM_L>0);

% functional LNM - continuous, pos + neg (L + R)
fLNM = read_avw('/Users/emilyolafson/GIT/ENIGMA/data/lnm/thr95per_1mm_Pos_olmax04_fLNM.nii.gz');

% calculate lesion load for each subject and append to growing table

% The lesion's intersection with each LNM was also quantified as the sum of voxel 
% intensities from a lesion network map that also appeared within the
% boundaries of a given WU patient's lesion mask

slnm = table();


for sub = 1:height(df)
    try
    lesion = read_avw([lesion_folder, df.BIDS_ID{sub},'.nii.gz']);
    lesion = logical(lesion);
    %df.BIDS_ID{sub}
    if sub_left(sub)==1
        slnm.sLNM_LL(sub)=sum(sLNM_L.*lesion,'all');
    end
    if sub_left(sub)==2
        slnm.sLNM_LL(sub)=sum(sLNM_R.*lesion,'all');
    end
    if sub_left(sub)==3 || sub_left(sub)==4 || sub_left(sub)==5 || sub_left(sub)==6
        slnm.sLNM_LL(sub)=(sum(sLNM_R.*lesion,'all')+sum(sLNM_L.*lesion,'all'))/2;
    end
    catch
        disp('Lesion data not found')
        continue
    end
end


df= [df, slnm]
writetable(df,'~/GIT/ENIGMA/data/Behaviour_Information_ALL_April7_2022_sorted_CST_12_ll_slnm.csv')



