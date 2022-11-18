% calculate the overlap between binarized sensorimotor area tracts and binary lesion mask.

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

%% load  atlas
% calculate ipsilesional CST lesion load using multiple motor regions
df=readtable('~/GIT/ENIGMA/data/Behaviour_Information_ALL_April7_2022_sorted.csv')
sub_left=df.LESIONED_HEMISPHERE==1 

% load SMATT
smatt_files = dir('/Users/emilyolafson/GIT/ENIGMA/data/lesion_load/smatt_all_motor_rois/smatt-template/*');
smatt_files=smatt_files([3:8, 10:15],:);

colnames = {'Left-M1';'Left-PMd';'Left-PMv';'Left-S1';'Left-SMA';'Left-preSMA';'Right-M1';'Right-PMd';'Right-PMv';'Right-S1';'Right-SMA';'Right-preSMA'}

% load niftis
LM1_CST =logical(read_avw([smatt_files(1).folder,'/', smatt_files(1).name]));
LPMd_CST =logical(read_avw([smatt_files(2).folder,'/', smatt_files(2).name]));
LPMv_CST =logical(read_avw([smatt_files(3).folder,'/', smatt_files(3).name]));
LS1_CST =logical(read_avw([smatt_files(4).folder,'/', smatt_files(4).name]));
LSMA_CST =logical(read_avw([smatt_files(5).folder,'/', smatt_files(5).name]));
LpreSMA_CST =logical(read_avw([smatt_files(6).folder,'/', smatt_files(6).name]));

RM1_CST =logical(read_avw([smatt_files(7).folder,'/', smatt_files(7).name]));
RPMd_CST =logical(read_avw([smatt_files(8).folder,'/', smatt_files(8).name]));
RPMv_CST =logical(read_avw([smatt_files(9).folder,'/', smatt_files(9).name]));
RS1_CST =logical(read_avw([smatt_files(10).folder,'/', smatt_files(10).name]));
RSMA_CST =logical(read_avw([smatt_files(11).folder,'/', smatt_files(11).name]));
RpreSMA_CST =logical(read_avw([smatt_files(12).folder,'/', smatt_files(12).name]));

lesion_folder = '/Users/emilyolafson/GIT/ENIGMA/data/lesionmasks/all_lesionmasks_2009tov6_usingSym_rename/';

smatt_ll = table();

% calculate lesion load for each subject and append to growing table
for sub = 1:height(df)
    try
    lesion = read_avw([lesion_folder, df.BIDS_ID{sub},'.nii.gz']);
    lesion = logical(lesion);
        if sub_left(sub)==1
            smatt_ll.M1_CST(sub)=sum(LM1_CST.*lesion,'all');
            smatt_ll.PMd_CST(sub)=sum(LPMd_CST.*lesion,'all');
            smatt_ll.PMv_CST(sub)=sum(LPMv_CST.*lesion,'all');
            smatt_ll.S1_CST(sub)=sum(LS1_CST.*lesion,'all');
            smatt_ll.SMA_CST(sub)=sum(LSMA_CST.*lesion,'all');
            smatt_ll.preSMA_CST(sub)=sum(LpreSMA_CST.*lesion,'all');
        end
        if sub_left(sub)==0
            smatt_ll.M1_CST(sub)=sum(RM1_CST.*lesion,'all');
            smatt_ll.PMd_CST(sub)=sum(RPMd_CST.*lesion,'all');
            smatt_ll.PMv_CST(sub)=sum(RPMv_CST.*lesion,'all');
            smatt_ll.S1_CST(sub)=sum(RS1_CST.*lesion,'all');
            smatt_ll.SMA_CST(sub)=sum(RSMA_CST.*lesion,'all');
            smatt_ll.preSMA_CST(sub)=sum(RpreSMA_CST.*lesion,'all');
        end
    catch
        disp('Lesion data not found')
        continue
    end
end



