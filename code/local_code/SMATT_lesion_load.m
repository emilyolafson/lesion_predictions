% calculate the pct overlap between sensorimotor area tracts and binary lesion mask.

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
sub_left=df.LESIONED_HEMISPHERE

% load SMATT
smatt_files = dir('/Users/emilyolafson/GIT/ENIGMA/data/lesion_load/smatt_all_motor_rois/smatt-template/*.nii');
smatt_files=smatt_files([1:12],:);

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

% calculate vol of each roi
vol_lm1 = sum(LM1_CST>0, 'all')
vol_lpmd = sum(LPMd_CST>0, 'all')
vol_lpmv = sum(LPMv_CST>0, 'all')
vol_ls1 = sum(LS1_CST>0, 'all')
vol_lsma = sum(LSMA_CST>0, 'all')
vol_lpsma = sum(LpreSMA_CST>0, 'all')

vol_rm1 = sum(RM1_CST>0, 'all')
vol_rpmd = sum(RPMd_CST>0, 'all')
vol_rpmv = sum(RPMv_CST>0, 'all')
vol_rs1 = sum(RS1_CST>0, 'all')
vol_rsma = sum(RSMA_CST>0, 'all')
vol_rpsma = sum(RpreSMA_CST>0, 'all')


bar([vol_lm1, vol_lpmd, vol_lpmv, vol_ls1, vol_lsma, vol_lpsma])
bar([vol_rm1, vol_rpmd, vol_rpmv, vol_rs1, vol_rsma, vol_rpsma])

lesion_folder = '/Users/emilyolafson/GIT/ENIGMA/data/lesionmasks/all_lesionmasks_2009tov6_usingSym_rename/';

smatt_ll = table();

% calculate lesion load for each subject and append to growing table
for sub = 1:height(df)

    try
    subid =  df.BIDS_ID{sub};
    lesion = read_avw([lesion_folder, subid,'.nii.gz']);
    lesion = logical(lesion);
        if sub_left(sub)==1 % left
            smatt_ll.M1_CST(sub)=sum(LM1_CST.*lesion,'all')/vol_lm1;
            smatt_ll.PMd_CST(sub)=sum(LPMd_CST.*lesion,'all')/vol_lpmd;
            smatt_ll.PMv_CST(sub)=sum(LPMv_CST.*lesion,'all')/vol_lpmv;
            smatt_ll.S1_CST(sub)=sum(LS1_CST.*lesion,'all')/vol_ls1;
            smatt_ll.SMA_CST(sub)=sum(LSMA_CST.*lesion,'all')/vol_lsma;
            smatt_ll.preSMA_CST(sub)=sum(LpreSMA_CST.*lesion,'all')/vol_lpsma;
        end
        if sub_left(sub)==2 % right
            smatt_ll.M1_CST(sub)=sum(RM1_CST.*lesion,'all')/vol_rm1;
            smatt_ll.PMd_CST(sub)=sum(RPMd_CST.*lesion,'all')/vol_rpmd;
            smatt_ll.PMv_CST(sub)=sum(RPMv_CST.*lesion,'all')/vol_rpmv;
            smatt_ll.S1_CST(sub)=sum(RS1_CST.*lesion,'all')/vol_rs1;
            smatt_ll.SMA_CST(sub)=sum(RSMA_CST.*lesion,'all')/vol_rsma;
            smatt_ll.preSMA_CST(sub)=sum(RpreSMA_CST.*lesion,'all')/vol_rpsma;
        end
        
        if sub_left(sub)==3 || sub_left(sub)==4 || sub_left(sub)==5 || sub_left(sub)==6
            % 3 = bilateral
            % 4 = brainstem
            % 5 = cerebellum
            % 6 = combination of cerebral hemisphere + brainstem/cerebellum

            smatt_ll.M1_CST(sub)=(sum(RM1_CST.*lesion,'all')/vol_rm1+sum(LM1_CST.*lesion,'all')/vol_lm1)/2;
            smatt_ll.PMd_CST(sub)=(sum(RPMd_CST.*lesion,'all')/vol_rpmd+sum(LPMd_CST.*lesion,'all')/vol_lpmd)/2;
            smatt_ll.PMv_CST(sub)=(sum(RPMv_CST.*lesion,'all')/vol_rpmv+sum(LPMv_CST.*lesion,'all')/vol_lpmv)/2;
            smatt_ll.S1_CST(sub)=(sum(RS1_CST.*lesion,'all')/vol_rs1+sum(LS1_CST.*lesion,'all')/vol_ls1)/2;
            smatt_ll.SMA_CST(sub)=(sum(RSMA_CST.*lesion,'all')/vol_rsma+sum(LSMA_CST.*lesion,'all')/vol_lsma)/2;
            smatt_ll.preSMA_CST(sub)=(sum(RpreSMA_CST.*lesion,'all')/vol_rpsma+sum(LpreSMA_CST.*lesion,'all')/vol_lpsma)/2;
        end   
    catch
        disp('Lesion data not found')
        disp( df.BIDS_ID{sub})
        newline
        smatt_ll.M1_CST(sub)=NaN;
        smatt_ll.PMd_CST(sub)=NaN;
        smatt_ll.PMv_CST(sub)=NaN;
        smatt_ll.S1_CST(sub)=NaN;
        smatt_ll.SMA_CST(sub)=NaN;
        smatt_ll.preSMA_CST(sub)=NaN;
        continue
    end
end

df = [df, smatt_ll]

writetable(df,'~/GIT/ENIGMA/data/Behaviour_Information_ALL_April7_2022_sorted_CSTll.csv')


smatt_ll_2h = table();

% calculate lesion load for each subject and append to growing table
for sub = 1:height(df)

    try
    subid =  df.BIDS_ID{sub};
    lesion = read_avw([lesion_folder, subid,'.nii.gz']);
    lesion = logical(lesion);

    smatt_ll_2h.L_M1_CST(sub)=sum(LM1_CST.*lesion,'all')/vol_lm1;
    smatt_ll_2h.L_PMd_CST(sub)=sum(LPMd_CST.*lesion,'all')/vol_lpmd;
    smatt_ll_2h.L_PMv_CST(sub)=sum(LPMv_CST.*lesion,'all')/vol_lpmv;
    smatt_ll_2h.L_S1_CST(sub)=sum(LS1_CST.*lesion,'all')/vol_ls1;
    smatt_ll_2h.L_SMA_CST(sub)=sum(LSMA_CST.*lesion,'all')/vol_lsma;
    smatt_ll_2h.L_preSMA_CST(sub)=sum(LpreSMA_CST.*lesion,'all')/vol_lpsma;

    smatt_ll_2h.R_M1_CST(sub)=sum(RM1_CST.*lesion,'all')/vol_rm1;
    smatt_ll_2h.R_PMd_CST(sub)=sum(RPMd_CST.*lesion,'all')/vol_rpmd;
    smatt_ll_2h.R_PMv_CST(sub)=sum(RPMv_CST.*lesion,'all')/vol_rpmv;
    smatt_ll_2h.R_S1_CST(sub)=sum(RS1_CST.*lesion,'all')/vol_rs1;
    smatt_ll_2h.R_SMA_CST(sub)=sum(RSMA_CST.*lesion,'all')/vol_rsma;
    smatt_ll_2h.R_preSMA_CST(sub)=sum(RpreSMA_CST.*lesion,'all')/vol_rpsma;

    catch
        disp('Lesion data not found')
        disp( df.BIDS_ID{sub})
        newline
        smatt_ll_2h.L_M1_CST(sub)=NaN;
        smatt_ll_2h.L_PMd_CST(sub)=NaN;
        smatt_ll_2h.L_PMv_CST(sub)=NaN;
        smatt_ll_2h.L_S1_CST(sub)=NaN;
        smatt_ll_2h.L_SMA_CST(sub)=NaN;
        smatt_ll_2h.L_preSMA_CST(sub)=NaN;
        smatt_ll_2h.R_M1_CST(sub)=NaN;
        smatt_ll_2h.R_PMd_CST(sub)=NaN;
        smatt_ll_2h.R_PMv_CST(sub)=NaN;
        smatt_ll_2h.R_S1_CST(sub)=NaN;
        smatt_ll_2h.R_SMA_CST(sub)=NaN;
        smatt_ll_2h.R_preSMA_CST(sub)=NaN;
        continue
    end
end

df = [df, smatt_ll_2h]

writetable(df,'~/GIT/ENIGMA/data/Behaviour_Information_ALL_April7_2022_sorted_CST_12_ll.csv')
