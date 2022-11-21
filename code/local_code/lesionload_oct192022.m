% calculate CST lesion load  for each subject

%% flip all subjects with L lesions to the R
df=readtable('~/GIT/ENIGMA/data/Behaviour_Information_ALL_April7_2022_sorted.csv')
sub_left=df.LESIONED_HEMISPHERE==1 %

df=df(~isnan(df.CHRONICITY),:)
df=df(~isnan(df.SEX),:)
df=df(~isnan(df.AGE),:)
df=df(~isnan(df.LESIONED_HEMISPHERE),:)
df=df(~isnan(df.NORMED_MOTOR),:)

df=df(df.CHRONICITY==180,:)

histogram(df.DAYS_POST_STROKE)
title('Days post stroke')

mean(df.DAYS_POST_STROKE, 'omitnan')
sum(isnan(df.DAYS_POST_STROKE))

max(df.DAYS_POST_STROKE)

sub_left=df.LESIONED_HEMISPHERE==1 %

%% load CST atlas
% calculate ipsilesional CST lesion load using multiple motor regions
df=readtable('~/GIT/ENIGMA/data/Behaviour_Information_ALL_April7_2022_sorted.csv')

% load SMATT

direct = dir('/Users/emilyolafson/GIT/ENIGMA/data/lesion_load/smatt_all_motor_rois/smatt-template/*');
direct=direct([3:8, 10:15],:)

colnames = {'Left-M1';'Left-PMd';'Left-PMv';'Left-S1';'Left-SMA';'Left-preSMA';'Right-M1';'Right-PMd';'Right-PMv';'Right-S1';'Right-SMA';'Right-preSMA'}

LM1_CST =logical(read_avw([direct(1).folder,'/', direct(1).name]));
LPMd_CST =logical(read_avw([direct(2).folder,'/', direct(2).name]));
LPMv_CST =logical(read_avw([direct(3).folder,'/', direct(3).name]));
LS1_CST =logical(read_avw([direct(4).folder,'/', direct(4).name]));
LSMA_CST =logical(read_avw([direct(5).folder,'/', direct(5).name]));
LpreSMA_CST =logical(read_avw([direct(6).folder,'/', direct(6).name]));


RM1_CST =logical(read_avw([direct(7).folder,'/', direct(7).name]));
RPMd_CST =logical(read_avw([direct(8).folder,'/', direct(8).name]));
RPMv_CST =logical(read_avw([direct(9).folder,'/', direct(9).name]));
RS1_CST =logical(read_avw([direct(10).folder,'/', direct(10).name]));
RSMA_CST =logical(read_avw([direct(11).folder,'/', direct(11).name]));
RpreSMA_CST =logical(read_avw([direct(12).folder,'/', direct(12).name]));

lesionloc = '/Users/emilyolafson/GIT/ENIGMA/data/lesionmasks/all_lesionmasks_2009tov6_usingSym_rename/';

for sub = 702:height(df)
    try
    lesion = read_avw([lesionloc, df.BIDS_ID{sub},'.nii.gz']);
    lesion = logical(lesion);
    
        if sub_left(sub)==1
            df.M1_CST(sub)=sum(LM1_CST.*lesion,'all');
            df.PMd_CST(sub)=sum(LPMd_CST.*lesion,'all');
            df.PMv_CST(sub)=sum(LPMv_CST.*lesion,'all');
            df.S1_CST(sub)=sum(LS1_CST.*lesion,'all');
            df.SMA_CST(sub)=sum(LSMA_CST.*lesion,'all');
            df.preSMA_CST(sub)=sum(LpreSMA_CST.*lesion,'all');
        end
        if sub_left(sub)==0
            df.M1_CST(sub)=sum(RM1_CST.*lesion,'all');
            df.PMd_CST(sub)=sum(RPMd_CST.*lesion,'all');
            df.PMv_CST(sub)=sum(RPMv_CST.*lesion,'all');
            df.S1_CST(sub)=sum(RS1_CST.*lesion,'all');
            df.SMA_CST(sub)=sum(RSMA_CST.*lesion,'all');
            df.preSMA_CST(sub)=sum(RpreSMA_CST.*lesion,'all');
        end
    catch
        'oops'
        continue
    end
end

writetable(df,'~/GIT/ENIGMA/data/Behaviour_Information_ALL_April7_2022_sorted_CSTll.csv')









