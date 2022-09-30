
%% flip all subjects with L lesions to the R
%findND program from FileExchange

df=readtable('~/GIT/ENIGMA/data/Behaviour_Information_ALL_April7_2022_sorted.csv')

%df=df(~isnan(df.CHRONICITY),:)
%df=df(~isnan(df.SEX),:)
%df=df(~isnan(df.AGE),:)
%df=df(~isnan(df.LESIONED_HEMISPHERE),:)
%df=df(~isnan(df.NORMED_MOTOR),:)

%df=df(df.CHRONICITY==180,:)

histogram(df.DAYS_POST_STROKE)
title('Days post stroke')

mean(df.DAYS_POST_STROKE, 'omitnan')
sum(isnan(df.DAYS_POST_STROKE))

max(df.DAYS_POST_STROKE)

sub_left=df.LESIONED_HEMISPHERE==1 % to flip
sub_left=df.BIDS_ID(logical(sub_left))

counter=0
for i=1:size(df,1) %copy everything to say flipped
    counter=counter+1;
    [status, cmdout]=system(sprintf('cp %s* %s', ['/Users/emilyolafson/GIT/ENIGMA/data/lesionmasks/2mm/', cell2mat(df.BIDS_ID(i))], ['/Users/emilyolafson/GIT/ENIGMA/data/lesionmasks/2mm/', cell2mat(df.BIDS_ID(i)), '_flipped.nii.gz'] ));
    if status==1
        disp(i)
    end
end

% get sagittal midcoordinate and determine whether it's on the L or R side.
cstatlas=read_avw('/usr/local/fsl/data/atlases/JHU/JHU-ICBM-tracts-prob-2mm.nii.gz');
Rcst=squeeze(cstatlas(:,:,:,3));
Rcst=double(Rcst)

% midsagittal plane: x = 45
% 0:44 = right
% 46:91 = left

addpath('/usr/local/fsl/')
counter=0

lesionlist=dir('/Users/emilyolafson/GIT/ENIGMA/data/lesionmasks/2mm/*flipped*')
lesionlist=struct2table(lesionlist)
lesionlist=lesionlist.name

for i=1:size(lesionlist,1)
    lesion=read_avw(['/Users/emilyolafson/GIT/ENIGMA/data/lesionmasks/2mm/', cell2mat(lesionlist(i))]);
    %sum(squeeze(sum(Rcst.*logical(lesion))))
    [x y z]=findND(logical(lesion));
    meanx=mean(x);
    if meanx<45
        % lesion trueside 0 = right, 1 = left
        lesion_trueside(i)=0 %right
    elseif meanx>45 
        lesion_trueside(i)=1 %left
    end
end

mni=niftiread('/usr/local/fsl/data/standard/MNI152_T1_1mm_brain.nii.gz');
mnihdr=niftiinfo('/usr/local/fsl/data/standard/MNI152_T1_1mm_brain.nii.gz');

size(mni)
% x = 90 to 218 KEEP
% y = 1 to 41 KEEP
% z =  1 to 182 KEEP 

zeromni=zeros(182, 218, 182);
zeromni2=zeromni;
zeromni2(1:81, 1:182, 1:79)=1;

zeromni(79:182, 1:218, 1:182)=1;

zero3=zeromni+zeromni2;

mninew=double(mni).*zeromni ;
tmp=int16(mninew);


niftiwrite(tmp, '~/GIT/display3', mnihdr);


