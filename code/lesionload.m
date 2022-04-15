% calculate CST lesion load  for each subject

%% flip all subjects with L lesions to the R
df=readtable('~/GIT/ENIGMA/data/Behaviour_Information_ALL_April7_2022_sorted.csv')

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

sub_left=df.LESIONED_HEMISPHERE==1 % to flip
sub_left=df.BIDS_ID(logical(sub_left))

counter=0
for i=1:size(df,1) %copy everything to say flipped
    counter=counter+1
    [status, cmdout]=system(sprintf('cp %s* %s', ['/Users/emilyolafson/GIT/ENIGMA/data/lesionmasks/2mm/', cell2mat(df.BIDS_ID(i))], ['/Users/emilyolafson/GIT/ENIGMA/data/lesionmasks/2mm/', cell2mat(df.BIDS_ID(i)), '_flipped.nii.gz'] ));
end

addpath('/usr/local/fsl/')
counter=0

for i=1:size(sub_left,1) %copy everything to say flipped
    counter=counter+1
    [status, cmdout]=system(sprintf('/usr/local/fsl/bin/fslswapdim %s -x y z %s', ['/Users/emilyolafson/GIT/ENIGMA/data/lesionmasks/2mm/', cell2mat(sub_left(i)), '_flipped.nii.gz'], ['/Users/emilyolafson/GIT/ENIGMA/data/lesionmasks/2mm/', cell2mat(sub_left(i)), '_flipped.nii.gz'] ));
end

%% load R CST atlas
cstatlas=read_avw('/usr/local/fsl/data/atlases/JHU/JHU-ICBM-tracts-prob-2mm.nii.gz');
Rcst=squeeze(cstatlas(:,:,:,3));
Rcst=double(Rcst)

%% load subject lesions and calculate overlap between R CST atlas and each subject's lesion
% via Zhu et al 2010
lesionlist=dir('/Users/emilyolafson/GIT/ENIGMA/data/lesionmasks/2mm/*flipped*')
lesionlist=lesionlist(2:end)
lesionlist=struct2table(lesionlist)
lesionlist=lesionlist.name


for i=1:size(lesionlist,1)
    lesion=read_avw(['/Users/emilyolafson/GIT/ENIGMA/data/lesionmasks/2mm/', cell2mat(lesionlist(i))]);
    %sum(squeeze(sum(Rcst.*logical(lesion))))
    
    cstll(i)=sum(sum(sum(0.1*Rcst.*lesion)));
    writematrix(cstll(i), ['/Users/emilyolafson/GIT/ENIGMA/data/lesionmasks/2mm/', cell2mat(lesionlist(i)), '.txt'])
end


cstll=cstll'

%% load subject fm scores

% normal linear regression
fmnorm=df.NORMED_MOTOR
mdl=fitlm(cstll, fmnorm)
plotResiduals(mdl)
plotResiduals(mdl,'fitted')

%with binary indicator
zeroes = cstll>0
x1=log(cstll)
x1(x1==-Inf)=0
x2=logical(zeroes)

tbl=table(x1, x2, fmnorm, 'VariableNames', {'x1', 'x2', 'fm'})
mdl = fitlm(tbl, 'fm~x1+x2')

plotResiduals(mdl) 
plotResiduals(mdl,'fitted')

histogram(cstll)
histogram(log(cstll+0.001))

% transformation
x=log(cstll+1)
histogram(x)
infs=x==-Inf
meanz=mean(x(~infs))
x(x==-Inf)=meanz

mdl=fitlm(normalize(fmnorm), x)
plotResiduals(mdl)
plotResiduals(mdl,'fitted')


corr(fmnorm, cstll)

scatter(cstll, fmnorm)


% logistic regression
ynew=logical(fmnorm>0.5)

[B,dev,stats]=mnrfit(cstll, ynew+1)

histogram(stats.resid(:,1))


% cutoffs
%(mild: 42-60, medium: 28-41, and severe: 0-27)
% Woytowicz, E. J., Rietschel, J. C., Goodman, R. N., Conroy, S. S., Sorkin, J. D., Whitall, J., & McCombe Waller, S. (2017). Determining Levels of Upper Extremity Movement Impairment by Applying a Cluster Analysis to the Fugl-Meyer Assessment of the Upper Extremity in Chronic Stroke. Archives of Physical Medicine and Rehabilitation, 98(3), 456–462.
% fmnorm was converted to [0, 1] via x/66*100
% apply same eq to cutoffs 
% multinomial logistic regression
for i=1:size(fmnorm,1)
    if fmnorm(i) < .41
        ymulti(i)= 3 % severe
    end
    if fmnorm(i) < .62 && fmnorm(i) > .41
        ymulti(i) = 2 % moderate
    end
    if fmnorm(i) < 100 && fmnorm(i) > .62
        ymulti(i) =1 % mild
    end
end
ymulti=ymulti'

[B,dev,stats]=mnrfit(cstll, categorical(ymulti))


