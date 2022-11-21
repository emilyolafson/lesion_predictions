% format activation weights to single matrices 


clear options
atlases = {'fs86subj'};
chaco_types = {'chacoconn'};
crossval_schemes = [1];

for atlas = 1:length(atlases)
    for chaco_type =1:length(chaco_types)
        for crossval = 1:length(crossval_schemes)

            options(1).name = 'results_path';
            options(1).value = '/Users/emilyolafson/GIT/ENIGMA/enigma_disconnections/results' ;
            options(2).name = 'atlas';
            options(2).value = atlases{atlas};
            options(3).name = 'y_var';
            options(3).value = 'normed_motor_scores';
            options(4).name = 'chaco_type';
            options(4).value = chaco_types{chaco_type};
            options(5).name ='mdl_label';
            options(5).value = 'ridge';
            options(6).name ='crossval';
            options(6).value = crossval_schemes(crossval);
            options(7).name ='n';
            options(7).value = 1;
            options(8).name ='subset';
            options(8).value = 'chronic';

            [~, ~, ~, ~, ~, ~, activations] = load_models(options);
        end
    end
end

%% Plot which regions are used in each analysis
one_cval = activations(1:5,:)
mean_one_cval = mean(one_cval(used_in_all))
logicalz = logical(one_cval)

sumlogicals = sum(logicalz)

used_in_all = sumlogicals >3
mean_one_cval = mean(one_cval.*used_in_all)

% fs lut
lut= readtable('~/Downloads/fs86_FreeSurferLUT_Readable.csv')
lut = lut(2:end, :)
lut2= readtable('~/Desktop/fs86_FreeSurferLUT.txt')

lut.Var3
lut.Var3

mean_one_cval(used_in_all)
labels = lut.Var4

labels = labels(used_in_all)
barh(mean_one_cval(used_in_all))
yticks(1:1:26)
yticklabels(labels)
set(gca, 'FontSize', 15)

gummibrain(mean_one_cval)
saveas(gcf, '/Users/emilyolafson/GIT/ENIGMA/enigma_disconnections/figures/shen268_5foldsused_haufe.png')



mat1 = [2.02, -0.3; -0.3, 0.15]
activ = [1,2]'

mat1*activ

activ_logical = logical(activations)
sum(activ_logical)

cv1 = activations
cv5 = activations

gummibrain(sum(activations,1))
fs86 = sum(activations,1)

%% 
% Chord plots for variable importance etc
% Generate sums of top 1% of weights for plotting in R.
% Also store HEX/RBG values to go between R and MATLAB.

% Find connections used in > 50% of the cross-validation folds.
logic = logical(activations);
sumlogicals = squeeze(sum(logic,1));
used_in_half = sumlogicals>75;
mean_haufe = squeeze(mean(activations,1)).*used_in_half;
fs86=mean_haufe;

%shen268=load('~/GIT/ENIGMA/enigma_disconnections/correlation_pairwiseSDC_shen268/shen268_counts.mat');
%shen268=shen268.data;

%[networks,names] = shen2yeo();

%% 
[networks,names] = fs2yeo();

% reorder weight vector to be grouped by yeo network
[a,b]=sort(networks);
networkorder = b(:,2); %ROIs ordered by yeo 
fs86=fs86+triu(fs86)'; % make full symmetric
imagesc(fs86)
xticks(1:1:86)
xticklabels(networks(:,2))
yticks(1:1:86)
yticklabels(networks(:,2))

% get top 1% of connections
%Ms = sort(fs86(:),'ascend');
%cutoff=Ms(round(length(fs86(:))*0.01)); % find cutoff value
%fs86_thresh = fs86.*(fs86<=cutoff) % make matrix sparse

% calculate the mean recurring weights for each network pair.
clear sum_all;
for i=1:8
    for j=1:8
        subnetwork = fs86(networks(:,2)==i, networks(:,2)==j);
        n_nonzero = sum(subnetwork~=0, 'all');
        subnetwork_includedinhalf = subnetwork(subnetwork~=0);
        sum_all(i,j) = mean(subnetwork_includedinhalf, 'all');
    end
end

imagesc(sum_all)

% save triu - so weights arent plotted twice.
writematrix(triu(sum_all), '/Users/emilyolafson/GIT/ENIGMA/enigma_disconnections/figures/sum_all_noleftright_FS86.csv')

%% custom colormaps for yeo networks

shen268_yeolabels=({'Visual', 'Somatomotor', 'Dorsal Attention', 'Ventral Attention','Limbic', 'Frontoparietal', 'Default Mode','Subcortical Structures','Brainstem','Cerebellum'});
fs86yeo_labels = ({'Visual', 'Somatomotor', 'Dorsal Attention', 'Ventral Attention', 'Limbic','Frontoparietal', 'Default Mode', 'Subcortical Structures', 'Cerebellum'});

% from Rstudio
yeoHEX_shen = {'#4D39B5', '#4FBB74', '#AAE8DE', '#79C7EF', '#E2C845', '#DE526A', '#D656C1',  '#F3E392', '#882255', '#6A5C84'};
yeoHEX_fs86 = {'#4D39B5', '#4FBB74', '#AAE8DE', '#79C7EF', '#E2C845', '#DE526A', '#D656C1',  '#F3E392', '#6A5C84'};

yeolabels=({'Visual', 'Somatomotor', 'Dorsal Attention', 'Ventral Attention','Limbic', 'Frontoparietal', 'Default Mode','Subcortical Structures','Brainstem','Cerebellum'});
yeo_RGBcolormap = [77, 57, 181; 79, 187, 116; 123, 230, 212; 121, 199, 239;226, 200, 69; 222, 82, 106; 214, 86, 193; 243, 227, 146;136, 34, 85; 106, 92, 132];

% make colormap for viz. 10 diff colors needed.
yeoshen_colormap=customcolormap([0 0.11 0.22 0.33 0.44 0.55 0.66 0.77 0.88 1.0], yeoHEX_shen)
yeofs86_colormap=customcolormap([0 0.11 0.22 0.33 0.44 0.55 0.66 0.77 1.0], yeoHEX_fs86)


%% sums for left and right separate
[sums, rr, rl, ll] = yeonetwork_remaps_fs86(fs86)

sums_lr_split1 =[triu(rr), fliplr(rl)] % flip l to r so that the inverse ordering works
sums_lr_split2 = [triu(rl), fliplr(triu(ll))];
sumsLRsplit = [sums_lr_split1;sums_lr_split2]

zero_idx=sum(sumsLRsplit,1)==0

sumsLRsplit(:,zero_idx)=-1

writematrix(triu(sumsLRsplit), '/Users/emilyolafson/GIT/ENIGMA/enigma_disconnections/figures/fs86_1_chacoconn_matrix_LR.csv')



 