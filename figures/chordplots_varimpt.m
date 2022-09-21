% Chord plots for variable importance etc
% Generate sums of top 1% of weights for plotting in R.
% Also store HEX/RBG values to go between R and MATLAB.

shen268=load('~/GIT/ENIGMA/enigma_disconnections/correlation_pairwiseSDC_shen268/shen268_counts.mat');
shen268=shen268.data;

[networks,names] = shen2yeo();

% reorder weight vector to be grouped by yeo network
[a,b]=sort(networks);
networkorder = b(:,2); %ROIs ordered by yeo 1-10
shen268=shen268+triu(shen268)'; % make full symmetric

% get top 1% of connections
Ms = sort(shen268(:),'ascend');
cutoff=Ms(round(length(shen268(:))*0.01)); % find cutoff value
shen268_thresh = shen268.*(shen268<=cutoff) % make matrix sparse

% calculate the sum of top 1% weights for each network pair.
for i=1:10
    for j=1:10
        subnetwork = shen268_thresh(networks(:,2)==i, networks(:,2)==j);
        n_nonzero = sum(subnetwork~=0, 'all');
        sum_all(i,j) = sum(subnetwork, 'all');
    end
end

% save triu - so weights arent plotted twice.
writematrix(triu(sum_all), '/Users/emilyolafson/GIT/ENIGMA/enigma_disconnections/figures/sum_all_noleftright.csv')

%% custom colormaps for yeo networks

yeolabels=({'Visual', 'Somatomotor', 'Dorsal Attention', 'Ventral Attention','Limbic', 'Frontoparietal', 'Default Mode','Subcortical Structures','Brainstem','Cerebellum'});

% from Rstudio
yeoHEX = {'#4D39B5', '#4FBB74', '#AAE8DE', '#79C7EF', '#E2C845', '#DE526A', '#D656C1',  '#F3E392', '#882255', '#6A5C84'};
yeolabels=({'Visual', 'Somatomotor', 'Dorsal Attention', 'Ventral Attention','Limbic', 'Frontoparietal', 'Default Mode','Subcortical Structures','Brainstem','Cerebellum'});
yeo_RGBcolormap = [77, 57, 181; 79, 187, 116; 123, 230, 212; 121, 199, 239;226, 200, 69; 222, 82, 106; 214, 86, 193; 243, 227, 146;136, 34, 85; 106, 92, 132];

% make colormap for viz. 10 diff colors needed.
yeo_colormap=customcolormap([0 0.11 0.22 0.33 0.44 0.55 0.66 0.77 0.88 1.0], yeoHEX)





%% sums for left and right separate
[sums, rr, rl, ll] = yeonetwork_remaps_new(shen268_thresh)

sums_lr_split1 =[triu(rr), fliplr(rl)] % flip l to r so that the inverse ordering works
sums_lr_split2 = [triu(rl), fliplr(triu(ll))];
sumsLRsplit = [sums_lr_split1;sums_lr_split2]

zero_idx=sum(sumsLRsplit,1)==0

sumsLRsplit(:,zero_idx)=-10

writematrix(triu(sumsLRsplit), '/Users/emilyolafson/GIT/ENIGMA/enigma_disconnections/figures/shen268_matrix_LR.csv')
 

