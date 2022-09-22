% make figures about pairwise model results

%% 1. violin/boxplots of R^2 
for i=1:100
    r2_shen(i)=mean(load(sprintf('~/GIT/ENIGMA/enigma_disconnections/correlation_pairwiseSDC_shen268/correlationp%i_SC_all_explvar.txt', i-1)));
end

for i=1:100
    r2_fs(i)=mean(load(sprintf('~/GIT/ENIGMA/enigma_disconnections/correlation_pairwiseSDC_fs86/correlationp%i_SC_all_explvar.txt', i-1)));
end

violinplot([r2_shen; r2_fs], [zeros(100,1);ones(100,1)])
ylim([0, 0.5])


%% 2. number of features/best lambdas

tiledlayout(2,1,'padding','none')

% regional 
lambda_shen=[]
for i=1:100
    load(sprintf('~/GIT/ENIGMA/enigma_disconnections/correlation_regionalSDC_shen268/correlationp%i_SC_bestalphas.mat', i-1));
    lambda_shen=vertcat(lambda_shen, data);
end

feats_shen=[]
for i=1:100
    load(sprintf('~/GIT/ENIGMA/enigma_disconnections/correlation_regionalSDC_shen268/correlationp%i_SC_features.mat', i-1));
    feats_shen=vertcat(feats_shen, data);
    sizefeats_shen=cellfun(@length, feats_shen);
end

lambdas_fs=[]
for i=1:100
    load(sprintf('~/GIT/ENIGMA/enigma_disconnections/correlation_regionalSDC_fs86/correlationp%i_SC_bestalphas.mat', i-1));
    lambdas_fs=vertcat(lambdas_fs, data);
end

feats_fs=[]
for i=1:100
    load(sprintf('~/GIT/ENIGMA/enigma_disconnections/correlation_regionalSDC_fs86/correlationp%i_SC_features.mat', i-1));
    feats_fs=vertcat(feats_fs, data);
    sizefeats_fs=cellfun(@length, feats_fs);
end

% regional fs86
nexttile;
X=reshape(sizefeats_fs, [], 1);
Y=log10(reshape(lambdas_fs,[],1));
XY = [X,Y];
% Find unique rows and corresponding indices
[uniqueAB, ~, n] = unique(XY, 'rows');
% Find number of occurrences
nHist = hist(n, unique(n));
mx = max(nHist);
% Create colors for each number of occurrence
colors = brewermap(mx,'Spectral');
colormap(colors);
% Construct a color matrix
cMatrix = colors(nHist, :);
scatter(uniqueAB(:, 1), uniqueAB(:, 2), 25, cMatrix, 'filled');
colorbar
caxis([1 mx])
xlabel('log-transformed number of features')
ylabel('log-transformed lambda values')
title('Fs-86')
set(gca, 'FontSize', 15)
%saveas(gcf, '/Users/emilyolafson/GIT/ENIGMA/figures/lambdas_nFeats_fs86.png')

% regional shen268
nexttile;
X=reshape(sizefeats_shen, [], 1);
Y=log10(reshape(lambda_shen,[],1));
XY = [X,Y];
% Find unique rows and corresponding indices
[uniqueAB, ~, n] = unique(XY, 'rows');
% Find number of occurrences
nHist = hist(n, unique(n));
mx = max(nHist);
% Create colors for each number of occurrence
colors = brewermap(mx,'Spectral');
colormap(colors);
% Construct a color matrix
cMatrix = colors(nHist, :);
scatter(uniqueAB(:, 1), uniqueAB(:, 2), 25, cMatrix, 'filled');
colorbar
caxis([1 mx])
xlabel('log-transformed number of features')
ylabel('log-transformed lambda values')
title('Shen-268')
set(gca, 'FontSize', 15)
%saveas(gcf, '/Users/emilyolafson/GIT/ENIGMA/figures/lambdas_nFeats_shen268.png')



tiledlayout(1,2,'padding','none')
%pairwise
lambda_shen=[]
for i=1:100
    load(sprintf('~/GIT/ENIGMA/enigma_disconnections/correlation_pairwiseSDC_shen268/correlationp%i_SC_bestalphas.mat', i-1));
    lambda_shen=vertcat(lambda_shen, data);
end

feats_shen=[]
for i=1:100
    load(sprintf('~/GIT/ENIGMA/enigma_disconnections/correlation_pairwiseSDC_shen268/correlationp%i_SC_features.mat', i-1));
    feats_shen=vertcat(feats_shen, data);
    sizefeats_shen=cellfun(@length, feats_shen);
end

lambdas_fs=[]
for i=1:100
    load(sprintf('~/GIT/ENIGMA/enigma_disconnections/correlation_pairwiseSDC_fs86/correlationp%i_SC_bestalphas.mat', i-1));
    lambdas_fs=vertcat(lambdas_fs, data);
end

feats_fs=[]
for i=1:100
    load(sprintf('~/GIT/ENIGMA/enigma_disconnections/correlation_pairwiseSDC_fs86/correlationp%i_SC_features.mat', i-1));
    feats_fs=vertcat(feats_fs, data);
    sizefeats_fs=cellfun(@length, feats_fs);
end
nexttile;

%pairwise fs86
X=log2(reshape(sizefeats_fs, [], 1));
Y=log10(reshape(lambdas_fs,[],1));
XY = [X,Y];
% Find unique rows and corresponding indices
[uniqueAB, ~, n] = unique(XY, 'rows');
% Find number of occurrences
nHist = hist(n, unique(n));
mx = max(nHist);
% Create colors for each number of occurrence
colors = parula(mx);
colormap(colors);
% Construct a color matrix
cMatrix = colors(nHist, :);
scatter(uniqueAB(:, 1), uniqueAB(:, 2), 25, cMatrix, 'filled');
colorbar
caxis([1 mx])
xlabel('log-transformed number of features')
ylabel('log-transformed lambda values')
title('Fs-86')
set(gca, 'FontSize', 15)
%saveas(gcf, '/Users/emilyolafson/GIT/ENIGMA/figures/lambdas_nFeats_fs86.png')

%pairwise 268
nexttile;
X=log2(reshape(sizefeats_shen, [], 1));
Y=log10(reshape(lambda_shen,[],1));
XY = [X,Y];
% Find unique rows and corresponding indices
[uniqueAB, ~, n] = unique(XY, 'rows');
% Find number of occurrences
nHist = hist(n, unique(n));
mx = max(nHist);
% Create colors for each number of occurrence
colors = parula(mx);
colormap(colors);
% Construct a color matrix
cMatrix = colors(nHist, :);
scatter(uniqueAB(:, 1), uniqueAB(:, 2), 25, cMatrix, 'filled');
colorbar
caxis([1 mx])
xlabel('log-transformed number of features')
ylabel('log-transformed lambda values')
title('Shen-268')
set(gca, 'FontSize', 15)
%saveas(gcf, '/Users/emilyolafson/GIT/ENIGMA/figures/lambdas_nFeats_shen268.png')

