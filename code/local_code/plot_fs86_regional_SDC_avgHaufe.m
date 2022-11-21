
load('/Users/emilyolafson/GIT/dynamic-brainstates/code/atlasblobs_saved.mat')

addpath('~/GIT/dynamic-brainstates/code/gummibrain')
addpath('~/GIT/ENIGMA/code/github_repo') % cmocean cmaps

%choose atlas
%options for whichatlas: aal, cc200, cc400, ez, ho, tt, fs86
whichatlas={'fs86'}
clc;

data=load('~/GIT/ENIGMA/results/correlation_regionalSDC_fs86/avgHaufe_fs86_regionalSDC_correlation_chronic.txt')

cmap=cmocean('deep')

data_min=min(data)
data_max=max(data)

img=display_atlas_blobs(data,atlasblobs_saved, ...
    'atlasname',whichatlas,...
    'render',true,...
    'backgroundimage',false,...
    'crop',true,...
    'colormap',cmap);

figure;
imshow(img);
%to show colorbar:
c=colorbar('SouthOutside', 'fontsize', 16);
c.Label.String='';
caxis([data_min data_max]);

set(gca,'colormap',cmap);
title(sprintf('Avg Haufe-transformed Activation Weights Regional SDC April 13, FS86'), 'fontsize',12);
saveas(gcf,'~/GIT/ENIGMA/results/correlation_regionalSDC_fs86/avghaufe_regionalSDC_fs86.png')

