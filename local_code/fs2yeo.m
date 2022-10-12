function [yeonetworks,names] = fs2yeo()
    % returns: 
    %
    % yeonetworks - 86x2 array, 1st column is ROI # (1-86), 2nd
    % column is Yeo network assignment for that ROI
    % names - 86 cell array of yeo network names

    % load shen and fs network assignments
    mapping = readtable('~/GIT/ENIGMA/fs86_to_yeo_map.csv');
    networknums=mapping.Var1;
    
    nums = [1:1:86];
    names = {'Visual', 'Somatomotor', 'Dorsal Attention', 'Ventral Attention', 'Limbic','Frontoparietal', 'Default Mode', 'Subcortical Structures', 'Cerebellum'};
    yeonetworks = [nums',networknums];
