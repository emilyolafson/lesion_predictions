% calculate the distance between the centroid of ROIs and the closest point
% of the lesion

subjlist = dir('/Users/emilyolafson/GIT/ENIGMA/data/lesionmasks/all_lesionmasks_2009tov6_usingSym/');

% get col names (fs region names)
lut = readtable('/Users/emilyolafson/GIT/ENIGMA/enigma_disconnections/cortical_atrophy/data/FREESURFER_ENIGMA/aseg+aparc_LUT.txt');
regionnames = lut.Var2;
regionidx = lut.Var1;

% n subs w fs parc
fsdir = dir('/Users/emilyolafson/GIT/ENIGMA/enigma_disconnections/cortical_atrophy/data/FREESURFER_ENIGMA/voxelwise_aseg_aparc/*_aseg_aparc_ro_toMNI.nii.gz');
nsubs = length(fsdir)-2;

% dont pop up figures
set(0,'DefaultFigureVisible','off')

for rowcount = 1:length(subjlist)-2 % ignore . and ..
    % get subject id
    subjname = subjlist(rowcount+2).name;
    subjname = subjname(1:12);
    if ~isfile(sprintf('/Users/emilyolafson/GIT/ENIGMA/enigma_disconnections/cortical_atrophy/data/FREESURFER_ENIGMA/distance_roi_lesion/%s_distance_FS_closestpt.txt', subjname))
        sprintf('calculating distance from lesion to ROIs for subject %s\n', subjname)
        % load lesion (MNI152 v6)
        lesionpath = strcat(subjlist(rowcount+2).folder,'/',subjlist(rowcount+2).name);
        lesion = read_avw(lesionpath);
        lesion = logical(lesion);

        % lesion must be in dimensions mxn where m is # of unique points and n is
        % n dimensions (rows of coordinates)
        [x,y,z]=ind2sub(size(lesion),find(lesion == 1));
        lesion_coords =[x, y, z];

        % load freesurfer parc (MNI152 v6)
        try
            fs = read_avw(sprintf('/Users/emilyolafson/GIT/ENIGMA/enigma_disconnections/cortical_atrophy/data/FREESURFER_ENIGMA/voxelwise_aseg_aparc/%s_aseg_aparc_ro_toMNI.nii.gz',subjname));
        catch
            fprintf('\n subject %s has no freesurfer file', subjname);
            continue;
        end
        
        uniqueROIs = unique(fs);
        [~,~,index_b] = intersect(uniqueROIs,regionidx,'stable');
        fs_names = regionnames(index_b);
        
        distance_table=zeros(nsubs, length(uniqueROIs));

        centroid=[];
        for roi = 1:length(uniqueROIs)
            binary_ROIvol = fs==uniqueROIs(roi);
            [RL, PA, IS] = ndgrid(1:size(binary_ROIvol, 1), 1:size(binary_ROIvol, 2),1:size(binary_ROIvol, 3));
            centroid(roi,:) = mean([RL(logical(binary_ROIvol)), PA(logical(binary_ROIvol)), IS(logical(binary_ROIvol))]);
        end

        % calculate distance to closest point to lesion for each ROI
        dist=[]
        for roi=1:length(uniqueROIs)
            rounded_centroid = round(centroid(roi,:));
            [k,dist(roi)] = dsearchn(lesion_coords,rounded_centroid); %also returns the distance from each point in P to the corresponding query point in PQ.
        end

        % save distance/subjnames
        distance_table=array2table(dist);
        distance_table.Properties.VariableNames = fs_names;

        writetable(distance_table, sprintf('/Users/emilyolafson/GIT/ENIGMA/enigma_disconnections/cortical_atrophy/data/FREESURFER_ENIGMA/distance_roi_lesion/%s_distance_FS_closestpt.txt', subjname));
    else
        sprintf('distance table for %s already exists!\n', subjname);
    end
end


%make figures of distances

for rowcount = 1:length(subjlist)-2 % ignore . and ..
    subjname = subjlist(rowcount+2).name;
    subjname = subjname(1:12);
    try
        tbl = readtable(sprintf('/Users/emilyolafson/GIT/ENIGMA/enigma_disconnections/cortical_atrophy/data/FREESURFER_ENIGMA/distance_roi_lesion/%s_distance_FS_closestpt.txt', subjname));

        lh1 = tbl(:,[6:10, 14, 15, 17, 18]);
        rh1 = tbl(:,[25:30, 31, 32, 33]);
        lh2 = tbl(:, [46:79]);
        rh2 = tbl(:, [81:114]);

        gummibrain(table2array([lh1, rh1, lh2, rh2]))
        saveas(gcf, sprintf('/Users/emilyolafson/GIT/ENIGMA/enigma_disconnections/cortical_atrophy/data/FREESURFER_ENIGMA/distance_roi_lesion/%s_distance_gummi.png', subjname));
        close all;
        catch
        disp('oops')
    end
end


make figures of ct?
tbl = readtable('/Users/emilyolafson/GIT/ENIGMA/enigma_disconnections/cortical_atrophy/data/FREESURFER_ENIGMA/enigma_combat_thickness_zscore_NANs.csv');
subs=tbl.Subject;

for rowcount = 1:length(subs)
    subjname = subs{rowcount};

    values(1:18)=NaN;
    zscores = tbl(rowcount, 3:end);
    values(19:86)=table2array(zscores);

    gummibrain(values')
    saveas(gcf, sprintf('/Users/emilyolafson/GIT/ENIGMA/enigma_disconnections/cortical_atrophy/data/FREESURFER_ENIGMA/distance_roi_lesion/%s_zCT_gummi.png', subjname));
    close all;

    disp('oops')
end


