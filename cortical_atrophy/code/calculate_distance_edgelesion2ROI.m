% calculate the distance between the centroid of ROIs and the closest point
% of the lesion

subjlist = dir('/Users/emilyolafson/GIT/ENIGMA/data/lesionmasks/all_lesionmasks_2009tov6_usingSym/');

% get col names (fs region names)
lut = readtable('/Users/emilyolafson/GIT/ENIGMA/enigma_disconnections/cortical_atrophy/data/FREESURFER_ENIGMA/aseg+aparc_LUT.txt');
regionnames = lut.Var2;
regionidx = lut.Var1;
fs_eg = read_avw('/Users/emilyolafson/GIT/ENIGMA/enigma_disconnections/cortical_atrophy/data/FREESURFER_ENIGMA/voxelwise_aseg_aparc/sub-r009s055_aseg_aparc_ro_toMNI.nii.gz');
uniqueidx = unique(fs_eg);
[~,~,index_b] = intersect(uniqueidx,regionidx,'stable');
fs_names = regionnames(index_b);

% n subs w fs parc
fsdir = dir('/Users/emilyolafson/GIT/ENIGMA/enigma_disconnections/cortical_atrophy/data/FREESURFER_ENIGMA/voxelwise_aseg_aparc/*_aseg_aparc_ro_toMNI.nii.gz');
nsubs = length(fsdir)-2;

% initialize empty table stuff to store distances
distance_table = array2table(zeros(nsubs, 114));

distance_table=zeros(nsubs, 114);
subject_table=[];

for rowcount = 1:length(subjlist)-2 % ignore . and ..
    % get subject id
    subjname = subjlist(rowcount+2).name;
    subjname = subjname(1:12);
    
    if ~isfile(sprintf('/Users/emilyolafson/GIT/ENIGMA/enigma_disconnections/cortical_atrophy/data/FREESURFER_ENIGMA/distance_roi_lesion/%s_distance_FS_closestpt.txt', subjname))
        sprintf('calculating distance from lesion to ROIs for subject %s\n', subjname)
        % load lesion (MNI152 v6)
        lesionpath = strcat(subjlist(rowcount+2).folder,'/',subjlist(rowcount+2).name);
        lesion = read_avw(lesionpath);

        % lesion must be in dimensions mxn where m is # of unique points and n is
        % 3 (dimensions)
        [x,y,z]=ind2sub(size(lesion),find(lesion == 1));
        lesion_coords =[x, y, z];

        % load freesurfer parc (MNI152 v6)
        try
            fs = read_avw(sprintf('/Users/emilyolafson/GIT/ENIGMA/enigma_disconnections/cortical_atrophy/data/FREESURFER_ENIGMA/voxelwise_aseg_aparc/%s_aseg_aparc_ro_toMNI.nii.gz',subjname));
        catch
            fprintf('\n subject %s has no freesurfer file', subjname);
            continue;
        end

        % calculate centroids of each ROI
        uniqueROIs = unique(fs);
        centroid=[];
        for roi = 1:length(uniqueROIs)
            binary_ROIvol = fs==uniqueROIs(roi);
            fs_id = fs_names(roi);
            [RL, PA, IS] = ndgrid(1:size(binary_ROIvol, 1), 1:size(binary_ROIvol, 2),1:size(binary_ROIvol, 3));
            centroid(roi,:) = mean([RL(logical(binary_ROIvol)), PA(logical(binary_ROIvol)), IS(logical(binary_ROIvol))]);
            %save_avw(binary_ROIvol, sprintf('/Users/emilyolafson/GIT/ENIGMA/enigma_disconnections/cortical_atrophy/data/FREESURFER_ENIGMA/asegaparc_%i.nii.gz', uniqueROIs(roi)), 'f', [1 1 1])
        end

        % calculate distance to closest point to lesion for each ROI
        %(P,PQ) 
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




