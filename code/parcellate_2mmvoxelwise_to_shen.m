% parcellate voxelwise chaco scores

atlas = read_avw('~/GIT/ENIGMA/data/ChaCo/voxelwise/shen_2mm_268_parcellation.nii');
atlas=reshape(atlas, [902629 1]);

fid = fopen('~/GIT/ENIGMA/data/ChaCo/voxelwise/subs_april4.txt');


for i=1:893
    line1 = fgetl(fid)
    img=read_avw(strcat('~/GIT/ENIGMA/data/ChaCo/voxelwise/', line1));
    img=reshape(img, [902629 1]);
    for t=1:size(unique(atlas))-1 %1:268 (zeros)
        roi=img(atlas==t);
        avg(t)=sum(roi,1)./size(roi,1);
        numvoxels(t)=size(roi,2);
    end
    writematrix(avg, strcat('~/GIT/ENIGMA/data/ChaCo/shen268_parcellated/', line1, '.txt'));
end

