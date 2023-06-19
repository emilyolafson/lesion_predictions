
list = readtable('/Users/emilyolafson/GIT/ENIGMA/enigma_disconnections/lefthem_lesion_arraysublist.txt')
list = table2array(list)
lesion_table=[]

for i =1:length(list)
    lesion = read_avw(['/Users/emilyolafson/GIT/ENIGMA/data/lesionmasks/all_lesionmasks_2009tov6_usingSym_rename_2mm/', strcat(list{i}, '_2mm')]);
    lesion_table = cat(4, lesion_table, lesion);
end

mni = read_avw('/usr/local/fsl/data/standard/MNI152_T1_2mm_brain.nii.gz');
mnimask = mnfi > 0;

lesionsub_mask=zeros(1067, 228453);
uniques=[]
for i = 1:1067
    lesionsub = logical(lesion_table(:,:,:,i));
    lesionsub_mask(i,:) = logical(lesionsub(mnimask));
    uniques = [uniques; unique(lesionsub)];
end
unique(uniques)

corrs_all=[]
for batch = 1:100:size(lesionsub_mask,2)
    A = triu(corr(lesionsub_mask(:,batch:batch+100)),1);
    [~, col]=find(A==1)
    At = A';
    m = tril(true(size(A)));
    v = At(m);
    corrs_all = [corrs_all;v];
end


corrs_all(isnan(corrs_all))=0;

histogram(corrs_all)
ylabel('Number of voxel-pairs')
xlabel('Pearson correlation')
set(gca, 'FontSize', 20)
saveas(gcf, '/Users/emilyolafson/GIT/ENIGMA/enigma_disconnections/figures/lesiondef_inf/pairwise_corrs.png')


writematrix(lesion_table, '/Users/emilyolafson/GIT/ENIGMA/enigma_disconnections/figures/lesiondef_inf/lesion_alltable.txt')


% 

lesion_table_log=logical(lesion_table);

lesion_table_vec = reshape(lesion_table_log, [91*109*91, size(lesion_table,4)]);
sumz1 = sum(sum(lesion_table_vec,2)>=1)
sumz3 = sum(sum(lesion_table_vec,2)>=5)
sumz3 = sum(sum(lesion_table_vec,2)==0)
sumz3 = sum(sum(lesion_table_vec,2)~=0)

load('/Users/emilyolafson/GIT/ENIGMA/enigma_disconnections/figures/lesiondef_inf/workspace.mat')

% threshold to get rid of every voxel that is damaged in less than 5
% subjects.
lesion_table_vec_thr5 = lesion_table_vec.*(sum(lesion_table_vec,2)>=5);

%lesion_table_vec_thr5 = logical(lesion_table_vec_thr5);
mnimask = reshape(mni > 0, [91*109*91, 1]);

build_ones_matrix = zeros(902629,1);

[gr5subjects_voxels, b,c]=find(sum(lesion_table_vec_thr5,2)>0);

for x = 1:size(lesion_table_vec_thr5,1)
    
    if mnimask(x)==0
        degree_x(x)=0;
        degree_x1(x)=0;
    elseif sum(lesion_table_vec_thr5(x,:), 2)==0
        degree_x(x)=0;
        degree_x1(x) = 0;
    else

        xycor=zeros(length(gr5subjects_voxels),1);
        allnorms=[];
 
        for y = 1:length(gr5subjects_voxels)
           
           xycor = corr(lesion_table_vec_thr5(x, :)'-lesion_table_vec_thr5(gr5subjects_voxels(y), :)');
           if xycor > 0
               allnorms = [allnorms, xycor];
           end
        end

        
        nodestr(x) = sum(allcors>0, 'omitnan');
        degree_x(x) = sum(allcors>0.7);
        degree_x1(x) = sum(allcors==1);
        writematrix(nodestr(x), ['/Users/emilyolafson/GIT/ENIGMA/enigma_disconnections/figures/lesiondef_inf/str_node_', num2str(x) , '.txt'])

        writematrix(degree_x(x), ['/Users/emilyolafson/GIT/ENIGMA/enigma_disconnections/figures/lesiondef_inf/degree_node_', num2str(x) , '.txt'])
        writematrix(degree_x1(x), ['/Users/emilyolafson/GIT/ENIGMA/enigma_disconnections/figures/lesiondef_inf/degree_node1_', num2str(x) , '.txt'])
        
    end
end


