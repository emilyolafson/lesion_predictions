
% load labels for fs86
map=load('fs86_to_yeo_map.csv')
labels=readtable('fs86_FreeSurferLUT_Readable.csv')
names=table2array(labels(2:end,2))
names2=table2array(labels(2:end,3))
names3=table2array(labels(2:end,4))

[idx, ord]= sort(map)

map_sorted=idx

table=load('~/GIT/ENIGMA/results/correlation_pairwiseSDC_fs86/fs86_activ_weights.txt')
table_sorted=table(ord, ord)
names_sorted=names3(ord)

jet
colormap=zeros(86, 3)

colormap(map_sorted==1, :)=repmat( [ 0, 0.2, 0.5156], sum(map_sorted==1), 1)
colormap(map_sorted==2, :)=repmat( [0, 0.01, 1.0000], sum(map_sorted==2), 1)
colormap(map_sorted==3, :)=repmat( [0, 1, 1.0000], sum(map_sorted==3), 1)
colormap(map_sorted==4, :)=repmat( [1, 0.7812, 0], sum(map_sorted==4), 1)
colormap(map_sorted==5, :)=repmat( [1, 0, 0], sum(map_sorted==5), 1)
colormap(map_sorted==6, :)=repmat( [0.5, 0, 0], sum(map_sorted==6), 1)
colormap(map_sorted==7, :)=repmat( [0.5,1, 0.5], sum(map_sorted==7), 1)
colormap(map_sorted==8, :)=repmat( [1, 0.5, 1.0000], sum(map_sorted==8), 1)


t=fliplr(sort(table_sorted))
t=t(1:3655)
t(35)

table_sorted(table_sorted<60000)=0
circularGraph( table_sorted, 'Colormap', colormap, 'label', names_sorted)


