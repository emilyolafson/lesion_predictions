% lesion volume of ENIGMA stroke lesions

allvol=dir('~/GIT/ENIGMA/data/lesionmasks/1mm/*lesionvol.txt')

for i=1:size(allvol,1)
    tmp=load(['~/GIT/ENIGMA/data/lesionmasks/1mm/', allvol(i).name]);
    lesionvolume(i)=tmp(1);
end



histogram(lesionvolume)

[n, edges]=histcounts(lesionvolume)
tbl=[n;edges(1:end-1)]

imagesctxt(tbl)
