
list = readtable('/Users/emilyolafson/GIT/ENIGMA/enigma_disconnections/lefthem_lesion_arraysublist.txt')
list = table2array(list)

for i =222:length(list)
    lesion = read_avw(['/Users/emilyolafson/GIT/ENIGMA/data/lesionmasks/all_lesionmasks_2009tov6_usingSym_rename/', list{i}]);
    lesion_table = cat(4, lesion_table, lesion);
end

writematrix(lesion_table, '/Users/emilyolafson/GIT/ENIGMA/enigma_disconnections/lefthem_lesion_alltable.txt')