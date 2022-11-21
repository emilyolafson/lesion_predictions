allresults=dir('~/GIT/ENIGMA/results/correlation_sc_regional/*all_explvar.txt')

for i=1:size(allresults,1)
    perm=load(strcat('~/GIT/ENIGMA/results/correlation_sc_regional/', allresults(i).name))
    permavg=mean(perm)
    allpermutations(i)=permavg
end

violinplot(allpermutations)
ylim([0 0.3])
ylabel('R^2')
xticklabels('Regional SDC')
set(gca, 'FontSize', 15)
text(0.9,0.27, num2str(round(mean(allpermutations),3)), 'FontSize', 14)

