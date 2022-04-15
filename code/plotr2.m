
for i=0:99
    allvar(i+1)=mean(load(sprintf('~/GIT/ENIGMA/results/correlation_regionalSDC_fs86/correlationp%i_SC_all_explvar.txt', i)));
end

for i=0:99
    allvar2(i+1)=mean(load(sprintf('~/GIT/ENIGMA/results/correlation_regionalSDC_shen268/correlationp%i_SC_all_explvar.txt', i)));
end

for i=0:99
    allvar3(i+1)=mean(load(sprintf('~/GIT/ENIGMA/results/correlation_pairwiseSDC_fs86/correlationp%i_SC_all_explvar.txt', i)));
end

cstll=load('~/GIT/ENIGMA/results/cst_ll/CSTLL_all_explvar.txt');
cstll=reshape(cstll, [100 5])
cstll=mean(cstll,2)

violinplot([cstll'; allvar; allvar2; allvar3]')
xticklabels({'CST-LL', 'Regional fs86', 'Regional shen268', 'Pairwise fs86'})
ylabel('R^2')
set(gca, 'FontSize', 14)

saveas(gcf, '~/GIT/ENIGMA/results/correlation_sc_chronic_cstll_regionalSDC_pairwiseSDC.png')
writematrix(mean(allvar_sc_correlation_chr2,2), '~/GIT/cognition_nemo/results/correlation_sc_chronic_cstll_regionalSDC_pairwiseSDC.txt')
