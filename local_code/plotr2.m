% load scores for each model

clear options
atlases = {'fs86subj','shen268'};
chaco_types = {'chacovol'};
crossval_schemes = [1, 5];

for atlas = 1:length(atlases)
    for chaco_type =1:length(chaco_types)
        for crossval = 1:length(crossval_schemes)

            options(1).name = 'results_path';
            options(1).value = '/Users/emilyolafson/GIT/ENIGMA/enigma_disconnections/results' ;
            options(2).name = 'atlas';
            options(2).value = atlases{atlas};
            options(3).name = 'y_var';
            options(3).value = 'normed_motor_scores';
            options(4).name = 'chaco_type';
            options(4).value = chaco_types{chaco_type};
            options(5).name ='mdl_label';
            options(5).value = 'ridge';
            options(6).name ='crossval';
            options(6).value = crossval_schemes(crossval);
            options(7).name ='n';
            options(7).value = 1;
            options(8).name ='subset';
            options(8).value = 'chronic';

            [r2scores, correl, nullr2scores, nullcorrel, ~, ~,~] = load_models(options);
        
            % calcualte the mean performance of all null distributions
            null_perm=29
            clear meanperm
           % for n=1:null_perm
           %    meanperm(n,:)=mean(nullr2scores(:,n,:),3)';
           % end

            % calcualte the mean performance of all null distributions
            null_perm=29
            %for n=1:null_perm
            %    meanperm_corr(n,:)=mean(nullcorrel(:,n,:),3)';
            %end

            writematrix(r2scores, sprintf('r2scores_%s_%s_%s.txt', atlas,chaco_type, num2str(crossval)));
            writematrix(correl, sprintf('corrs_%s_%s_%s.txt', atlas,chaco_type, num2str(crossval)));
           % writematrix(meanperm, sprintf('null_r2scores_%s_%s_%s.txt', atlas,chaco_type, num2str(crossval)));
           % writematrix(meanperm_corr, sprintf('null_corrs_%s_%s_%s.txt', atlas,chaco_type, num2str(crossval)));
        end
    end
end

median(meanperm_corr,2)

figure('Position', [0 0 1000 500])
tiledlayout(1,2,'padding','none')
nexttile;
violinplot(rscores)
xticklabels({'KFold', 'GroupKFold', 'Shuffle'})
title('R^2 scores')
ylim([-1 1])
nexttile;

violinplot(corrs, grouping_var)
xticklabels({'KFold', 'GroupKFold', 'Shuffle'})
title('Correlations')
ylim([-1 1])

sgtitle(sprintf('%s %s', atlas, chaco_type))
saveas(gcf, sprintf('%s/%s_%s_plots.png',results_path, atlas, chaco_type))



