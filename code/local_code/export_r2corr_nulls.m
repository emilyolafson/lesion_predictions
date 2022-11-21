% load scores for each model

clear options
atlases = {'shen268', 'fs86subj'};
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

            [r2scores, correl, nullr2scores, nullcorrel] = load_null_models(options);
        
            % calcualte the mean performance of all null distributions
            %null_perm=29
            clear meanperm
           % for n=1:null_perm
           %     meanperm(n,:)=mean(nullr2scores(:,n,:),3)';
           % end

            % calcualte the mean performance of all null distributions
           % null_perm=29
           % for n=1:null_perm
           %     meanperm_corr(n,:)=mean(nullcorrel(:,n,:),3)';
           % end

            writematrix(r2scores, sprintf('%s/r2scores_%s_%s_%s.txt', options(1).value,options(2).value,options(4).value, num2str(options(6).value)));
            writematrix(correl, sprintf('%s/corrs_%s_%s_%s.txt', options(1).value,options(2).value,options(4).value, num2str(options(6).value)));
            %writematrix(meanperm, sprintf('%s/null_r2scores_%s_%s_%s.txt', options(1).value,options(2).value,options(4).value, num2str(options(6).value)));
            %writematrix(meanperm_corr, sprintf('%s/null_corrs_%s_%s_%s.txt', options(1).value,options(2).value,options(4).value, num2str(options(6).value)));
        end
    end
end


% lesion load - all cst
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

            [r2scores, correl, nullr2scores, nullcorrel] = load_models_lesionload(options);
        
            % calcualte the mean performance of all null distributions
            %null_perm=29
            clear meanperm
           % for n=1:null_perm
           %     meanperm(n,:)=mean(nullr2scores(:,n,:),3)';
           % end

            % calcualte the mean performance of all null distributions
           % null_perm=29
           % for n=1:null_perm
           %     meanperm_corr(n,:)=mean(nullcorrel(:,n,:),3)';
           % end

            writematrix(r2scores, sprintf('%s/r2scores_%s_%s_%s_lesionload.txt', options(1).value,options(2).value,options(4).value, num2str(options(6).value)));
            writematrix(correl, sprintf('%s/corrs_%s_%s_%s_lesionload.txt', options(1).value,options(2).value,options(4).value, num2str(options(6).value)));
            %writematrix(meanperm, sprintf('%s/null_r2scores_%s_%s_%s.txt', options(1).value,options(2).value,options(4).value, num2str(options(6).value)));
            %writematrix(meanperm_corr, sprintf('%s/null_corrs_%s_%s_%s.txt', options(1).value,options(2).value,options(4).value, num2str(options(6).value)));
        end
    end
end



% lesion load - m1 only
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
            options(5).value = 'ensemble_reg';
            options(6).name ='crossval';
            options(6).value = crossval_schemes(crossval);
            options(7).name ='n';
            options(7).value = 1;
            options(8).name ='subset';
            options(8).value = 'chronic';

            [r2scores, correl, nullr2scores, nullcorrel] = load_models_lesionload_cst(options);
        
            % calcualte the mean performance of all null distributions
            %null_perm=29
            clear meanperm
           % for n=1:null_perm
           %     meanperm(n,:)=mean(nullr2scores(:,n,:),3)';
           % end

            % calcualte the mean performance of all null distributions
           % null_perm=29
           % for n=1:null_perm
           %     meanperm_corr(n,:)=mean(nullcorrel(:,n,:),3)';
           % end

            writematrix(r2scores, sprintf('%s/r2scores_%s_%s_%s_lesionload_cst.txt', options(1).value,options(2).value,options(4).value, num2str(options(6).value)));
            writematrix(correl, sprintf('%s/corrs_%s_%s_%s_lesionload_cst.txt', options(1).value,options(2).value,options(4).value, num2str(options(6).value)));
            %writematrix(meanperm, sprintf('%s/null_r2scores_%s_%s_%s.txt', options(1).value,options(2).value,options(4).value, num2str(options(6).value)));
            %writematrix(meanperm_corr, sprintf('%s/null_corrs_%s_%s_%s.txt', options(1).value,options(2).value,options(4).value, num2str(options(6).value)));
        end
    end
end


% ensemble model - with demographics/clinical variables
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

            [~, correl, ~, ~] = load_null_models_ensemble(options);
        
            % calcualte the mean performance of all null distributions
            %null_perm=29
            clear meanperm
           % for n=1:null_perm
           %     meanperm(n,:)=mean(nullr2scores(:,n,:),3)';
           % end

            % calcualte the mean performance of all null distributions
           % null_perm=29
           % for n=1:null_perm
           %     meanperm_corr(n,:)=mean(nullcorrel(:,n,:),3)';
           % end

           % writematrix(r2scores, sprintf('%s/r2scores_%s_%s_%s.txt', options(1).value,options(2).value,options(4).value, num2str(options(6).value)));
            writematrix(correl, sprintf('%s/corrs_%s_%s_%s_ensemble.txt', options(1).value,options(2).value,options(4).value, num2str(options(6).value)));
            %writematrix(meanperm, sprintf('%s/null_r2scores_%s_%s_%s.txt', options(1).value,options(2).value,options(4).value, num2str(options(6).value)));
            %writematrix(meanperm_corr, sprintf('%s/null_corrs_%s_%s_%s.txt', options(1).value,options(2).value,options(4).value, num2str(options(6).value)));
        end
    end
end

