% make figures about pairwise model results

%% 1. number of features/best lambdas


clear options
atlases = {'fs86subj', 'shen268'};
chaco_types = {'chacovol', 'chacoconn'};
crossval_schemes = [1, 5];
figure('Position', [0 0 300 300])

tiledlayout(2,4,'padding','none')
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
            [~, ~, ~, ~, alphas, feats] = load_models(options);
            % regional fs86
            nexttile
            X=reshape(feats, [], 1);
            Y=log10(reshape(alphas,[],1))
            XY = [X,Y];
            % Find unique rows and corresponding indices
            [uniqueAB, ~, n] = unique(XY, 'rows');
            % Find number of occurrences
            nHist = hist(n, unique(n));
            mx = max(nHist);
            % Create colors for each number of occurrence
            colors = brewermap(mx,'Spectral');
            colormap(colors);
            % Construct a color matrix
            cMatrix = colors(nHist, :);
            scatter(uniqueAB(:, 1), uniqueAB(:, 2), 100, cMatrix, 'filled');
            colorbar
            caxis([1 mx])
            xlim([min(X), max(X)])
            xlabel('Number of features')
            ylabel('Lambda values')
            title([options(2).value, num2str(options(6).value), options(4).value])
            set(gca, 'FontSize', 20)
            %saveas(gcf, sprintf('/Users/emilyolafson/GIT/ENIGMA/enigma_disconnections/figures/%s_%s_%s_nFeats_Alphas.png',options(2).value,options(4).value,num2str(options(6).value)))

         end
    end
end
saveas(gcf,'/Users/emilyolafson/GIT/ENIGMA/enigma_disconnections/figures/nFeats_Alphas.png')

