% Chord plots for variable importance etc

% pairwise shen 
feats_shen=[]
for i=1:100
    c=load(sprintf('~/GIT/ENIGMA/enigma_disconnections/correlation_regionalSDC_shen268/correlationp%i_SC_features.mat', i-1));
    feats_shen=vertcat(feats_shen, data);
end

