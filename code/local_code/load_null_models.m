function [r2scores, correl, nullr2scores, nullcorrel] = load_null_models(varargin)
    
    defaults(1).name = 'results_path';
    defaults(1).value = '/Users/emilyolafson/GIT/ENIGMA/enigma_disconnections/results' ;
    defaults(2).name = 'atlas';
    defaults(2).value ='fs86subj';
    defaults(3).name = 'y_var';
    defaults(3).value = 'normed_motor_scores';
    defaults(4).name = 'chaco_type';
    defaults(4).value = 'chacovol';
    defaults(5).name ='mdl_label';
    defaults(5).value = 'ridge';
    defaults(6).name ='crossval';
    defaults(6).value = 1;
    defaults(7).name ='n';
    defaults(7).value = 1;
    defaults(8).name = 'subset';
    defaults(8).value = 'chronic';

    if nargin == 0
        options = defaults;
    else
        options = varargin{1};
    end
    
    % Load model results
    filename_base= sprintf('%s/%s_%s_%s_%s_%s_crossval%i_',...
            options(1).value,options(2).value, options(3).value, ...
            options(4).value,options(8).value, options(5).value, ...
            options(6).value);
        
    for perm_number=1:25
        r2file = sprintf('%s%i_scores.txt', filename_base,perm_number-1);
        corrfile = sprintf('%s%i_correlations.txt', filename_base,perm_number-1);

        r2scores(perm_number,:) = readtable(r2file).Var1;
        correl(perm_number,:) = readtable(corrfile).Var1;
    end
    nullcorrel=[]
    nullr2scores=[]
    % Load null model results
  %  for perm_number=1:25
   %     for null_number = 1:29
   %         r2file = sprintf('%s%i_null_%i_scores.txt', filename_base,perm_number-1,null_number);
   %         corrfile = sprintf('%s%i_null_%i_correlations.txt', filename_base,perm_number-1,null_number);
%
  %         nullr2scores(perm_number, null_number,:) = readtable(r2file).Var1;
  % 5         nullcorrel(perm_number, null_number,:) =readtable(corrfile).Var1;
   %     end
   % end
    
    
    
    