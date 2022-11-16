
import run_regression_models
from run_regression_models import run_models
from imp import reload
reload(run_regression_models)



# figure: 
# 

kwargs = {'covariates':['AGE', 'SEX', 'DAYS_POST_STROKE'], \
          'results_path':'/home/ubuntu/enigma/results/test_1/',\
          'lesionload_types': ['all', 'M1', 'none'], \
          'nperms':8}

run_models(**kwargs)
