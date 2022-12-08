import numpy as np 
from scipy.stats import pearsonr
import os
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import explained_variance_score,mean_absolute_error
import matplotlib
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Lasso, Ridge, ElasticNet,LinearRegression,LogisticRegression
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, f_regression
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, LeaveOneGroupOut
import logging
import glob
import math
from sklearn.ensemble import RandomForestClassifier

import warnings
warnings.filterwarnings('ignore') 

# Function comments are enhanced with ChatGPT.


def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
    '''
    FROM: https://stackoverflow.com/questions/7404116/defining-the-midpoint-of-a-colormap-in-matplotlib
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero.

    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower offset). Should be between
          0.0 and `midpoint`.
      midpoint : The new center of the colormap. Defaults to 
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax / (vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highest point in the colormap's range.
          Defaults to 1.0 (no upper offset). Should be between
          `midpoint` and 1.0.
    '''
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False), 
        np.linspace(midpoint, 1.0, 129, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=newcmap)

    return newcmap

def prepare_data(X):
    '''Clean X-data (remove zero-value input variables)'''

    # remove inputs that are 0 for all subjects
    zeros=X==0
    zeros=np.sum(zeros,0)
    zeros=zeros==X.shape[0]
    X=X[:,~zeros]
    logprint("Final size of X: " + str(X.shape))
    
    return X

def prepare_image_data(X):
    '''Clean X-data (remove zero-value input variables)'''

    # remove inputs that are 0 for all subjects
    X=np.reshape(X, (101,902629))
    logprint("Final size of X: " + str(X.shape))
    
    return X

def np_pearson_cor(x, y):
    '''Fast array-based pearson correlation that is more efficient. 
    FROM: https://cancerdatascience.org/blog/posts/pearson-correlation/.
        x - input N x p
        y - output N x 1
        
        returns correlation p x 1 '''
    xv = x - x.mean(axis=0)
    yv = y - y.mean(axis=0)
    xvss = (xv * xv).sum(axis=0)
    yvss = (yv * yv).sum(axis=0)
    result = np.matmul(xv.transpose(), yv) / np.sqrt(np.outer(xvss, yvss))
    
    # bound the values to -1 to 1 in the event of precision issues
    return np.maximum(np.minimum(result, 1.0), -1.0)


def np_pearson_cor_abs(x, y):
    '''Fast array-based pearson correlation that is more efficient. 
    FROM: https://cancerdatascience.org/blog/posts/pearson-correlation/.
        x - input N x p
        y - output N x 1
        
        returns correlation p x 1 '''
    logprint(x.shape)
    logprint(y.shape)
    xv = x - x.mean(axis=0)
    yv = y - y.mean(axis=0)
    xvss = (xv * xv).sum(axis=0)
    yvss = (yv * yv).sum(axis=0)
    result = np.matmul(xv.transpose(), yv) / np.sqrt(np.outer(xvss, yvss))
    
    # bound the values to -1 to 1 in the event of precision issues
    return abs(np.maximum(np.minimum(result, 1.0), -1.0))

        
def save_plots_true_pred(true, pred,filename, corr):
    f1 = plt.figure()
    plt.scatter(true, pred,s=10, c='black')
    plt.xlim(-0.1, 1.1)
    plt.ylim(-0.1, 1.1)
    plt.text(0.1, 0.1, corr)
    plt.xlabel('True normed motor score')
    plt.ylabel('Predicted normed motor score')
    np.save(os.path.join(filename + "_true.npy"), true)
    np.save(os.path.join( filename + "_pred.npy"), pred)

    plt.savefig(filename +'_truepred.png')
      
def naive_pearson_cor(X, Y):
    '''Naive (scipy-based/iterative) pearson correlation. 
    FROM: https://cancerdatascience.org/blog/posts/pearson-correlation/.
        x - input N x p
        y - output N x 1
        
        returns correlation p x 1 '''
    result = np.zeros(shape=(X.shape[1], Y.shape[1]))
    for i in range(X.shape[1]):
        for j in range(Y.shape[1]):
            r, _ = pearsonr(X[:,i], Y[:,j])
            result[i,j] = r
    return result[0]


def feature_select_correlation(x_train, x_test, y, a):
    # The feature_select_correlation function takes as input the training and test data matrices x_train and 
    # x_test, the variable to be predicted y, and the number of features to retain a. The function first checks that 
    # the number of features in the input data is greater than a. Then it calculates the absolute Pearson correlation 
    # between each feature and the target variable y. The function returns the indices of the top a features, as well as 
    # the training and test data matrices with only the selected features. The returned matrices and indices can be used 
    # for training and testing a machine learning model.
    
    """Return values for the top a features of x based on abs. value Spearman correlation with y.
         Inputs:
             x_train = input matrix for training subjects
             x_test = input matrix for test subjects
             y = variable/s predicted 
             a = number of features to retain
        
        Returns:
            x_train_featselect = training data with selected features 
            x_test_featselect = test data matrix with selected features
            ind = indices of top a features """
    
    # check that dimension of x is greater than a
    if x_train.shape[1]<a:
        raise Exception('Number of features in X is less than the number of features specified to retain (a).') 
        
        
    # Feature selection: use only the top n features based on correlation of training features with y
    correl = abs(np_pearson_cor(x_train, y))
    ind = np.argpartition(correl, -a, 0)[-a:] # select top a features

    # return training/test data with only top features
    x_train_featselect=np.squeeze(x_train[:,ind],2)
    
    x_test_featselect=np.squeeze(x_test[:,ind],2)

    return x_train_featselect,x_test_featselect, ind

def determine_featselect_range(X):
    #This function determines the range of values to use for the k parameter in the SelectKBest
    # feature selection method. The k parameter specifies the number of top-ranking features to select.

    # The range of values is determined using the number of columns in the input matrix X, with a minimum value of
    # X_min_dim and a maximum value of X_max_dim. The range is determined by taking n logarithmically spaced values
    # between the logarithms of these minimum and maximum values, with a base of base. These values are then converted
    # to int values and returned as the k_range.
    X_max_dim = X.shape[1]
    X_min_dim = 5
    n = 30
    base = 2
    k_range=np.logspace(math.log(X_min_dim,base),math.log(X_max_dim,base), n,base=base, dtype=int)
    return k_range

def get_models(model_type='regression', model_list=None):
    # This code defines a function named get_models() that takes two arguments: model_type and model_list. 
    # The function uses the model_type argument to determine which set of models to return. If model_type is set to 'regression',
    # the function will return a list of regression models. If model_type is set to 'classification', the function 
    # will return a list of classification models. The model_list argument is used to filter the models that are returned 
    # by the function. Only models whose names are included in the model_list will be returned. The function returns 
    # a tuple containing two elements: a list of models and a list of labels for those models.
    
    mdls=[]
    if model_type == 'regression':
        if 'ridge' in model_list:
            ridge = Pipeline([('featselect', SelectKBest(f_regression)), ('ridge', Ridge(normalize=True, max_iter=1000000, random_state=0))])
            mdls.append(ridge)
        if 'elastic_net' in model_list:
            elastic_net =  Pipeline([('featselect', SelectKBest(f_regression)),('elastic_net', ElasticNet(normalize=True, max_iter=1000000, random_state=0))])
            mdls.append(elastic_net)
        if 'lasso' in model_list:
            lasso =  Pipeline([('featselect', SelectKBest(f_regression)),('lasso', Lasso(normalize=True, max_iter=1000000, random_state=0))])
            mdls.append(lasso)
        if 'ensemble_reg' in model_list:
            ensemble_reg = Pipeline([('ensemble_reg', LinearRegression())])
            mdls.append(ensemble_reg)
        if 'linear_regression' in model_list:
            linear_regression = Pipeline([('linear_regression', LinearRegression())])
            mdls.append(linear_regression)
        if 'ridge_nofeatselect' in model_list:
            ridge_nofeatselect = Pipeline([('ridge_nofeatselect', Ridge(normalize=True, max_iter=1000000, random_state=0))])
            mdls.append(ridge_nofeatselect)
        mdls_labels = model_list
        
        return mdls, mdls_labels
        
    elif model_type == 'classification': 
        logprint('Selecting classification models')
        if 'svm' in model_list:
            svm = Pipeline([ ('svm', SVC(probability=True, class_weight='balanced', kernel='linear', random_state=0))])
            mdls.append(svm)
        if 'rbf_svm' in model_list:
            rbf_svm = Pipeline([('svm', SVC(probability=True, class_weight='balanced', kernel='rbf', random_state=0))])
            mdls.append(rbf_svm)  
        if 'log' in model_list:
            log = Pipeline([('logistic', LogisticRegression(class_weight='balanced', solver='liblinear', random_state=0))])
            mdls.append(log)
        if 'xgboost' in model_list:
            xgboost = Pipeline([('xgboost', XGBClassifier(eval_metric='auc', nthread=1, random_state=0))])
            mdls.append(xgboost)
        if 'rf' in model_list:
            rf = Pipeline([('rf', RandomForestClassifier())])
            mdls.append(rf)
        mdls_labels = model_list 
    return mdls, mdls_labels

def inner_loop(mdl, mdl_label, X, Y, group, inner_cv, n_jobs):
    
    if mdl_label =='ensemble_reg':
        logprint('No feature selection')
    elif mdl_label=='linear_regression':
        logprint('No feature selection -- Linear regression')
    elif mdl_label=='ridge_nofeatselect':
        logprint('No feature selection -- Ridge regression')
    else:
        k_range = determine_featselect_range(X)
    
    if mdl_label=='ridge':
        grid_params ={'ridge__alpha': np.logspace(-2, 2, 30, base=10,dtype=None),
                      'featselect__k':k_range}
        score = 'explained_variance'
        
    elif mdl_label =='ridge_nofeatselect':
        grid_params ={'ridge_nofeatselect__alpha': np.logspace(-2, 2, 30, base=10,dtype=None)}
        score = 'explained_variance'
        
    elif mdl_label=='ensemble_reg':
        score = 'explained_variance'
        
    elif mdl_label=='linear_regression':
        score = 'explained_variance'
            
    elif mdl_label=='elastic_net':
        grid_params ={'elastic_net__alpha': np.logspace(-2, 2, 30, base=10,dtype=None),
                      'featselect__k':k_range}
        score = 'explained_variance'
        
    elif mdl_label=='lasso':
        grid_params ={'lasso__alpha': np.logspace(-2, 2, 30, base=10,dtype=None),
                      'featselect__k':k_range}
        score = 'explained_variance'
        
    elif mdl_label == 'svm':
        grid_params = {'svm__C': [0.0001, 0.001, 0.01, 0.1, 1]}
        score = 'roc_auc'
    elif mdl_label== 'rbf_svm':
        grid_params = {'svm__C': [0.0001, 0.001, 0.01, 0.1, 1],
                        'svm__gamma': [0.001, 0.01, 0.1, 1]}
        score = 'roc_auc'
    elif mdl_label=='log':
        grid_params = {'logistic__C': [0.0001, 0.001, 0.01, 0.1, 1, 10],
                       'logistic__penalty': ['l1', 'l2']}
        score = 'roc_auc'
    elif mdl_label=='xgboost':
        grid_params = {'xgboost__gamma': [0.5, 1, 1.5, 2, 5],
                       'xgboost__learning_rate': [0.01, 0.1, 0.3],
                       'xgboost__max_depth': [3, 4, 5]}
    elif mdl_label=='rf':
        grid_params = {'rf__n_estimators': [5, 10, 15, 20],
                 'rf__max_depth': [2, 5, 7, 9]}
        score = 'roc_auc'
    else:
        logprint('Model not found..')
        return mdl
    
    if mdl_label == 'ensemble_reg':
        return mdl
    elif mdl_label=='linear_regression':
        return mdl
    else:
        logprint('Performing grid search for: {} \n'.format(mdl_label))

        grid_search = GridSearchCV(estimator=mdl, param_grid=grid_params, scoring=score, cv=inner_cv, refit=True, verbose=1,
                                n_jobs=n_jobs, return_train_score=False, pre_dispatch='2*n_jobs')

    grid_search.fit(X, Y, group)
    best_mdl = grid_search.best_estimator_

    return best_mdl

def haufe_transform_results(X_train, y_train, cols, mdl, mdl_label, chaco_type, atlas, X):
    # This code mplements a transformation for the results of a machine learning model trained on brain imaging data. 
    # The function takes as input the training data (X_train, y_train), the indices of the columns in the training data used as features (cols),
    # the trained model (mdl), the label of the model (mdl_label), the type of cross-validation used (chaco_type), the atlas used (atlas),
    # and the full data matrix (X).

    # The function first computes the covariance matrices of the dependent and independent variables. Then, it computes the activation and 
    # beta coefficients of the model. Depending on the value of chaco_type and atlas, the function may apply further transformations
    # to the activation and beta coefficients. Finally, the function returns the transformed activation and beta coefficients. 
    
    cov_y=np.cov(np.transpose(y_train))
    x_sub = X_train[:,cols]
    cov_x=np.cov(np.transpose(x_sub))
    beta_coeffs = mdl.named_steps[mdl_label].coef_
    
    weight=beta_coeffs
    activation=np.matmul(cov_x,weight)*(1/cov_y)

    # first transform the beta coefficients

    if chaco_type =='chacovol':
        if atlas == 'fs86subj':
            
            idx=np.ones(shape=(86,1), dtype='bool')
            idx[cols]=False # set SC weights that are features to be 1
            idx=idx.flatten()
            zeroidx=np.arange(0, 86, dtype='int')
            zeroidx=zeroidx[idx]
            
            # fill spots with 0's (up to 3192)
            k=0
            activation_full = beta_coeffs
            while k < zeroidx.shape[0]:
                activation_full=np.insert(activation_full, zeroidx[k],0)
                k=k+1
            
            #logprint("Full 3192: " + str(np.sum(activation_full>0)))
            # fill spots with 0's (up to 3655)
            zeros=X==0
            zeros=np.sum(zeros,0) # number of zeros across subjects
            zeros=zeros==X.shape[0] # find columns with zeros for all 101 subjects
            X=X[:,~zeros]
            
            zeroidx=np.arange(0, 86)
            zeroidx=zeroidx[zeros]
            
            # fill spots with 0's
            k=0
            a = activation_full
            while k < zeroidx.shape[0]:
                a=np.insert(a, zeroidx[k],0)
                k=k+1
            
            beta_coeffs = a
        elif atlas == 'shen268':
            
            idx=np.ones(shape=(268,1), dtype='bool')
            idx[cols]=False # set SC weights that are features to be 1
            idx=idx.flatten()
            zeroidx=np.arange(0, 268, dtype='int')
            zeroidx=zeroidx[idx]
            
            # fill spots with 0's (up to 3192)
            k=0
            activation_full = beta_coeffs
            while k < zeroidx.shape[0]:
                activation_full=np.insert(activation_full, zeroidx[k],0)
                k=k+1
            
            #logprint("Full 3192: " + str(np.sum(activation_full>0)))
            # fill spots with 0's (up to 3655)
            zeros=X==0
            zeros=np.sum(zeros,0) # number of zeros across subjects
            zeros=zeros==X.shape[0] # find columns with zeros for all 101 subjects
            X=X[:,~zeros]
            
            zeroidx=np.arange(0, 268)
            zeroidx=zeroidx[zeros]
            
            # fill spots with 0's
            k=0
            a = activation_full
            while k < zeroidx.shape[0]:
                a=np.insert(a, zeroidx[k],0)
                k=k+1
            
            activation = a
            beta_coeffs = a
    elif chaco_type=='chacoconn':
        if atlas == 'fs86subj':
                
            idx=np.ones(shape=(3192,1), dtype='bool')
            idx[cols]=False # set SC weights that are features to be 1
            idx=idx.flatten()
            zeroidx=np.arange(0, 3192, dtype='int')
            zeroidx=zeroidx[idx]
            
            # fill spots with 0's (up to 3192)
            k=0
            activation_full = beta_coeffs
            while k < zeroidx.shape[0]:
                activation_full=np.insert(activation_full, zeroidx[k],0)
                k=k+1
            
            #logprint("Full 3192: " + str(np.sum(activation_full>0)))
            # fill spots with 0's (up to 3655)
            zeros=X==0
            zeros=np.sum(zeros,0) # number of zeros across subjects
            zeros=zeros==X.shape[0] # find columns with zeros for all 101 subjects
            X=X[:,~zeros]
            
            zeroidx=np.arange(0, 3655)
            logprint(zeroidx.shape)
            logprint(zeros.shape)
            zeroidx=zeroidx[zeros]
            
            # fill spots with 0's
            k=0
            a = activation_full
            while k < zeroidx.shape[0]:
                a=np.insert(a, zeroidx[k],0)
                k=k+1
            
            activation = a
            fs86_counts = np.zeros((86, 86))
            inds = np.triu_indices(86, k=1)
            fs86_counts[inds] = activation
            beta_coeffs = fs86_counts
        elif atlas == 'shen268':
            idx=np.ones(shape=(25056,1), dtype='bool')
            idx[cols]=False # set SC weights that are features to be 1
            idx=idx.flatten()
            zeroidx=np.arange(0, 25056, dtype='int')
            zeroidx=zeroidx[idx]
            
            # fill spots with 0's (up to 3192)
            k=0
            activation_full = beta_coeffs
            while k < zeroidx.shape[0]:
                activation_full=np.insert(activation_full, zeroidx[k],0)
                k=k+1
            
            #logprint("Full 3192: " + str(np.sum(activation_full>0)))
            # fill spots with 0's (up to 3655)
            zeros=X==0
            zeros=np.sum(zeros,0) # number of zeros across subjects
            zeros=zeros==X.shape[0] # find columns with zeros for all 101 subjects
            X=X[:,~zeros]
            
            zeroidx=np.arange(0, 35778)
            zeroidx=zeroidx[zeros]
            
            # fill spots with 0's
            k=0
            a = activation_full
            while k < zeroidx.shape[0]:
                a=np.insert(a, zeroidx[k],0)
                k=k+1
            
            activation = a
            shen268_counts = np.zeros((268, 268))
            inds = np.triu_indices(268, k=1)
            shen268_counts[inds] = activation
            beta_coeffs = shen268_counts
        

    # then transform the haufe activations
    activation=np.matmul(cov_x,weight)*(1/cov_y)

    if chaco_type =='chacovol':
        if atlas == 'fs86subj':
            
            idx=np.ones(shape=(86,1), dtype='bool')
            idx[cols]=False # set SC weights that are features to be 1
            idx=idx.flatten()
            zeroidx=np.arange(0, 86, dtype='int')
            zeroidx=zeroidx[idx]
            
            # fill spots with 0's (up to 3192)
            k=0
            activation_full = activation
            while k < zeroidx.shape[0]:
                activation_full=np.insert(activation_full, zeroidx[k],0)
                k=k+1
            
            #logprint("Full 3192: " + str(np.sum(activation_full>0)))
            # fill spots with 0's (up to 3655)
            zeros=X==0
            zeros=np.sum(zeros,0) # number of zeros across subjects
            zeros=zeros==X.shape[0] # find columns with zeros for all 101 subjects
            X=X[:,~zeros]
            
            zeroidx=np.arange(0, 86)
            zeroidx=zeroidx[zeros]
            
            # fill spots with 0's
            k=0
            a = activation_full
            while k < zeroidx.shape[0]:
                a=np.insert(a, zeroidx[k],0)
                k=k+1
            
            activation = a
        elif atlas == 'shen268':
            
            idx=np.ones(shape=(268,1), dtype='bool')
            idx[cols]=False # set SC weights that are features to be 1
            idx=idx.flatten()
            zeroidx=np.arange(0, 268, dtype='int')
            zeroidx=zeroidx[idx]
            
            # fill spots with 0's (up to 3192)
            k=0
            activation_full = activation
            while k < zeroidx.shape[0]:
                activation_full=np.insert(activation_full, zeroidx[k],0)
                k=k+1
            
            #logprint("Full 3192: " + str(np.sum(activation_full>0)))
            # fill spots with 0's (up to 3655)
            zeros=X==0
            zeros=np.sum(zeros,0) # number of zeros across subjects
            zeros=zeros==X.shape[0] # find columns with zeros for all 101 subjects
            X=X[:,~zeros]
            
            zeroidx=np.arange(0, 268)
            zeroidx=zeroidx[zeros]
            
            # fill spots with 0's
            k=0
            a = activation_full
            while k < zeroidx.shape[0]:
                a=np.insert(a, zeroidx[k],0)
                k=k+1
            
            activation = a
    elif chaco_type=='chacoconn':
        if atlas == 'fs86subj':
                
            idx=np.ones(shape=(3192,1), dtype='bool')
            idx[cols]=False # set SC weights that are features to be 1
            idx=idx.flatten()
            zeroidx=np.arange(0, 3192, dtype='int')
            zeroidx=zeroidx[idx]
            
            # fill spots with 0's (up to 3192)
            k=0
            activation_full = activation
            while k < zeroidx.shape[0]:
                activation_full=np.insert(activation_full, zeroidx[k],0)
                k=k+1
            
            #logprint("Full 3192: " + str(np.sum(activation_full>0)))
            # fill spots with 0's (up to 3655)
            zeros=X==0
            zeros=np.sum(zeros,0) # number of zeros across subjects
            zeros=zeros==X.shape[0] # find columns with zeros for all 101 subjects
            X=X[:,~zeros]
            
            zeroidx=np.arange(0, 3655)
            logprint(zeroidx.shape)
            logprint(zeros.shape)
            zeroidx=zeroidx[zeros]
            
            # fill spots with 0's
            k=0
            a = activation_full
            while k < zeroidx.shape[0]:
                a=np.insert(a, zeroidx[k],0)
                k=k+1
            
            activation = a
            fs86_counts = np.zeros((86, 86))
            inds = np.triu_indices(86, k=1)
            fs86_counts[inds] = activation
            activation = fs86_counts
        elif atlas == 'shen268':
            idx=np.ones(shape=(25056,1), dtype='bool')
            idx[cols]=False # set SC weights that are features to be 1
            idx=idx.flatten()
            zeroidx=np.arange(0, 25056, dtype='int')
            zeroidx=zeroidx[idx]
            
            # fill spots with 0's (up to 3192)
            k=0
            activation_full = activation
            while k < zeroidx.shape[0]:
                activation_full=np.insert(activation_full, zeroidx[k],0)
                k=k+1
            
            #logprint("Full 3192: " + str(np.sum(activation_full>0)))
            # fill spots with 0's (up to 3655)
            zeros=X==0
            zeros=np.sum(zeros,0) # number of zeros across subjects
            zeros=zeros==X.shape[0] # find columns with zeros for all 101 subjects
            X=X[:,~zeros]
            
            zeroidx=np.arange(0, 35778)
            zeroidx=zeroidx[zeros]
            
            # fill spots with 0's
            k=0
            a = activation_full
            while k < zeroidx.shape[0]:
                a=np.insert(a, zeroidx[k],0)
                k=k+1
            
            activation = a
            shen268_counts = np.zeros((268, 268))
            inds = np.triu_indices(268, k=1)
            shen268_counts[inds] = activation
            activation = shen268_counts
        
    return activation, beta_coeffs

def create_outer_cv(outer_cv_id):
    #This code defines a function that creates a cross-validation object for use in training and evaluating machine learning models.
    # The function takes as input an identifier for the type of cross-validation to use (outer_cv_id).
    
    if outer_cv_id=="1": # random
        outer_cv = KFold(n_splits=5, shuffle=True)
    elif outer_cv_id =="2": # leave one group out
        outer_cv = LeaveOneGroupOut()
    elif outer_cv_id =='3':
        outer_cv = GroupKFold(n_splits=5)
    elif outer_cv_id == "4" or outer_cv_id =="5":
        outer_cv = GroupShuffleSplit(train_size=.8)
    return outer_cv
        
def create_inner_cv(inner_cv_id, perm):
    #This code defines a function that creates a cross-validation object for use in training and evaluating machine learning models.
    # The function takes as input an identifier for the type of cross-validation to use (inner_cv_id) and a random seed (perm).
    
    # Based on the value of inner_cv_id, the function returns a different cross-validation object. For example,
    # if inner_cv_id is 1, the function returns a KFold object with 5 splits and shuffling enabled. If inner_cv_id is 3, 
    # the function returns a GroupKFold object with 5 splits. This allows the user to easily create different types of
    # cross-validation objects without having to specify all the parameters each time.
    
    if inner_cv_id=="1": # random
        inner_cv = KFold(n_splits=5, shuffle=True, random_state=perm)
    elif inner_cv_id =="2": # leave one group out
        inner_cv = KFold(n_splits=5, shuffle=True, random_state = perm)
    elif inner_cv_id =='3':
        inner_cv = GroupKFold(n_splits=5)
    elif inner_cv_id == "4":
        inner_cv = KFold(n_splits=5, shuffle=True,random_state=perm)
    elif inner_cv_id == "5":
        inner_cv = GroupShuffleSplit(train_size = 0.8)
    return inner_cv
    
def run_regression(x, Y, group, inner_cv_id, outer_cv_id, model_tested, atlas, y_var, chaco_type, subset, save_models,results_path,crossval_type,nperms,null, output_path):
    # This code implements a cross-validation procedure for training and evaluating machine learning models on brain imaging data. 
    # The function takes as input the features (x), labels (Y), grouping information (group), inner and outer cross-validation indices 
    # (inner_cv_id, outer_cv_id), a list of models to test (model_tested), an atlas of the brain (atlas), the name of the dependent variable 
    # (y_var), a string indicating the type of structural disconnection (chaco_type), a subset of the data to use (subset), a flag indicating whether
    # to save trained models (save_models), a path to save the results (results_path), the type of cross-validation (crossval_type), the number
    # of permutations to run (nperms), a flag indicating whether to run a null model (null), a path to save the output (output_path).
    #
    #The function first prepares the features x for training by reshaping them based on the value of atlas. Next, it creates the outer cross-validation object
    # and splits the data into training and testing sets using the provided indices. Then, it loops through the different models specified in
    # model_tested and trains and evaluates each model using the training and testing sets. The function returns the trained models, 
    # explained variance, variable importance, correlations, and size of the test group for each model.
    # - ChatGPT

        
    if atlas =='lesionload_m1' or atlas == 'lesionload_slnm':
        X=np.array(x).reshape(-1,1)
    elif atlas == 'lesionload_all':
        X=np.array(x)
    elif atlas == 'lesionload_all_2h':
        X=np.array(x)
    else:
        X = prepare_data(x) 
        
    logprint(X.shape)
    
    outer_cv = create_outer_cv(outer_cv_id)
    
    outer_cv_splits = outer_cv.get_n_splits(X, Y, group)
    
    models = np.zeros((1, outer_cv_splits), dtype=object)
    explained_var  = np.zeros((1,outer_cv_splits), dtype=object)
    variable_importance  = []
    correlations  = np.zeros((1,outer_cv_splits), dtype=object)
    size_testgroup =[]

    for n in range(0,nperms):
        
        logprint('\n\n~ ~ ~ ~ ~ ~ ~ ~ ~ ~ PERMUTATION: {}/{} ~ ~ ~ ~ ~ ~ ~ ~ ~ \n\n'.format(n, nperms))
        
        activation_weights=[]
        beta_coeffs_weights=[]
        for cv_fold, (train_id, test_id) in enumerate(outer_cv.split(X, Y, group)): # split on chronic data
            
            logprint("------ Outer Fold: {}/{} ------".format(cv_fold + 1, outer_cv_splits))
            
            X_train, X_test = X[train_id], X[test_id]
            y_train, y_test = Y[train_id], Y[test_id]
   
            group_train, group_test = group[train_id], group[test_id]
            
            logprint('Size of test group: {}'.format(group_test.shape[0]))
            logprint('Size of train group: {}'.format(group_train.shape[0]))
            
            logprint('Number of sites in test set: {}'.format(np.unique(group_test).shape[0]))            
            mdls, mdls_labels = get_models('regression', model_tested) 

            size_testgroup.append(group_test.shape[0])
            
            mdl_idx=0
            inner_cv = create_inner_cv(inner_cv_id,n)
            
            for mdl, mdl_label in zip(mdls, mdls_labels): 
                mdl = inner_loop(mdl, mdl_label, X_train, y_train, group_train, inner_cv, 10)  
                #print('alpha: {}'.format(mdl.named_steps[mdl_label].alpha))
                mdl.fit(X_train, y_train)
                #print( mdl.named_steps[mdl_label].coef_)
                if model_tested == 'ridge':
                    cols = mdl['featselect'].get_support(indices=True)
                    ## HAUFE TRANSFORMS:
                    activation, beta_coeffs = haufe_transform_results(X_train, y_train, cols, mdl, mdl_label, chaco_type, atlas, x)
                    print(beta_coeffs.shape)
                    activation_weights.append(activation)
                    beta_coeffs_weights.append(beta_coeffs)

                elif model_tested == 'ridge_nofeatselect':
                    print('yabadabadoo')
                    if atlas =='lesionload_all':
                        cols = [0, 1, 2, 3, 4, 5]
                    elif atlas == 'lesionload_all_2h':
                        cols = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

                    ## HAUFE TRANSFORMS:
                    activation, beta_coeffs = haufe_transform_results(X_train, y_train, cols, mdl, mdl_label, chaco_type, atlas, x)
                    activation_weights.append(activation)
                    beta_coeffs_weights.append(beta_coeffs)

                else:
                    activation_weights=[]
                    beta_coeffs_weights=[]


                y_pred= mdl.predict(X_test)
                
                expl=explained_variance_score(y_test, y_pred)
                filename =  '/{}_{}_{}_{}_{}_crossval{}_perm{}'.format(atlas, y_var, chaco_type, subset, mdl_label,crossval_type,n)

                
                variable_importance.append(mdl.named_steps[mdl_label].coef_)
                correlations[mdl_idx, cv_fold] = np_pearson_cor(y_test,y_pred)[0]
                
                logprint('R^2 score: {} '.format(np.round(explained_variance_score(y_test, y_pred), 3)))
                logprint('Correlation: {} '.format(np.round(np_pearson_cor(y_test,y_pred)[0][0], 3)))
                explained_var[mdl_idx, cv_fold]=expl
                if save_models:
                    models[mdl_idx, cv_fold] = mdl
                    
                mdl_idx += 1
                logprint('\n')

        if null>0:
            logprint('NULL!')
            filename = filename + '_null_' + str(null)

        logprint('\n\n')
        np.save(os.path.join(results_path, output_path,filename+ "_scores.npy"), explained_var)
        np.save(os.path.join(results_path,output_path, filename + "_model.npy"), models)
        np.save(os.path.join(results_path,output_path, filename+"_correlations.npy"), correlations)
        np.save(os.path.join(results_path,output_path, filename + "_beta_coeffs.npy"), beta_coeffs_weights)

        np.save(os.path.join(results_path,output_path,filename + "_activation_weights.npy"), activation_weights)

        np.save(os.path.join(results_path,output_path, filename + "_model_labels.npy"), mdls_labels)
        np.save(os.path.join(results_path,output_path,filename+ "_variable_impts.npy"), variable_importance)
        np.save(os.path.join(results_path, output_path,filename+ "_test_group_sizes.npy"), size_testgroup)

def run_regression_ensemble(X1, C, Y, group, inner_cv_id, outer_cv_id, model_tested, atlas, y_var, chaco_type, subset, save_models,results_path,crossval_type,nperms,null,output_path):
    X2 = C

            
    logprint('\nRunning ensemble model!')
    
    if atlas =='lesionload_m1' or atlas == 'lesionload_slnm':
        X1=np.array(X1).reshape(-1,1)
    elif atlas == 'lesionload_all':
        X1=np.array(X1)
    elif atlas == 'lesionload_all_2h':
        X1=np.array(X1)
    else:
        X1 = prepare_data(X1) 
        
    outer_cv = create_outer_cv(outer_cv_id)

    outer_cv_splits = outer_cv.get_n_splits(X1, Y, group)
    
    models = np.zeros((1, outer_cv_splits), dtype=object)
    explained_var  = np.zeros((1,outer_cv_splits), dtype=object)

    correlations_ensemble  = np.zeros((1,outer_cv_splits), dtype=object)
    mean_abs_error = np.zeros((1,outer_cv_splits), dtype=object)

    size_testgroup =[]
    for n in range(0,nperms):
        logprint('\n\n~ ~ ~ ~ ~ ~ ~ ~ ~ ~ PERMUTATION: {}/{} ~ ~ ~ ~ ~ ~ ~ ~ ~ \n\n'.format(n, nperms))
        
        for cv_fold, (train_id, test_id) in enumerate(outer_cv.split(X1, Y, group)):
            inner_cv = create_inner_cv(inner_cv_id,n)

            logprint("------ Outer Fold: {}/{} ------".format(cv_fold + 1, outer_cv_splits))
            
            X1_train, X1_test = X1[train_id], X1[test_id]
            X2_train, X2_test = X2[train_id], X2[test_id]

            y_train, y_test = Y[train_id], Y[test_id]

            group_train, group_test = group[train_id], group[test_id]
            
            logprint('Size of test group: {}'.format(group_test.shape[0]))
            logprint('Size of train group: {}'.format(group_train.shape[0]))
            logprint('Number of sites in test set: {}\n'.format(np.unique(group_test).shape[0]))            

            size_testgroup.append(group_test.shape[0])
            
            mdl_idx=0
            mdls, mdls_labels = get_models('regression', model_tested) 
            
            # first model: X1 (lesion data)
            logprint('~~ Running model 1: lesion info ~~~')
            for mdl, mdl_label in zip(mdls, mdls_labels): 
                mdl1 = inner_loop(mdl, mdl_label, X1_train, y_train, group_train, inner_cv, 10)  
                mdl1.fit(X1_train, y_train)
                y1_pred= mdl1.predict(X1_test)
                
            logprint('~~ Running model 2: demographics ~~~')
            # second model: demographic data
            mdls, mdl_labels = get_models('regression', ['linear_regression'])
            for mdl, mdl_label2 in zip(mdls, mdl_labels):
                mdl = inner_loop(mdl, mdl_label2, X2_train, y_train, group_train, inner_cv, 10)
                mdl.fit(X2_train, y_train)
                y2_pred= mdl.predict(X2_test)
            
            
            avg_pred = np.mean([y1_pred, y2_pred], axis=0)
            expl=explained_variance_score(y_test, avg_pred)
            filename = '/{}_{}_{}_{}_{}_crossval{}_perm{}_ensemble_demog'.format(atlas, y_var, chaco_type, subset, mdl_label,crossval_type,n)
            
            correlations_ensemble[mdl_idx, cv_fold] = np_pearson_cor(y_test,avg_pred)[0]
 
            
            mean_abs_error[mdl_idx, cv_fold] = mean_absolute_error(y_test, avg_pred)
   
            
            explained_var[mdl_idx, cv_fold] =explained_variance_score(y_test, avg_pred)

            logprint('\n')
            logprint('R^2 score (ensemble): {} '.format(np.round(explained_variance_score(y_test, avg_pred), 3)))
            logprint('Correlation (ensemble): {} '.format(np.round(np_pearson_cor(y_test,avg_pred)[0][0], 3)))
            logprint('\n')
            logprint('Corr chaco only: {} '.format(np.round(np_pearson_cor(y_test, y1_pred)[0][0], 3)))
            logprint('Corr demog only: {} '.format(np.round(np_pearson_cor(y_test, y2_pred)[0][0], 3)))
            logprint('\n')
            explained_var[mdl_idx, cv_fold]=expl
            if save_models:
                models[mdl_idx, cv_fold] = mdl1
                
            mdl_idx += 1


        if null>0:
            logprint('NULL!')
            filename = filename + '_null_' + str(null)

                
        np.save(os.path.join(results_path, output_path,filename + "_scores.npy"), explained_var)
        np.save(os.path.join(results_path,output_path, filename + "_model.npy"), models)
        np.save(os.path.join(results_path,output_path, filename +"_correlations_ensemble.npy"), correlations_ensemble)
        np.save(os.path.join(results_path,output_path, filename + "_model_labels.npy"), mdls_labels)
        np.save(os.path.join(results_path, output_path,filename + "_test_group_sizes.npy"), size_testgroup)

def run_regression_chaco_ll(X1, X2, Y, group, inner_cv_id, outer_cv_id, model_tested, atlas, y_var, chaco_type, subset, save_models,results_path,crossval_type,nperms,null,output_path,ensemble_atlas):
    
    if atlas =='lesionload_m1' or atlas == 'lesionload_slnm':
        X1=np.array(X1).reshape(-1,1)
    elif atlas == 'lesionload_all':
        X1=np.array(X1)
    elif atlas == 'lesionload_all_2h':
        X1=np.array(X1)
        
        
    X2 = prepare_data(X2)
    
    logprint('\nRunning ensemble model!')
   
    outer_cv = create_outer_cv(outer_cv_id)

    outer_cv_splits = outer_cv.get_n_splits(X1, Y, group)
    
    models = np.zeros((1, outer_cv_splits), dtype=object)
    explained_var  = np.zeros((1,outer_cv_splits), dtype=object)
    variable_importance  = []
    correlations_ensemble  = np.zeros((1,outer_cv_splits), dtype=object)
    mean_abs_error = np.zeros((1,outer_cv_splits), dtype=object)
    size_testgroup =[]
    
    for n in range(0,nperms):
        logprint('\n\n~ ~ ~ ~ ~ ~ ~ ~ ~ ~ PERMUTATION: {}/{} ~ ~ ~ ~ ~ ~ ~ ~ ~ \n\n'.format(n, nperms))
        
        for cv_fold, (train_id, test_id) in enumerate(outer_cv.split(X1, Y, group)):
            inner_cv = create_inner_cv(inner_cv_id,n)

            logprint("------ Outer Fold: {}/{} ------".format(cv_fold + 1, outer_cv_splits))
            
            X1_train, X1_test = X1[train_id], X1[test_id]
            X2_train, X2_test = X2[train_id], X2[test_id]

            y_train, y_test = Y[train_id], Y[test_id]

            group_train, group_test = group[train_id], group[test_id]
                

            logprint('Size of test group: {}'.format(group_test.shape[0]))
            logprint('Size of train group: {}'.format(group_train.shape[0]))
            logprint('Number of sites in test set: {}\n'.format(np.unique(group_test).shape[0]))            

            size_testgroup.append(group_test.shape[0])
            
            mdl_idx=0
            mdls, mdls_labels = get_models('regression', model_tested) 
            
            # first model: X1 (lesion data)
            logprint('~~ Running model 1: lesion info ~~~')
            for mdl, mdl_label in zip(mdls, mdls_labels): 
                mdl1 = inner_loop(mdl, mdl_label, X1_train, y_train, group_train, inner_cv, 10)  
                mdl1.fit(X1_train, y_train)
                y1_pred= mdl1.predict(X1_test)
                
            logprint('~~ Running model 2: chaco ~~~')
            # second model: X2 (chaco data)
            mdls, mdl_labels = get_models('regression', ['ridge'])
            for mdl, mdl_label2 in zip(mdls, mdl_labels):
                mdl = inner_loop(mdl, mdl_label2, X2_train, y_train, group_train, inner_cv, 10)
                mdl.fit(X2_train, y_train)
                y2_pred= mdl.predict(X2_test)
                
            print(X2.shape[1])
            if X2.shape[1] == 86:
                chacoatlas = 'fs86subj'
            elif X2.shape[1] == 268:
                chacoatlas= 'shen268'
            
            avg_pred = np.mean([y1_pred, y2_pred], axis=0)
            expl=explained_variance_score(y_test, avg_pred)
            filename = '/{}_{}_{}_{}_{}_crossval{}_perm{}_ensemble_chacoLL_{}'.format(atlas, y_var, chaco_type, subset, mdl_label,crossval_type,n, ensemble_atlas)
            
            #variable_importance.append(mdl.named_steps[mdl_label].coef_)
            correlations_ensemble[mdl_idx, cv_fold] = np_pearson_cor(y_test,avg_pred)[0]

            mean_abs_error[mdl_idx, cv_fold] = mean_absolute_error(y_test, avg_pred)         
            explained_var[mdl_idx, cv_fold] =explained_variance_score(y_test, avg_pred)

            logprint('\n')
            logprint('R^2 score (ensemble): {} '.format(np.round(explained_variance_score(y_test, avg_pred), 3)))
            logprint('Correlation (ensemble): {} '.format(np.round(np_pearson_cor(y_test,avg_pred)[0][0], 3)))
            logprint('\n')
            logprint('Corr lesion only: {} '.format(np.round(np_pearson_cor(y_test, y1_pred)[0][0], 3)))
            logprint('Corr ChaCo only: {} '.format(np.round(np_pearson_cor(y_test, y2_pred)[0][0], 3)))
            
            logprint('\n')
            explained_var[mdl_idx, cv_fold]=expl
            if save_models:
                models[mdl_idx, cv_fold] = mdl1
                
            mdl_idx += 1


        if null>0:
            logprint('NULL!')
            filename = filename + '_null_' + str(null)

        np.save(os.path.join(results_path, output_path,filename + "_scores.npy"), explained_var)
        np.save(os.path.join(results_path,output_path, filename + "_model.npy"), models)
        np.save(os.path.join(results_path,output_path, filename +"_correlations_ensemble.npy"), correlations_ensemble)
        np.save(os.path.join(results_path,output_path, filename + "_model_labels.npy"), mdls_labels)
        np.save(os.path.join(results_path, output_path,filename + "_test_group_sizes.npy"), size_testgroup)


def run_regression_chaco_ll_demog(X1, X2, C, Y, group, inner_cv_id, outer_cv_id, model_tested, atlas, y_var, chaco_type, subset, save_models,results_path,crossval_type,nperms,null,output_path,ensemble_atlas):
    
    # X1 = lesion load 
    # X2 = chaco scores
    # X3 = C (demographic)
    if atlas =='lesionload_m1' or atlas == 'lesionload_slnm':
        X1=np.array(X1).reshape(-1,1)
    elif atlas == 'lesionload_all':
        X1=np.array(X1)
    elif atlas == 'lesionload_all_2h':
        X1=np.array(X1)
        
    X2 = prepare_data(X2)
    X3 = C
    
    logprint('\nRunning ensemble model!')
   
    outer_cv = create_outer_cv(outer_cv_id)

    outer_cv_splits = outer_cv.get_n_splits(X1, Y, group)
    
    models = np.zeros((1, outer_cv_splits), dtype=object)
    explained_var  = np.zeros((1,outer_cv_splits), dtype=object)
    correlations_ensemble  = np.zeros((1,outer_cv_splits), dtype=object)
    mean_abs_error = np.zeros((1,outer_cv_splits), dtype=object)
    size_testgroup =[]
    
    for n in range(0,nperms):
        logprint('\n\n~ ~ ~ ~ ~ ~ ~ ~ ~ ~ PERMUTATION: {}/{} ~ ~ ~ ~ ~ ~ ~ ~ ~ \n\n'.format(n, nperms))
        
        for cv_fold, (train_id, test_id) in enumerate(outer_cv.split(X1, Y, group)):
            inner_cv = create_inner_cv(inner_cv_id,n)

            logprint("------ Outer Fold: {}/{} ------".format(cv_fold + 1, outer_cv_splits))
            
            X1_train, X1_test = X1[train_id], X1[test_id]
            X2_train, X2_test = X2[train_id], X2[test_id]
            X3_train, X3_test = X3[train_id], X3[test_id]
            y_train, y_test = Y[train_id], Y[test_id]

            group_train, group_test = group[train_id], group[test_id]
                

            logprint('Size of test group: {}'.format(group_test.shape[0]))
            logprint('Size of train group: {}'.format(group_train.shape[0]))
            logprint('Number of sites in test set: {}\n'.format(np.unique(group_test).shape[0]))            

            size_testgroup.append(group_test.shape[0])
            
            mdl_idx=0
            mdls, mdls_labels = get_models('regression', model_tested) 
                
            # first model: X1 (lesion data)
            logprint('~~ Running model 1: lesion info ~~~')
            for mdl, mdl_label in zip(mdls, mdls_labels): 
                mdl1 = inner_loop(mdl, mdl_label, X1_train, y_train, group_train, inner_cv, 10)  
                mdl1.fit(X1_train, y_train)
                y1_pred= mdl1.predict(X1_test)
                
            logprint('~~ Running model 2: chaco ~~~')
            # second model: X2 (chaco data)
            mdls, mdl_labels = get_models('regression', ['ridge'])
            for mdl, mdl_label2 in zip(mdls, mdl_labels):
                mdl = inner_loop(mdl, mdl_label2, X2_train, y_train, group_train, inner_cv, 10)
                mdl.fit(X2_train, y_train)
                y2_pred= mdl.predict(X2_test)
                
            logprint('~~ Running model 3: demographics ~~~')
            # second model: demographic data
            mdls, mdl_labels = get_models('regression', ['linear_regression'])
            for mdl, mdl_label3 in zip(mdls, mdl_labels):
                mdl = inner_loop(mdl, mdl_label3, X3_train, y_train, group_train, inner_cv, 10)
                mdl.fit(X3_train, y_train)
                y3_pred= mdl.predict(X3_test)
                   
            print(X2.shape[1])
            if X2.shape[1] == 86:
                chacoatlas = 'fs86subj'
            elif X2.shape[1] == 268:
                chacoatlas= 'shen268'
            
            avg_pred = np.mean([y1_pred, y2_pred, y3_pred], axis=0)
            expl=explained_variance_score(y_test, avg_pred)
            filename = '/{}_{}_{}_{}_{}_crossval{}_perm{}_ensemble_chacoLLdemog_{}'.format(atlas, y_var, chaco_type, subset, mdl_label,crossval_type,n, ensemble_atlas)
            
            #variable_importance.append(mdl.named_steps[mdl_label].coef_)
            correlations_ensemble[mdl_idx, cv_fold] = np_pearson_cor(y_test,avg_pred)[0]

            mean_abs_error[mdl_idx, cv_fold] = mean_absolute_error(y_test, avg_pred)         
            explained_var[mdl_idx, cv_fold] =explained_variance_score(y_test, avg_pred)

            logprint('\n')
            logprint('R^2 score (ensemble): {} '.format(np.round(explained_variance_score(y_test, avg_pred), 3)))
            logprint('Correlation (ensemble): {} '.format(np.round(np_pearson_cor(y_test,avg_pred)[0][0], 3)))
            logprint('\n')
            logprint('Corr lesion only: {} '.format(np.round(np_pearson_cor(y_test, y1_pred)[0][0], 3)))
            logprint('Corr ChaCo only: {} '.format(np.round(np_pearson_cor(y_test, y2_pred)[0][0], 3)))
            logprint('Corr demog only: {} '.format(np.round(np_pearson_cor(y_test, y3_pred)[0][0], 3)))
            logprint('\n')
            
            explained_var[mdl_idx, cv_fold]=expl
            if save_models:
                models[mdl_idx, cv_fold] = mdl1
                
            mdl_idx += 1


        if null>0:
            logprint('NULL!')
            filename = filename + '_null_' + str(null)


        np.save(os.path.join(results_path, output_path,filename + "_scores.npy"), explained_var)
        np.save(os.path.join(results_path,output_path, filename + "_model.npy"), models)
        np.save(os.path.join(results_path,output_path, filename +"_correlations.npy"), correlations_ensemble)
        np.save(os.path.join(results_path,output_path, filename + "_model_labels.npy"), mdls_labels)
        np.save(os.path.join(results_path, output_path,filename + "_test_group_sizes.npy"), size_testgroup)


def set_vars_for_ll(lesionload_type):
    if lesionload_type =='M1':
        atlas = 'lesionload_m1'
        model_tested = 'linear_regression'
        chaco_type = 'NA'
    if lesionload_type =='slnm':
        atlas = 'lesionload_slnm'
        model_tested = 'linear_regression'
        chaco_type = 'NA'

    elif lesionload_type =='all':
        atlas = 'lesionload_all'
        model_tested= 'ridge_nofeatselect'
        chaco_type = 'NA'
        
    elif lesionload_type =='all_2h':
        atlas = 'lesionload_all_2h'
        model_tested= 'ridge_nofeatselect'
        chaco_type = 'NA' 
    return atlas, model_tested, chaco_type
        

def set_up_and_run_model(crossval, model_tested,lesionload,lesionload_type, X, Y, C, site, atlas, y_var, chaco_type, subset, save_models, results_path, nperms, null, ensemble, output_path,ensemble_atlas):
    # This function sets up the parameters for a machine learning model.
    # The function sets up the cross-validation method to be used based on the crossval input. 
    # Finally, it runs a machine learning regression using the specified parameters.
    
    if crossval == '1':
        logprint('1. Outer CV: Random partition fixed fold sizes, Inner CV: Random partition fixed fold sizes')
        # is random when random_state not specified 
        #outer_cv = KFold(n_splits=5, shuffle=True)
        outer_cv_id ='1'
        #inner_cv = KFold(n_splits=5, shuffle=True)
        inner_cv_id = '1'
    elif crossval == '2':
        logprint('2. Outer CV: Leave-one-site-out, Inner CV: Random partition fixed fold sizes')
        #outer_cv = LeaveOneGroupOut()
        #inner_cv = KFold(n_splits=5, shuffle=True)
        outer_cv_id = '2'
        inner_cv_id ='2'

    elif crossval == '3':
        logprint('3. Outer CV: Group K-fold, Inner CV: Group K-fold')
        #outer_cv = GroupKFold(n_splits=5)
        #inner_cv = GroupKFold(n_splits=5)
        outer_cv_id ='3'
        inner_cv_id = '3'
        
    elif crossval == '4':
        logprint('4 Outer CV: GroupShuffleSplit, Inner CV:  Random partition fixed fold sizes')
        #outer_cv = GroupShuffleSplit(train_size=.8)
        #inner_cv = KFold(n_splits=5, shuffle=True)
        outer_cv_id = '4'
        inner_cv_id = '4'
        
    elif crossval == '5':
        logprint('5 Outer CV: GroupShuffleSplit, Inner CV:  GroupShuffleSplit')
        #outer_cv = GroupShuffleSplit(train_size=.8)
        #inner_cv = GroupShuffleSplit(train_size = 0.8)
        outer_cv_id = '5'
        inner_cv_id = '5'
    
    
    if y_var == 'normed_motor_scores':
        if ensemble == 'none':
            if lesionload_type == 'none':
                if model_tested=='ridge':
                    kwargs = {'x':X, 'Y':Y, 'group':site, 'model_tested':model_tested, 'inner_cv_id':inner_cv_id, 'outer_cv_id':outer_cv_id, 'atlas':atlas, 'y_var':y_var, 'chaco_type':chaco_type, 'subset':subset,\
                        'save_models':save_models, 'results_path':results_path, 'crossval_type':crossval, 'nperms':nperms, 'null':null, 'output_path':output_path}
                    run_regression(**kwargs)
            elif lesionload_type =='M1':
                atlas = 'lesionload_m1'
                model_tested = ['linear_regression']
                chaco_type ='NA'
                kwargs = {'x':lesionload, 'Y':Y, 'group':site,  'model_tested':model_tested,'inner_cv_id':inner_cv_id, 'outer_cv_id':outer_cv_id, 'atlas':atlas, 'y_var':y_var, 'chaco_type':chaco_type, 'subset':subset,\
                    'save_models':save_models, 'results_path':results_path, 'crossval_type':crossval, 'nperms':nperms, 'null':null, 'output_path':output_path}
                run_regression(**kwargs)            
            elif lesionload_type =='all':
                atlas = 'lesionload_all'
                model_tested = ['ridge_nofeatselect']
                chaco_type ='NA'
                kwargs = {'x':lesionload, 'Y':Y, 'group':site,  'model_tested':model_tested,'inner_cv_id':inner_cv_id, 'outer_cv_id':outer_cv_id, 'atlas':atlas, 'y_var':y_var, 'chaco_type':chaco_type, 'subset':subset,\
                    'save_models':save_models, 'results_path':results_path, 'crossval_type':crossval, 'nperms':nperms, 'null':null, 'output_path':output_path}
                run_regression(**kwargs) 
            elif lesionload_type =='all_2h':
                atlas = 'lesionload_all_2h'
                model_tested = ['ridge_nofeatselect']
                chaco_type ='NA'
                kwargs = {'x':lesionload, 'Y':Y, 'group':site,  'model_tested':model_tested,'inner_cv_id':inner_cv_id, 'outer_cv_id':outer_cv_id, 'atlas':atlas, 'y_var':y_var, 'chaco_type':chaco_type, 'subset':subset,\
                    'save_models':save_models, 'results_path':results_path, 'crossval_type':crossval, 'nperms':nperms, 'null':null, 'output_path':output_path}
                run_regression(**kwargs) 
            elif lesionload_type =='slnm':
                atlas = 'lesionload_slnm'
                model_tested = ['linear_regression']
                chaco_type ='NA'
                kwargs = {'x':lesionload, 'Y':Y, 'group':site,  'model_tested':model_tested,'inner_cv_id':inner_cv_id, 'outer_cv_id':outer_cv_id, 'atlas':atlas, 'y_var':y_var, 'chaco_type':chaco_type, 'subset':subset,\
                    'save_models':save_models, 'results_path':results_path, 'crossval_type':crossval, 'nperms':nperms, 'null':null, 'output_path':output_path}
                run_regression(**kwargs) 
                
        elif ensemble == 'demog':
            if lesionload_type == 'none':
                if model_tested=='ridge':
                    kwargs = {'X1':X, 'C':C, 'Y':Y, 'group':site,  'model_tested':model_tested,'inner_cv_id':inner_cv_id, 'outer_cv_id':outer_cv_id, 'atlas':atlas, 'y_var':y_var, 'chaco_type':chaco_type, 'subset':subset,\
                        'save_models':save_models, 'results_path':results_path, 'crossval_type':crossval, 'nperms':nperms, 'null':null, 'output_path':output_path}
                    run_regression_ensemble(**kwargs)            
            elif lesionload_type =='M1':
                atlas = 'lesionload_m1'
                model_tested = 'linear_regression'
                chaco_type = 'NA'
                kwargs = {'X1':lesionload, 'C':C, 'Y':Y, 'group':site, 'model_tested':model_tested, 'inner_cv_id':inner_cv_id, 'outer_cv_id':outer_cv_id, 'atlas':atlas, 'y_var':y_var, 'chaco_type':chaco_type, 'subset':subset,\
                    'save_models':save_models, 'results_path':results_path, 'crossval_type':crossval, 'nperms':nperms, 'null':null, 'output_path':output_path}
                run_regression_ensemble(**kwargs)
            elif lesionload_type =='all':
                atlas = 'lesionload_all'
                model_tested= 'ridge_nofeatselect'
                chaco_type ='NA'
                kwargs = {'X1':lesionload, 'C':C, 'Y':Y, 'group':site,  'model_tested':model_tested,'inner_cv_id':inner_cv_id, 'outer_cv_id':outer_cv_id, 'atlas':atlas, 'y_var':y_var, 'chaco_type':chaco_type, 'subset':subset,\
                    'save_models':save_models, 'results_path':results_path, 'crossval_type':crossval, 'nperms':nperms, 'null':null, 'output_path':output_path}
                run_regression_ensemble(**kwargs)
            elif lesionload_type =='all_2h':
                atlas = 'lesionload_all_2h'
                model_tested= 'ridge_nofeatselect'
                chaco_type ='NA'
                kwargs = {'X1':lesionload, 'C':C, 'Y':Y, 'group':site,  'model_tested':model_tested,'inner_cv_id':inner_cv_id, 'outer_cv_id':outer_cv_id, 'atlas':atlas, 'y_var':y_var, 'chaco_type':chaco_type, 'subset':subset,\
                    'save_models':save_models, 'results_path':results_path, 'crossval_type':crossval, 'nperms':nperms, 'null':null, 'output_path':output_path}
                run_regression_ensemble(**kwargs)
            elif lesionload_type =='slnm':
                atlas = 'lesionload_slnm'
                model_tested= 'linear_regression'
                chaco_type ='NA'
                kwargs = {'X1':lesionload, 'C':C, 'Y':Y, 'group':site,  'model_tested':model_tested,'inner_cv_id':inner_cv_id, 'outer_cv_id':outer_cv_id, 'atlas':atlas, 'y_var':y_var, 'chaco_type':chaco_type, 'subset':subset,\
                    'save_models':save_models, 'results_path':results_path, 'crossval_type':crossval, 'nperms':nperms, 'null':null, 'output_path':output_path}
                run_regression_ensemble(**kwargs)
                
        elif ensemble == 'chaco_ll':
            print('\n Running ensemble model with ChaCo scores AND lesion loads.. \n')
            if lesionload_type =='M1':
                atlas = 'lesionload_m1'
                model_tested = 'linear_regression'
                chaco_type = 'NA'
                kwargs = {'X1':lesionload, 'X2':X, 'Y':Y, 'group':site, 'model_tested':model_tested, 'inner_cv_id':inner_cv_id, 'outer_cv_id':outer_cv_id, 'atlas':atlas, 'y_var':y_var, 'chaco_type':chaco_type, 'subset':subset,\
                    'save_models':save_models, 'results_path':results_path, 'crossval_type':crossval, 'nperms':nperms, 'null':null, 'output_path':output_path, 'ensemble_atlas':ensemble_atlas}
                run_regression_chaco_ll(**kwargs)
            elif lesionload_type =='all':
                atlas = 'lesionload_all'
                model_tested= 'ridge_nofeatselect'
                chaco_type ='NA'
                kwargs = {'X1':lesionload, 'X2':X, 'Y':Y, 'group':site,  'model_tested':model_tested,'inner_cv_id':inner_cv_id, 'outer_cv_id':outer_cv_id, 'atlas':atlas, 'y_var':y_var, 'chaco_type':chaco_type, 'subset':subset,\
                    'save_models':save_models, 'results_path':results_path, 'crossval_type':crossval, 'nperms':nperms, 'null':null, 'output_path':output_path, 'ensemble_atlas':ensemble_atlas}
                run_regression_chaco_ll(**kwargs)
            elif lesionload_type =='all_2h':
                atlas = 'lesionload_all_2h'
                model_tested= 'ridge_nofeatselect'
                chaco_type ='NA'
                kwargs = {'X1':lesionload, 'X2':X, 'Y':Y, 'group':site,  'model_tested':model_tested,'inner_cv_id':inner_cv_id, 'outer_cv_id':outer_cv_id, 'atlas':atlas, 'y_var':y_var, 'chaco_type':chaco_type, 'subset':subset,\
                    'save_models':save_models, 'results_path':results_path, 'crossval_type':crossval, 'nperms':nperms, 'null':null, 'output_path':output_path, 'ensemble_atlas':ensemble_atlas}
                run_regression_chaco_ll(**kwargs)
                
                
        elif ensemble == 'chaco_ll_demog':
            print('\n Running ensemble model with ChaCo scores AND lesion loads AND demographics.. \n')
            if lesionload_type =='M1':
                atlas = 'lesionload_m1'
                model_tested = 'linear_regression'
                chaco_type = 'NA'
                kwargs = {'X1':lesionload, 'X2':X, 'C':C, 'Y':Y, 'group':site, 'model_tested':model_tested, 'inner_cv_id':inner_cv_id, 'outer_cv_id':outer_cv_id, 'atlas':atlas, 'y_var':y_var, 'chaco_type':chaco_type, 'subset':subset,\
                    'save_models':save_models, 'results_path':results_path, 'crossval_type':crossval, 'nperms':nperms, 'null':null, 'output_path':output_path, 'ensemble_atlas':ensemble_atlas}
                run_regression_chaco_ll_demog(**kwargs)
            elif lesionload_type =='all':
                atlas = 'lesionload_all'
                model_tested= 'ridge_nofeatselect'
                chaco_type ='NA'
                kwargs = {'X1':lesionload, 'X2':X, 'C':C, 'Y':Y, 'group':site,  'model_tested':model_tested,'inner_cv_id':inner_cv_id, 'outer_cv_id':outer_cv_id, 'atlas':atlas, 'y_var':y_var, 'chaco_type':chaco_type, 'subset':subset,\
                    'save_models':save_models, 'results_path':results_path, 'crossval_type':crossval, 'nperms':nperms, 'null':null, 'output_path':output_path, 'ensemble_atlas':ensemble_atlas}
                run_regression_chaco_ll_demog(**kwargs)
            elif lesionload_type =='all_2h':
                atlas = 'lesionload_all_2h'
                model_tested= 'ridge_nofeatselect'
                chaco_type ='NA'
                kwargs = {'X1':lesionload, 'X2':X, 'C':C, 'Y':Y, 'group':site,  'model_tested':model_tested,'inner_cv_id':inner_cv_id, 'outer_cv_id':outer_cv_id, 'atlas':atlas, 'y_var':y_var, 'chaco_type':chaco_type, 'subset':subset,\
                    'save_models':save_models, 'results_path':results_path, 'crossval_type':crossval, 'nperms':nperms, 'null':null, 'output_path':output_path, 'ensemble_atlas':ensemble_atlas}
                run_regression_chaco_ll_demog(**kwargs)

          
def save_model_outputs(results_path, output_path, atlas, y_var, chaco_type, subset, model_tested, crossval, nperms, ensemble,n_outer_folds,ensemble_atlas):
    # This function is a helper function for saving the outputs of a machine learning model. It takes a number of 
    # inputs, including results_path, output_path, atlas, y_var, chaco_type, subset, model_tested, crossval, nperms, and ensemble. 
    # The function then sets up a number of arrays to store the outputs of the model, and then saves those outputs to files in the 
    # specified directory. The function is able to handle saving the outputs of different types of models, such as 
    # those using different atlases, cross-validation methods, and so on.
    
    logprint('Saving model outputs to directory: {}'.format(os.path.join(results_path, output_path)))
    
    mdl_label = model_tested

    rootname = os.path.join(results_path, output_path,'{}_{}_{}_{}_{}_crossval{}'.format(atlas, y_var, chaco_type, subset, mdl_label,crossval))

    r2scores_allperms=np.zeros(shape=(nperms, n_outer_folds))
    correlation_allperms=np.zeros(shape=(nperms, n_outer_folds))
    
    if atlas == 'lesionload_all':
        varimpts_allperms = np.zeros(shape=(nperms, 6))
        mean_betas_allperms = np.zeros(shape=(nperms, 6))
        std_betas_allperms = np.zeros(shape=(nperms,6))
        betas_allperms = np.zeros(shape=(nperms,6))
    if atlas == 'lesionload_all_2h':
        varimpts_allperms = np.zeros(shape=(nperms, 12))
        mean_betas_allperms = np.zeros(shape=(nperms, 12))
        std_betas_allperms = np.zeros(shape=(nperms,12))
        betas_allperms = np.zeros(shape=(nperms,12))
    if chaco_type=='chacoconn':
        if atlas == 'fs86subj':
            varimpts_allperms = np.empty(shape=(0, 86, 86))
        if atlas == 'shen268':
            varimpts_allperms = np.empty(shape=(0, 268, 268))
    elif chaco_type=='chacovol':
        if atlas == 'fs86subj':
            varimpts_allperms = np.empty(shape=(0, 86))
            betas_allperms = np.zeros(shape=(0,86))

        if atlas == 'shen268':
            varimpts_allperms = np.empty(shape=(0, 268))
            betas_allperms = np.zeros(shape=(0,268))

        
        
    correlation_allperms=np.zeros(shape=(nperms, n_outer_folds))


    for n in range(0, nperms): #
        # if ensemble model was run, the filename is different because i'm a silly billy. catch it here. 
        # don't care about feature weights for demographic information, and any lesion feature weights are the same as no-ensemble models.
        if ensemble =='demog':
            
            r2scores_ensemble=np.load(rootname +'_perm'+ str(n) + '_ensemble_demog'+ '_scores.npy',allow_pickle=True)
            correlation_ensemble = np.load(rootname +'_perm'+ str(n) + '_ensemble_demog'+ '_correlations_ensemble.npy',allow_pickle=True)
            #varimpts_ensemble=np.load(rootname +'_perm'+ str(n) +  '_ensemble'+ '_activation_weights.npy',allow_pickle=True)
            #mdl=np.load(rootname +'_perm'+ str(n) + '_ensemble'+  '_model.npy',allow_pickle=True)
            r2scores_allperms[n,] = r2scores_ensemble
            correlation_allperms[n,] = correlation_ensemble
            
        if ensemble =='chaco_ll':
            
            r2scores_ensemble=np.load(rootname +'_perm'+ str(n) + '_ensemble_chacoLL_' + ensemble_atlas + '_scores.npy',allow_pickle=True)
            correlation_ensemble = np.load(rootname +'_perm'+ str(n) + '_ensemble_chacoLL_' + ensemble_atlas + '_correlations_ensemble.npy',allow_pickle=True)
            #varimpts_ensemble=np.load(rootname +'_perm'+ str(n) +  '_ensemble'+ '_activation_weights.npy',allow_pickle=True)
            #mdl=np.load(rootname +'_perm'+ str(n) + '_ensemble'+  '_model.npy',allow_pickle=True)
            r2scores_allperms[n,] = r2scores_ensemble
            correlation_allperms[n,] = correlation_ensemble
        if ensemble =='chaco_ll_demog':
            
            r2scores_ensemble=np.load(rootname +'_perm'+ str(n) + '_ensemble_chacoLLdemog_' + ensemble_atlas + '_scores.npy',allow_pickle=True)
            correlation_ensemble = np.load(rootname +'_perm'+ str(n) + '_ensemble_chacoLLdemog_' + ensemble_atlas + '_correlations_ensemble.npy',allow_pickle=True)
            #varimpts_ensemble=np.load(rootname +'_perm'+ str(n) +  '_ensemble'+ '_activation_weights.npy',allow_pickle=True)
            #mdl=np.load(rootname +'_perm'+ str(n) + '_ensemble'+  '_model.npy',allow_pickle=True)
            r2scores_allperms[n,] = r2scores_ensemble
            correlation_allperms[n,] = correlation_ensemble
            
        # no ensemble model.
        if ensemble =='none':
            r2scores=np.load(rootname +'_perm'+ str(n) + '_scores.npy',allow_pickle=True)
            correlation = np.load(rootname +'_perm'+ str(n) +'_correlations.npy',allow_pickle=True)
            betas=np.load(rootname +'_perm'+ str(n) + '_beta_coeffs.npy',allow_pickle=True)

            if atlas == 'lesionload_all' or atlas=='lesionload_all_2h':
                # bc there is no feature selection, we can average the weights of the lesioad load CSTs together together.
                
                mean_betas_allperms[n,]=np.median(betas,axis=0)
                std_betas_allperms[n,]=np.std(betas,axis=0)
                betas_allperms[n,]=np.median(betas,axis=0)
                
            if atlas == 'fs86subj' or atlas == 'shen268':
                # there is feature selection. so let's concatenate the outer loop features together and only look at features that are included in >50% of the outer folds

                
                betas_allperms=np.concatenate((betas_allperms,betas),axis=0)

            mdl=np.load(rootname +'_perm'+ str(n) + '_model.npy',allow_pickle=True)
            
            r2scores_allperms[n,] = r2scores
            correlation_allperms[n,] = correlation
            
            if mdl_label == 'ridge':
                alphas=[]
                feats=[]
                
                for outer_fold in range(0,5):
                    alphas.append(mdl[0][outer_fold][mdl_label].alpha)
                    feats.append(mdl[0][outer_fold]['featselect'].k) 
                    
                np.savetxt(rootname +'_perm'+ str(n)  +'_alphas.txt', alphas)
                np.savetxt(rootname +'_perm'+ str(n)  +'_nfeats.txt', feats)
                
        
    #after data from all permutations collected
    # save the average feature weight from features included in 50%, 90%, or 99% of outer folds.
    if atlas == 'fs86subj' or atlas == 'shen268':
        
        n_outer_folds_total = nperms*n_outer_folds # 500 for k=5 and nperm=100
        
        threshold_50 = n_outer_folds_total-n_outer_folds_total/2 # 50%
        threshold_90 = n_outer_folds_total-n_outer_folds_total/10 # 90%
        threshold_99 = n_outer_folds_total-n_outer_folds_total/100 # 99%
        nonzero_outerfolds = np.count_nonzero(betas_allperms,axis=0)
        print(nonzero_outerfolds)
        mean_betas_allperms = np.median(betas_allperms,axis=0)
        mean_betas_allperms_0 = mean_betas_allperms
        print((nonzero_outerfolds > threshold_50))
        mean_betas_allperms_50 = mean_betas_allperms*(nonzero_outerfolds > threshold_50)
        mean_betas_allperms_90 = mean_betas_allperms*(nonzero_outerfolds > threshold_90)
        mean_betas_allperms_99 = mean_betas_allperms*(nonzero_outerfolds > threshold_99)
        print(mean_betas_allperms_99)
        np.savetxt(rootname +'_meanbetas_allperms_0.txt', mean_betas_allperms_0)
        np.savetxt(rootname +'_meanbetas_allperms_50.txt', mean_betas_allperms_50)
        np.savetxt(rootname +'_meanbetas_allperms_90.txt', mean_betas_allperms_90)
        np.savetxt(rootname +'_meanbetas_allperms_99.txt', mean_betas_allperms_99)
        
    if atlas =='lesionload_all' or atlas =='lesionload_all_2h':
        np.savetxt(rootname +'_meanfeatureweight_allperms.txt', np.median(varimpts_allperms,axis=0))   
        np.savetxt(rootname +'_meanbetas_allperms.txt', np.median(mean_betas_allperms,axis=0))   
        np.savetxt(rootname +'_stdbetas_allpearms.txt', np.median(std_betas_allperms,axis=0))   
        np.savetxt(rootname +'_betas.txt', betas_allperms)

    np.savetxt(rootname + '_r2_scores.txt', r2scores_allperms)
    np.savetxt(rootname +'_correlations.txt', correlation_allperms)   
          
    return r2scores_allperms, correlation_allperms

def logprint(string):
    # This function is a helper function for logging. It takes a string input, string, 
    # and then prints that string to the console and writes it to a log file using the logging module.
    # This allows the user to keep track of what the program is doing and any important messages it outputs.
    
    print(string)
    logging.info(string)
 
 
def check_if_files_exist_already(crossval,model_tested,atlas,chaco_type,results_path, ensemble, y_var, subset,ensemble_atlas):
    # The function then searches through the specified results_path directory to see if any files with the specified parameters 
    # already exist. If it finds any such files, it returns True and the path to the directory where the files were found. 
    # If no such files are found, it returns False and the results_path directory. 
    
    # This function is used to avoid running the same machine learning model multiple times and overwriting the output files.
    
    # get list of subdirectories:
    subfolders = glob.glob(os.path.join(results_path, 'analysis_*'),recursive = False)
    print('\n')
    for folder in subfolders:
        if ensemble == 'demog':
            filename = os.path.join(folder, '{}_{}_{}_{}_{}_crossval{}_perm99_ensemble_demog_scores.npy'.format(atlas, y_var, chaco_type, subset, model_tested,crossval))

            if os.path.exists(filename):
                print('Files already exist in folder {}!'.format(folder))
                return True, folder
            
        elif ensemble == 'chaco_ll':
            filename = os.path.join(folder, '{}_{}_{}_{}_{}_crossval{}_perm99_ensemble_chacoLL_{}_scores.npy'.format(atlas, y_var, chaco_type, subset, model_tested,crossval, ensemble_atlas))

            if os.path.exists(filename):
                print('Files already exist in folder {}!'.format(folder))
                return True, folder
            
        elif ensemble == 'chaco_ll_demog':
            filename = os.path.join(folder, '{}_{}_{}_{}_{}_crossval{}_perm99_ensemble_chacoLLdemog_{}_scores.npy'.format(atlas, y_var, chaco_type, subset, model_tested,crossval, ensemble_atlas))

            if os.path.exists(filename):
                print('Files already exist in folder {}!'.format(folder))
                return True, folder
        else:
            filename = os.path.join(folder, '{}_{}_{}_{}_{}_crossval{}_perm99_beta_coeffs.npy'.format(atlas, y_var, chaco_type, subset, model_tested,crossval))
            
            if os.path.exists(filename):
                print('File already exists in folder: {}'.format(folder))
                return True, folder 
             
   
    return False, results_path


def announce_runningmodel(lesionload_type, ensemble, atlas, chaco_type, crossval, override_rerunmodels):
    logprint('~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ \n ')
    logprint('~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ Running machine learning model: ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ \n ')
    logprint('~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ \n ')
    
    logprint('Running machine learning model: \n')
    logprint('lesionload type: {}'.format(lesionload_type))
    logprint('ensemble type: {}'.format(ensemble))
    logprint('atlas type: {}'.format(atlas))
    logprint('chacotype: {}'.format(chaco_type))
    logprint('crossval type: {}'.format(crossval))
    print('override rerunmodels: {}'.format(override_rerunmodels))