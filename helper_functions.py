import sys; sys.path
import pandas as pd
import numpy as np 
import seaborn as sns
import scipy.io as sio
from scipy.stats import pearsonr
import os
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold, StratifiedKFold
from sklearn import preprocessing, linear_model
from sklearn.metrics import explained_variance_score, r2_score, make_scorer
from sklearn.preprocessing import StandardScaler
from joblib import parallel_backend, Parallel, delayed
import matplotlib
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest,f_regression
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import warnings
import math
from functools import partial

warnings.filterwarnings("ignore", message="RuntimeWarning: invalid value encountered in true_divide corr /= X_norms")


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
    print("Final size of X: " + str(X.shape))
    
    return X

def prepare_image_data(X):
    '''Clean X-data (remove zero-value input variables)'''

    # remove inputs that are 0 for all subjects
    X=np.reshape(X, (101,902629))
    print("Final size of X: " + str(X.shape))
    
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
    print(x.shape)
    print(y.shape)
    xv = x - x.mean(axis=0)
    yv = y - y.mean(axis=0)
    xvss = (xv * xv).sum(axis=0)
    yvss = (yv * yv).sum(axis=0)
    result = np.matmul(xv.transpose(), yv) / np.sqrt(np.outer(xvss, yvss))
    
    # bound the values to -1 to 1 in the event of precision issues
    return abs(np.maximum(np.minimum(result, 1.0), -1.0))


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


def feature_select_PCA(x_train, x_test, a):
    """Return values for the top a PCs of x based on PCA of x.
         
         Inputs:
             x = input matrix
             y = variable/s predicted 
             a = number of PCs
         
        Returns:
            x_train_featselect = training data with selected features 
            x_test_featselect = test data matrix with selected features 
            var_expl = variance explained by top a components
            components = PCs selected"""
    
    # check that dimension of x is greater than a
    if x_train.shape[1]<a:
        raise Exception('Number of features in X is less than the number of features specified to retain (a).') 
    
    # Feature selection: use only the top n features based on top a PCs in training data 
    pca = PCA(n_components=a, copy=True, random_state=42)
    x_train_featselect = pca.fit(x_train).transform(x_train)
    x_test_featselect = pca.transform(x_test)
    components = pca.components_

    
    var_expl = pca.explained_variance_


    return x_train_featselect,x_test_featselect, components

def feature_select_correlation(x_train, x_test, y, a):
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


def scale_data(x_train, x_test):
    '''Scale the training data and apply transformation to the test/validation data.

        Inputs:
            x_train = training predictors
            x_test = training predictors 
        
        Returns:
            x_train_scaled
            x_test_scaled '''
    
    # Scale x_train 
    scaler = preprocessing.StandardScaler().fit(x_train)
    
    # apply transformation to train & test set.
    x_train_scaled = scaler.transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    
    return x_train_scaled, x_test_scaled

def normalize_data(x_train, x_test):
    '''subtracting the mean and dividing by the l2-norm'''
    x_train_mean=np.mean(x_train)
    x_train_norm=np.linalg.norm(x_train)
    
    x_test_mean=np.mean(x_test)
    x_test_norm=np.linalg.norm(x_test)
    
    x_train=(x_train-x_train_mean)/x_train_norm
    x_test=(x_test-x_test_mean)/x_test_norm

    return x_train, x_test

def gcv_ridge(hyperparam, x, y, k, featsel='None', a=10):
    """Perform gridsearch using k-fold cross-validation on a single hyperparameter 
    in ridge regression, and return mean R^2 across inner folds.
    
    Inputs: 
        hyperparam = list of hyperparameter values to train & test on validation est
        x = N x p input matrix
        y = 1 x p variable to predict
        k = k value in k-fold cross validation 
        featsel = type string, feature selection method, default="None"
            'None' - no feature selection; use all variables for prediction
            'correlation'- calculate the abs. val. Pearson correlation between all training variables with the varibale to predict. Use the highest 'a' variables based on their correlation for prediction
            'PCA' - perform PCA on the training variables and use the top 'a' PCs as input variables, by variance explained, for prediction
        a = number of features to select using one of the above methods, default=10 
    
    Returns:
        expl_var = the mean R^2 (coefficient of determination) across inner loop folds for the given hyperparameter
    """
    
    # make sure k is reasonable 
    if x.shape[0]/k <= 2:
        raise Exception('R^2 is not well-defined with less than 2 subjects.')   
    
    # set alpha in ridge regression
    alpha = hyperparam

    comp = []
    explvar=[]
    
    # Split data into test and train: random state fixed for reproducibility
    kf = KFold(n_splits=k,shuffle=True,random_state=43)
    
    # K-fold cross-validation 
    for train_index, valid_index in kf.split(x):
        x_train, x_valid = x[train_index], x[valid_index]
        y_train, y_valid = y[train_index], y[valid_index]
                
        if featsel=='correlation':
            x_train, x_valid, ind = feature_select_correlation(x_train, x_valid, y_train, a)
            
        elif featsel=='PCA':
            x_train, x_valid, components = feature_select_PCA(x_train, x_valid, a)
        
        # Fit ridge regression with (x_train_scaled, y_train), and predict x_train_scaled
        regr = linear_model.Ridge(alpha=alpha, normalize=True, max_iter=1000000, random_state=42)
        y_pred = regr.fit(x_train, y_train).predict(x_valid)
        explvar.append(explained_variance_score(y_valid, y_pred))
    
    # use explained_variance_score instead:
    expl_var=np.mean(explvar)
    
    return expl_var


def parallel_featsearch(alpha, X, Y, k, featselect, feat):
    ''' Grid search in parallel.

    Returns:
        expl_var - explained variance for given combination of alpha/feat/featselect'''
    
    expl_var=gcv_ridge(alpha, X, Y, k, featselect, feat)
    return expl_var

# Inner loop - grid search
def gridsearch_cv(k, x, y, featselect, alphas, feats):
    ''' Performs grid search using fixed predefined hyperparameter ranges and returns the best alpha and # of features for 
        a given training/validation sample. 
        
        Input:
            x = N x p input matrix
            y = 1 x p variable to predict
            k = k value in k-fold cross validation 
            featsel = type string, feature selection method, default="None"
                'None' - no feature selection; use all variables for prediction
                'correlation'- calculate the abs. val. Pearson correlation between all training variables with the varibale to predict. Use the highest 'a' variables based on their correlation for prediction
                'PCA' - perform PCA on the training variables and use the top 'a' PCs as input variables, by variance explained, for prediction
            alphas - range of alpha parameters to search 
            feats - range of # features to search
        
        Returns:
            bestalpha - optimal alpha based on grid search
            bestfeats - optimal number of features based on grid search
            bestr2 - mean R^2 across folds obtained for the optimal combination of hyperparameters
            gcv_values - R^2 across all combinations of hyperparametrs.'''
    
    print(str(k)+"-fold cross-validation results in "+str((x.shape[0]/k)*(k-1))+ " subjects in the training set, and "+ str(x.shape[0]/k) + " subjects in the validation set")

    # initialize array to store r2
    gcv_values=np.empty(shape=(len(alphas),len(feats)),dtype='float')

    # iterate through alphas
    for alpha in alphas:
        row, = np.where(alphas==alpha)
        
        # run feature selection (# of components) in parallel
        gcv=Parallel(n_jobs=-1,verbose=0)(delayed(parallel_featsearch)(alpha,x, y, k, featselect, feat) for feat in feats)
        gcv=np.array(gcv)
        gcv_values[row]=gcv
            
        row=np.argmax(np.max(gcv_values, axis=1))
        col=np.argmax(np.max(gcv_values, axis=0))

    bestalpha=alphas[row]
    bestfeats=feats[col]
    bestr2=np.max(gcv_values)

    return bestalpha, bestfeats, bestr2, gcv_values

def plot_figure(gcv_values, string, midpoint):
    '''Plots the R^2 value obtained across all grid-search pairs (# features and regularization values.)
    
    Inputs:
        gcv_values - matrix to plot
        string - title
        midpoint - point at which blue turns to red.'''
    
    plt.figure(figsize=(17,14))
    shifted_cmap = shiftedColorMap(plt.get_cmap('bwr'), midpoint=midpoint, name='shifted')

    plt.imshow(gcv_values, cmap=shifted_cmap, interpolation='nearest')

    plt.xlabel('# Features', fontsize=15, fontweight='bold')
    plt.ylabel('Alphas', fontsize=15, fontweight='bold')

    row=np.argmax(np.max(gcv_values, axis=0))
    col=np.argmax(np.max(gcv_values, axis=1))

    ax = plt.axes()

    ax.xaxis.set_major_locator(plt.MultipleLocator(1))
    ax.yaxis.set_major_locator(plt.MultipleLocator(1))

    xticks= np.linspace(feat_start, feat_end,n_feats, dtype=int)
    yticks= np.linspace(alpha_start, alpha_end, n_alphas,dtype=None)

    plt.xticks(np.arange(len(feats)), fontsize=18)
    plt.yticks(np.arange(len(alphas)), fontsize=18)

    ax.set_xticklabels(xticks)
    ax.set_yticklabels(np.round(yticks,3))

    #ax.xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{xticks: .2f}'))
    cbar = plt.colorbar()
    cbar.set_label('R^2')
    plt.title(string + '_Best R^2 = ' + str(np.round(np.max(gcv_values), 3)), fontsize=20)

    plt.scatter(row,col,color='k')
    plt.savefig(results_dir+string+ '.png')
    plt.show()
     
     
def determine_featselect_range(X):
    X_max_dim = X.shape[1]
    X_min_dim = 5
    n=30
    base = 2
    k_range=np.logspace(math.log(X_min_dim,base),math.log(X_max_dim,base), n,base=base, dtype=int)
    return k_range

def get_models(model_type='regression', model_list=None):
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
        mdls_labels = model_list
        
        return mdls, mdls_labels
        
    elif model_type == 'classification': 
        svm = Pipeline([('svm', SVC(probability=True, class_weight='balanced', kernel='linear', random_state=0))])
        rbf_svm = Pipeline([('svm', SVC(probability=True, class_weight='balanced', kernel='rbf', random_state=0))])
        log = Pipeline([('logistic', LogisticRegression(class_weight='balanced', solver='liblinear', random_state=0))])
        pca_log = Pipeline([('pca', PCA(n_components=0.9)),
                            ('logistic', LogisticRegression(class_weight='balanced', solver='liblinear', random_state=0))])

        mdls = [svm, rbf_svm, log, pca_log]
        mdls_labels = ['svm', 'rbf_svm', 'log', 'pca_log']
        
    return mdls, mdls_labels

def inner_loop(mdl, mdl_label, X, Y, cv, n_jobs):
    
    k_range = determine_featselect_range(X)
    
    if mdl_label=='ridge':
        grid_params ={'ridge__alpha': np.logspace(-2, 2, 30, base=10,dtype=None),
                      'featselect__k':k_range}
        score = 'explained_variance'
        
    elif mdl_label=='elastic_net':
        grid_params ={'elastic_net__alpha': np.logspace(-2, 2, 30, base=10,dtype=None),
                      'featselect__k':k_range}
        score = 'explained_variance'
        
    elif mdl_label=='lasso':
        grid_params ={'lasso__alpha': np.logspace(-2, 2, 30, base=10,dtype=None),
                      'featselect__k':k_range}
        score = 'explained_variance'
        
    elif mdl_label == 'rfc':
        grid_params = {'rfc__n_estimators': np.linspace(10, 200, 5, dtype=int),  # Should be as HIGH as possible
                       'rfc__max_features': ['sqrt', 'log2', 0.25, 0.5, 0.75]}
    elif 'svm' in mdl_label:
        grid_params = {'svm__C': [0.0001, 0.001, 0.01, 0.1, 1]}
        if 'rbf' in mdl_label:
            grid_params = {'svm__C': [0.0001, 0.001, 0.01, 0.1, 1],
                           'svm__gamma': [0.001, 0.01, 0.1, 1]}
    elif 'log' in mdl_label:
        grid_params = {'logistic__C': [0.0001, 0.001, 0.01, 0.1, 1, 10],
                       'logistic__penalty': ['l1', 'l2']}
    else:
        print('Model not found..')
        return mdl, None
    
    grid_search = GridSearchCV(estimator=mdl, param_grid=grid_params, scoring=score, cv=cv, refit=True, verbose=0,
                               n_jobs=n_jobs, return_train_score=False, pre_dispatch='2*n_jobs')
    
    
    grid_search.fit(X, Y)

    best_mdl = grid_search.best_estimator_
    return best_mdl

