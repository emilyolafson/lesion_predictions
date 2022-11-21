from stat import FILE_ATTRIBUTE_SPARSE_FILE
import sys; sys.path
import numpy as np 
from scipy.stats import pearsonr
import os
from sklearn.model_selection import GridSearchCV, KFold
from sklearn import preprocessing, linear_model
from sklearn.metrics import explained_variance_score,accuracy_score,roc_auc_score,mean_absolute_error
import matplotlib
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Lasso, Ridge, ElasticNet,LinearRegression,LogisticRegression
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest,f_regression
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import GroupShuffleSplit,ShuffleSplit,GroupKFold, LeaveOneGroupOut, KFold
import logging
import math
from sklearn.preprocessing import LabelEncoder, OneHotEncoder 
from sklearn.ensemble import RandomForestClassifier
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
from functools import partial
import warnings
warnings.filterwarnings('ignore') 


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
def run_classification(X, Y, group, inner_cv, outer_cv, models_tested, atlas, y_var, chaco_type, subset, save_models,results_path,crossval_type,nperms,null):
    
    outer_cv_splits = outer_cv.get_n_splits(X, Y, group)
 
    models = np.zeros((len(models_tested), outer_cv_splits), dtype=object)
    balanced_accuracies  = np.zeros((len(models_tested),outer_cv_splits), dtype=object)
    variable_importance  = []
    size_testgroup =[]

    for n in range(0,nperms):
        
        logprint('PERMUTATION: {}'.format(n))
        
        for cv_fold, (train_id, test_id) in enumerate(outer_cv.split(X, Y, group)):

            logprint("Fold: {}".format(cv_fold + 1))
            
            X_train, X_test = X[train_id], X[test_id]
            y_train, y_test = Y[train_id], Y[test_id]
            group_train, group_test = group[train_id], group[test_id]
            
            logprint('Size of test group: {}'.format(group_test.shape[0]))
            logprint('Size of train group: {}'.format(group_train.shape[0]))

            mdls, mdls_labels = get_models('classification', models_tested)
            size_testgroup.append(group_test.shape[0])
            mdl_idx=0
            for mdl, mdl_label in zip(mdls, mdls_labels): 
                filename = results_path + '/{}_{}_{}_{}_{}_crossval{}_{}'.format(atlas, y_var, chaco_type, subset, mdl_label,crossval_type,n)
        
                if null>0:
                    logprint('NULL!')
                    filename = filename + '_null_' + str(null)
                    
                logprint('-------------- saving file as: {} -------------'.format(filename))
                
                logprint('Performing grid search for: {} \n'.format(mdl_label))
                mdl = inner_loop(mdl, mdl_label, X_train, y_train, group_train, inner_cv, 10)  
                mdl.fit(X_train, y_train)
                y_pred= mdl.predict(X_test)
                
                y_score = mdl.predict_proba(X_test)[:, 1]
                accuracies =  accuracy_score(y_test, y_pred)
                logprint(accuracies)
                
                bacc=balanced_accuracy(y_test,y_pred)
                ppvscore = ppv(y_test,y_pred)
                npvscore = npv(y_test,y_pred)
                auc = roc_auc_score(y_test,y_pred)
                #variable_importance.append(mdl.named_steps[mdl_label].coef_)
                #correlations[mdl_idx, cv_fold] = np_pearson_cor(y_test,y_pred)[0]
                
                logprint('Accuracy: {} \n'.format(accuracies))
                logprint('Balanced accuracy: {} \n'.format(bacc))
                logprint('PPV: {}'.format(ppvscore))
                logprint('NPV: {}'.format(npvscore))
                logprint('AUC: {}'.format(auc))

                #logprint('Correlation: {} \n'.format(np_pearson_cor(y_test,y_pred)))

                balanced_accuracies[mdl_idx, cv_fold]=bacc
                
                if save_models:
                    models[mdl_idx, cv_fold] = mdl
                    
                mdl_idx += 1

        logprint("Saving data...")
        #np.save(os.path.join(results_path, filename + "_scores.npy"), balanced_accuracies)
        np.save(os.path.join(results_path, filename + "_model.npy"), models)
       # np.save(os.path.join(results_path, filename + "_correlations.npy"), correlations)

        np.save(os.path.join(results_path, filename + "_model_labels.npy"), mdls_labels)
       # np.save(os.path.join(results_path, filename + "_variable_impts.npy"), variable_importance)
        np.save(os.path.join(results_path, filename + "_test_group_sizes.npy"), size_testgroup)
        
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
    n = 30
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
    elif mdl_label == 'ridge_nofeatselect':
        return mdl
    else:
        logprint('Performing grid search for: {} \n'.format(mdl_label))

        grid_search = GridSearchCV(estimator=mdl, param_grid=grid_params, scoring=score, cv=inner_cv, refit=True, verbose=1,
                                n_jobs=n_jobs, return_train_score=False, pre_dispatch='2*n_jobs')

    grid_search.fit(X, Y, group)
    best_mdl = grid_search.best_estimator_
    return best_mdl

def haufe_transform_results(X_train, y_train, cols, mdl, mdl_label, chaco_type, atlas, X):
    cov_y=np.cov(np.transpose(y_train))
    x_sub = X_train[:,cols]

    cov_x=np.cov(np.transpose(x_sub))
    activationweight = mdl.named_steps[mdl_label].coef_
    weight=np.transpose(activationweight)
    
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
    return activation

def create_outer_cv(outer_cv_id):
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
    
def run_regression(x, Y, group, inner_cv_id, outer_cv_id, models_tested, atlas, y_var, chaco_type, subset, save_models,results_path,crossval_type,nperms,null, output_path):

    if atlas =='lesionload_m1':
        X=np.array(x).reshape(-1,1)
    elif atlas == 'lesionload_all':
        X=np.array(x)
    else:
        X = prepare_data(x) 
        
    logprint(X.shape)
    
    outer_cv = create_outer_cv(outer_cv_id)
    
    outer_cv_splits = outer_cv.get_n_splits(X, Y, group)
    
    models = np.zeros((len(models_tested), outer_cv_splits), dtype=object)
    explained_var  = np.zeros((len(models_tested),outer_cv_splits), dtype=object)
    variable_importance  = []
    correlations  = np.zeros((len(models_tested),outer_cv_splits), dtype=object)
    size_testgroup =[]

    for n in range(0,nperms):
        
        logprint('\n\n~ ~ ~ ~ ~ ~ ~ ~ ~ ~ PERMUTATION: {}/{} ~ ~ ~ ~ ~ ~ ~ ~ ~ \n\n'.format(n, nperms))
        
        activation_weights=[]
        
        for cv_fold, (train_id, test_id) in enumerate(outer_cv.split(X, Y, group)):
            
            inner_cv = create_inner_cv(inner_cv_id,n)
            logprint("------ Outer Fold: {}/{} ------".format(cv_fold + 1, outer_cv_splits))
            
            X_train, X_test = X[train_id], X[test_id]
            y_train, y_test = Y[train_id], Y[test_id]
            group_train, group_test = group[train_id], group[test_id]
            
            logprint('Size of test group: {}'.format(group_test.shape[0]))
            logprint('Size of train group: {}'.format(group_train.shape[0]))
            
            logprint('Number of sites in test set: {}'.format(np.unique(group_test).shape[0]))            
            mdls, mdls_labels = get_models('regression', models_tested) 

            size_testgroup.append(group_test.shape[0])
            
            mdl_idx=0
            
            for mdl, mdl_label in zip(mdls, mdls_labels): 

                mdl = inner_loop(mdl, mdl_label, X_train, y_train, group_train, inner_cv, 10)  

                mdl.fit(X_train, y_train)
                
                if models_tested[0] == 'ridge':
                    cols = mdl['featselect'].get_support(indices=True)

                    ## HAUFE TRANSFORMS:
                    activation = haufe_transform_results(X_train, y_train, cols, mdl, mdl_label, chaco_type, atlas, x)
                    activation_weights.append(activation)

                y_pred= mdl.predict(X_test)
                
                expl=explained_variance_score(y_test, y_pred)
                filename =  '/{}_{}_{}_{}_{}_crossval{}_perm{}'.format(atlas, y_var, chaco_type, subset, mdl_label,crossval_type,n)
                filename_plot = results_path + output_path + '/{}_{}_{}_{}_{}_crossval{}_perm{}_outerfold{}'.format(atlas, y_var, chaco_type, subset, mdl_label,crossval_type,n, cv_fold)

                save_plots_true_pred(y_test,y_pred,filename_plot, np_pearson_cor(y_test,y_pred)[0] )
                
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

        logprint('-------------- saving file w root name: {} -------------'.format(results_path + output_path + filename))
        logprint('\n\n')
        print(results_path+ output_path+ filename + "_scores.npy")
        np.save(results_path+ output_path +filename + "_scores.npy", explained_var)
        np.save(results_path+output_path+ filename + "_model.npy", models)
        np.save(results_path+output_path+ filename + "_correlations.npy", correlations)
        if models_tested != 'linear_regression':
            np.save(results_path,output_path+filename + "_activation_weights.npy", activation_weights)

        np.save(results_path+output_path+ filename + "_model_labels.npy", mdls_labels)
        np.save(results_path+output_path+filename + "_variable_impts.npy", variable_importance)
        np.save(results_path+ output_path+filename + "_test_group_sizes.npy", size_testgroup)

def run_regression_ensemble(X1, C, Y, group, inner_cv_id, outer_cv_id, models_tested, atlas, y_var, chaco_type, subset, save_models,results_path,crossval_type,nperms,null,output_path):
    X2 = C
    
    logprint('\nRunning ensemble model!')
    
    if atlas =='lesionload_m1':
        X1=np.array(X1).reshape(-1,1)
    elif atlas == 'lesionload_all':
        X1=np.array(X1)
    else:
        X1 = prepare_data(X1) 
        
    logprint(X1.shape)
    outer_cv = create_outer_cv(outer_cv_id)

    outer_cv_splits = outer_cv.get_n_splits(X1, Y, group)
    
    models = np.zeros((len(models_tested), outer_cv_splits), dtype=object)
    explained_var  = np.zeros((len(models_tested),outer_cv_splits), dtype=object)
    explained_var_lesion  = np.zeros((len(models_tested),outer_cv_splits), dtype=object)
    explained_var_demog  = np.zeros((len(models_tested),outer_cv_splits), dtype=object)

    variable_importance  = []
    correlations_ensemble  = np.zeros((len(models_tested),outer_cv_splits), dtype=object)
    correlations_lesion  = np.zeros((len(models_tested),outer_cv_splits), dtype=object)
    correlations_demog  = np.zeros((len(models_tested),outer_cv_splits), dtype=object)
    mean_abs_error = np.zeros((len(models_tested),outer_cv_splits), dtype=object)
    mean_abs_error_lesion = np.zeros((len(models_tested),outer_cv_splits), dtype=object)
    mean_abs_error_demog = np.zeros((len(models_tested),outer_cv_splits), dtype=object)

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
            mdls, mdls_labels = get_models('regression', models_tested) 
            
            # first model: X1 (lesion data)
            logprint('~~ Running model 1: lesion info ~~~')
            for mdl, mdl_label in zip(mdls, mdls_labels): 
                mdl1 = inner_loop(mdl, mdl_label, X1_train, y_train, group_train, inner_cv, 10)  
                mdl1.fit(X1_train, y_train)
                y1_pred= mdl1.predict(X1_test)
                
            logprint('~~ Running model 2: demographics ~~~')
            # second model: demographic data (unpenalized)
            mdls, mdl_labels = get_models('regression', ['linear_regression'])
            for mdl, mdl_label2 in zip(mdls, mdl_labels):
                mdl = inner_loop(mdl, mdl_label2, X2_train, y_train, group_train, inner_cv, 10)
                mdl.fit(X2_train, y_train)
                y2_pred= mdl.predict(X2_test)
            
            
            avg_pred = np.mean([y1_pred, y2_pred], axis=0)
            expl=explained_variance_score(y_test, avg_pred)
            filename = '/{}_{}_{}_{}_{}_crossval{}_perm{}_ensemble'.format(atlas, y_var, chaco_type, subset, mdl_label,crossval_type,n)
            
            filename_plot = results_path + output_path + '/{}_{}_{}_{}_{}_crossval{}_perm{}_outerfold{}_ensemble'.format(atlas, y_var, chaco_type, subset, mdl_label,crossval_type,n, cv_fold)
            save_plots_true_pred(y_test,avg_pred, filename_plot, np_pearson_cor(y_test,avg_pred)[0] )

            #variable_importance.append(mdl.named_steps[mdl_label].coef_)
            correlations_ensemble[mdl_idx, cv_fold] = np_pearson_cor(y_test,avg_pred)[0]
            correlations_lesion[mdl_idx, cv_fold] = np_pearson_cor(y_test,y1_pred)[0]
            correlations_demog[mdl_idx, cv_fold] = np_pearson_cor(y_test,y2_pred)[0]
            
            mean_abs_error[mdl_idx, cv_fold] = mean_absolute_error(y_test, avg_pred)
            mean_abs_error_lesion[mdl_idx, cv_fold] = mean_absolute_error(y_test, y1_pred)
            mean_abs_error_demog[mdl_idx, cv_fold] = mean_absolute_error(y_test, y2_pred)
            
            explained_var[mdl_idx, cv_fold] =explained_variance_score(y_test, avg_pred)
            explained_var_lesion[mdl_idx, cv_fold] = explained_variance_score(y_test, y1_pred)
            explained_var_demog[mdl_idx, cv_fold] = explained_variance_score(y_test, y2_pred)
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
        logprint('-------------- saving file w root name: {} -------------'.format(results_path + output_path + filename))
        logprint('\n\n')
        np.save(results_path+ output_path +  filename + "_scores_ensemble.npy", explained_var)
        np.save(results_path+ output_path +  filename + "_scores_lesion.npy", explained_var_lesion)
        np.save(results_path+ output_path + filename + "_scores_demog.npy", explained_var_demog)
        np.save(results_path+ output_path +  filename + "_scores.npy", explained_var)

        np.save(results_path+ output_path + filename + "_model.npy", models)
        np.save(results_path+ output_path + filename + "_correlations_ensemble.npy", correlations_ensemble)
        np.save(results_path+ output_path +  filename + "_correlations_demog.npy", correlations_demog)
        np.save(results_path+ output_path +  filename + "_correlations_lesion.npy", correlations_lesion)
        np.save(results_path+ output_path +  filename + "_mean_absolute_error.npy", mean_abs_error)
        np.save(results_path+ output_path +  filename + "_mean_absolute_error_lesion.npy", mean_abs_error_lesion)
        np.save(results_path+ output_path +  filename + "_mean_absolute_error_demog.npy", mean_abs_error_demog)

        np.save(results_path+ output_path +  filename + "_model_labels.npy", mdls_labels)
        np.save(results_path+ output_path +  filename + "_variable_impts.npy", variable_importance)
        np.save(results_path+ output_path +  filename + "_test_group_sizes.npy", size_testgroup)

def set_up_and_run_model(crossval, model_tested,lesionload,lesionload_type, X, Y, C, site, atlas, y_var, chaco_type, subset, save_models, results_path, nperms, null, ensemble, output_path):
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
                if model_tested[0]=='ridge':
                    print("RUNNING")
                    run_regression(X, Y, site, inner_cv_id,outer_cv_id,model_tested, atlas, y_var, chaco_type, subset,save_models, results_path,crossval, nperms,null, output_path)
            elif lesionload_type =='M1':
                atlas = 'lesionload_m1'
                model_tested = ['linear_regression']
                chaco_type ='NA'
                run_regression(lesionload, Y, site, inner_cv_id,outer_cv_id,model_tested, atlas, y_var, chaco_type, subset,1, results_path,crossval, nperms,null,output_path)
            elif lesionload_type =='all':
                atlas = 'lesionload_all'
                model_tested= ['ridge_nofeatselect']
                chaco_type ='NA'
                run_regression(lesionload, Y, site, inner_cv_id,outer_cv_id,model_tested, atlas, y_var, chaco_type, subset,1, results_path,crossval, nperms,null,output_path)
        elif ensemble == 'demog':
            if lesionload_type == 'none':
                if model_tested[0]=='ridge':
                    run_regression_ensemble(X, C, Y, site, inner_cv_id,outer_cv_id,model_tested, atlas, y_var, chaco_type, subset,1, results_path,crossval, nperms,null,output_path)
            elif lesionload_type =='M1':
                atlas = 'lesionload_m1'
                model_tested = ['linear_regression']
                chaco_type = 'NA'
                run_regression_ensemble(lesionload, C, Y, site, inner_cv_id,outer_cv_id,model_tested, atlas, y_var, chaco_type, subset,1, results_path,crossval, nperms,null, output_path)
            elif lesionload_type =='all':
                atlas = 'lesionload_all'
                model_tested= ['ridge_nofeatselect']
                chaco_type ='NA'
                run_regression_ensemble(lesionload, C, Y, site, inner_cv_id,outer_cv_id,model_tested, atlas, y_var, chaco_type, subset,1, results_path,crossval, nperms,null, output_path)
          
def save_model_outputs(results_path, output_path, atlas, y_var, chaco_type, subset, model_tested, crossval, nperms,lesionload_type, ensemble,shape_2):
    
    logprint('Saving model outputs to directory: {}'.format(results_path + output_path))
    
    mdl_label = model_tested[0]
    rootname = results_path + output_path + '/{}_{}_{}_{}_{}_crossval{}'.format(atlas, y_var, chaco_type, subset, mdl_label,crossval)
    
    rootname_truepred = results_path + output_path + '/{}_{}_{}_{}_{}_crossval{}'.format(atlas, y_var, chaco_type, subset, mdl_label,crossval)

    r2scores_allperms=np.zeros(shape=(nperms, shape_2))
    correlation_allperms=np.zeros(shape=(nperms, shape_2))
    
    # if leave-one-site-out, make a figure showing the distribution of predicted scores across permutations 
        # in addition to creating outputs below \/
    if crossval == '2': # leave one site out
        # have to do this here, i guess.
        
        # shape_2 = number of outer folds per permutation.
        for site in range(0,shape_2):
            # oh jeez
            site_size = np.load(rootname_truepred +'_perm0_outerfold'+str(site) +'_true.npy', allow_pickle=True).shape[0]
            true_array = np.zeros(shape=(site_size,nperms))
            pred_array = np.zeros(shape=(site_size,nperms))

            for n in range(0,nperms): 
                true_array[:,n]=np.load(rootname_truepred +'_perm'+ str(n) + '_outerfold'+str(site) +'_true.npy', allow_pickle=True)
                pred_array[:,n]=np.load(rootname_truepred +'_perm'+ str(n) + '_outerfold'+str(site) +'_pred.npy', allow_pickle=True)
            
            #
            np.savetxt(rootname_truepred+ '_outerfold'+str(site) +'_true_allperms.txt', true_array)
            np.savetxt(rootname_truepred+ '_outerfold'+str(site) +'_pred_allperms.txt', pred_array)

            #print(pred_array.shape)
            #create_site_truepred_figures(true_array, pred_array, site_size,results_path, site)
    
    for n in range(0, nperms): #
        # no ensemble model.
        
        if ensemble =='none':
            r2scores=np.load(rootname +'_perm'+ str(n) + '_scores.npy',allow_pickle=True)
            correlation = np.load(rootname +'_perm'+ str(n) +'_correlations.npy',allow_pickle=True)
           # if crossval !="1":
               # varimpts=np.load(rootname +'_perm'+ str(n) + '_activation_weights.npy',allow_pickle=True)
            mdl=np.load(rootname +'_perm'+ str(n) + '_model.npy',allow_pickle=True)
            
            r2scores_allperms[n,] = r2scores
            correlation_allperms[n,] = correlation
            
            if mdl_label == 'ridge':
                alphas=[]
                feats=[]
                
                for a in range(0,5):
                    alphas.append(mdl[0][a][mdl_label].alpha)
                    feats.append(mdl[0][a]['featselect'].k) 
                    
                np.savetxt(rootname_truepred +'_perm'+ str(n)  +'_alphas.txt', alphas)
                np.savetxt(rootname_truepred +'_perm'+ str(n)  +'_nfeats.txt', feats)
                
            
        # if ensemble model was run, the filename is different. catch it here.
        if ensemble =='demog':
            
            r2scores_ensemble=np.load(rootname +'_perm'+ str(n) + '_ensemble'+ '_scores.npy',allow_pickle=True)
            correlation_ensemble = np.load(rootname +'_perm'+ str(n) + '_ensemble'+ '_correlations_ensemble.npy',allow_pickle=True)
            #varimpts_ensemble=np.load(rootname +'_perm'+ str(n) +  '_ensemble'+ '_activation_weights.npy',allow_pickle=True)
            mdl=np.load(rootname +'_perm'+ str(n) + '_ensemble'+  '_model.npy',allow_pickle=True)
            r2scores_allperms[n,] = r2scores_ensemble
            correlation_allperms[n,] = correlation_ensemble

            # logprint('varimpts dim: {}'.format(varimpts_ensemble.shape))

            if mdl_label == 'ridge':
                alphas=[]
                feats=[]
                
                for a in range(0,5):
                    alphas.append(mdl[0][a][mdl_label].alpha)
                    feats.append(mdl[0][a]['featselect'].k) 
                    
    np.savetxt(rootname_truepred + '_r2_scores.txt', r2scores_allperms)
    np.savetxt(rootname_truepred +'_correlations.txt', correlation_allperms)   
                     
    return r2scores_allperms, correlation_allperms

def logprint(string):
    print(string)
    logging.info(string)
 
 
def create_site_truepred_figures(true, pred,site_size,results_path,site):
    font = {'family' : 'normal',
            'size'   : 15}
    fig, ax1 = plt.subplots(figsize =(10, 10))
  
    meanpred = np.mean(pred,axis=1)
    meanpred_rep = np.repeat(np.reshape(meanpred, [site_size, 1]), 8, axis=1)
    meantrue = np.mean(true, axis=1)
    
    err = np.mean(np.abs(pred - meanpred_rep), axis=1) # MAE
    
    print(err)
    plt.scatter(meantrue, meanpred)
    plt.errorbar(meantrue,meanpred,yerr=err, linestyle="none")
    ax1.plot([0,1],[0,1],'k--',transform=ax1.transAxes)
    plt.ylim([0, 1])
    plt.xlim([0, 1])


    #plt.scatter(np.mean(true_, pred,)
    #plt.errorbar(a,b,yerr=c, linestyle="None")
    plt.savefig(results_path + '/figures/' + 'truepredfig_site_' + str(site)+'.png')

def create_performance_figures_loo(r2_scores, correlations,label, results_path, output_path, filename):
    font = {'family' : 'normal',
            'size'   : 22}

    matplotlib.rc('font', **font)
    
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize =(20, 20))
    plt.subplots_adjust(bottom=0.6)

    ax1.boxplot(np.transpose(r2_scores))
    ax1.set_ylim([np.min(r2_scores)-0.1, np.max(r2_scores)+0.3])
    ax1.set_ylabel('{}'.format('$R^2$'))
    ax1.xaxis.set_ticks([x for x in range(1,len(label)+1)])
    ax1.xaxis.set_ticklabels(label, rotation=90)
    
    correlations = np.transpose(correlations)
    
    # remove data points that are nan or from a site with one subject
    mask = np.logical_and(~np.isnan(correlations),correlations<0.99)
    filtered_data = [d[m] for d, m in zip(correlations.T, mask.T)]
    
    ax2.boxplot(filtered_data)
    ax2.set_ylim([np.min(filtered_data)-0.1, np.max(filtered_data)+0.3])
    ax2.set_ylabel('Correlation')
    
    print([x for x in range(1,len(label)+1)])
    
    ax2.xaxis.set_ticks([x for x in range(1,len(label)+1)])
    ax2.xaxis.set_ticklabels(label, rotation=90)
    
    print([results_path + output_path + '/'+ filename + '.png'])
    plt.savefig(results_path + output_path + '/'+ filename + '.png')
    
def create_performance_figures(r2_scores, correlations,label, results_path, output_path, filename):
    font = {'family' : 'normal',
            'size'   : 22}

    matplotlib.rc('font', **font)
    
    # boxplots
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize =(20, 10))
    plt.subplots_adjust(bottom=0.4)

    ax1.boxplot(np.transpose(r2_scores))
    ax1.set_ylim([np.min(r2_scores), np.max(r2_scores)+0.3])
    ax1.set_ylabel('{}'.format('$R^2$'))
    ax1.xaxis.set_ticks([x for x in range(1,len(label)+1)])
    ax1.xaxis.set_ticklabels(label, rotation=90)

    ax2.boxplot(np.transpose(correlations))
    ax2.set_ylim([np.min(r2_scores), np.max(correlations)+0.2])
    ax2.set_ylabel('Correlation')
        
    ax2.xaxis.set_ticks([x for x in range(1,len(label)+1)])
    ax2.xaxis.set_ticklabels(label, rotation=90)
    
    print([results_path + output_path + '/'+ filename + '.png'])
    plt.savefig(results_path + output_path + '/'+ filename + '.png')
    
    