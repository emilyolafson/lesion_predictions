
import sys; sys.path
import pandas as pd
import numpy as np 
import os
import pickle
from helper_functions import *
import glob
import re
#import label encoder
from sklearn import preprocessing 
from sklearn.model_selection import LeaveOneGroupOut

LESIONMASK_PATH = os.path.join("/home/ubuntu/enigma/lesionmasks/")  # Set this path accordingly

def prepare_data(X):
    '''Clean X-data (remove zero-value input variables)'''

    # remove inputs that are 0 for all subjects
    zeros=X==0
    zeros=np.sum(zeros,0)
    zeros=zeros==X.shape[0]
    X=X[:,~zeros]    
    return X

def access_elements(nums, list_index):
    result = [nums[i] for i in list_index]
    return result

def find_missing_scans(ids, parc, chacovar):
    # returns list of files without missing scans, as well as ids of missing subjects
    files_in_dir = glob.glob("{}{}".format(LESIONMASK_PATH,'*_ses-1_space-MNI152_desc-T1-lesion_mask_MNI_1mm_nemo_output_sdstream_{}_{}_mean.pkl'.format(chacovar,parc)))
    list_of_files = LESIONMASK_PATH + ids + ['_ses-1_space-MNI152_desc-T1-lesion_mask_MNI_1mm_nemo_output_sdstream_{}_{}_mean.pkl'.format(chacovar,parc)]
    missing_scans = [x for x in list_of_files if x not in files_in_dir]
    
    ids_fullpaths = LESIONMASK_PATH + ids + ['_ses-1_space-MNI152_desc-T1-lesion_mask_MNI_1mm_nemo_output_sdstream_{}_{}_mean.pkl'.format(chacovar,parc)]
    ids_fullpaths_nonemissing = [x for x in ids_fullpaths if x not in missing_scans]
    missing_id = [x for x in ids_fullpaths if x in missing_scans]
    missinglist=[]
    for missing in missing_id:
        missinglist.append(missing[len(LESIONMASK_PATH):44]) # find subid of subject(s) with missing scans
    
    
    return ids_fullpaths_nonemissing, missinglist

def load_chaco_data(ids,chacovar):
    for i in range(0,len(ids)):
        
        with open(ids[i], 'r+b') as e:
            
            if chacovar == 'chacoconn':
                data = pickle.load(e)
                data= data.todense()
                np.fill_diagonal(data, 0)
                nROIs = data.shape[0]
                if i==0:
                    X = data[np.triu_indices(nROIs,k=1)] # take upper triangular
                    continue
                X = np.concatenate((X, data[np.triu_indices(nROIs,k=1)]), axis=0)
                
            elif chacovar=='chacovol':
                data = pickle.load(e)    
                if i==0:
                    X = data
                    continue
                X = np.concatenate((X, data), axis=0)   
    return X
            
def remove_missing_motor(df):
    print('Removing subjects who do not have motor scores.')
    print('---- Original N = {}'.format(df.shape[0]))

    idx=np.isnan(df['NORMED_MOTOR'])
    df=df[~idx]
    print('---- New N = {}'.format(df.shape[0]))

    return df 

def remove_missing_scans(df, missinglist):
    print('Removing subjects who do not have lesion masks.')
    print('---- Original N = {}'.format(df.shape[0]))

    for missingscan in missinglist:
        df = df[df['BIDS_ID'] != missingscan] 

    print('---- New N = {}'.format(df.shape[0]))
    print('\n')
    return df

def remove_missing_demographics(df,covariates):
    missing_ids = np.zeros(df.shape[0])
    print('\n')
    print('Removing subjects that do not have any of: {}'.format(covariates))
    print('---- Original N = {}'.format(df.shape[0]))
    for cov in covariates:
        
        missing_ids = np.isnan(df[cov]) + missing_ids
        
    missing_ids = missing_ids>0
    df=df[~missing_ids]
    print('---- New N = {}'.format(df.shape[0]))
    print('\n')
    return df

def load_csv(csv_path):
    df = pd.read_csv(csv_path, header =0)
    return df

def get_chronicity_subset(df, subset_data):
    print('\n')

    print('Removing subjects that are not {} stroke subjects'.format(subset_data))
    print('---- Original N = {}'.format(df.shape[0]))
    if subset_data == 'chronic':
            df_chronic = df[df['CHRONICITY']==180]
            df_chronic = df_chronic.reset_index(drop=True)
            df_final = df_chronic
            ids = df_chronic['BIDS_ID']  
    elif subset_data == 'acute':
        df_acute = df[df['CHRONICITY']==90]
        df_acute = df_acute.reset_index(drop=True)
        df_final = df_acute
        ids = df_acute['BIDS_ID']    
    else:
        ids = df['BIDS_ID']
        df_final = df
    print('---- New N = {}'.format(df_final.shape[0]))

    return df_final, ids

def create_data_set(csv_path=None, atlas=None, covariates=None, verbose=False, y_var=None,chaco_type=None, subset=None, remove_demog =None, ll=None):
    """
    Formats ENIGMA data (ChaCo scores, demographic & clinical info) for classification or regression.

    :param csv_path : str, default=CSV_PATH
        Path to ENIGMA csv file. Default is set in CSV_PATH global.
    :param subset : str, default=None
        Specify whether all subjects or only chronic stroke subjects are included in the model. Default is 'chronic'. Options: 'chronic', 'all', 'acute'
    :param atlas : str, default=None
        Specify parcellation for ChaCo score data. Default is 'fs86'. Options: 'fs86subj', 'shen268'
    :param chaco_type : str, default=None
        Pairwise or regional ChaCo scores used in prediction. Options: 'chacovol', 'chacoconn', 
    :param covariates : str or list, default=None
        Covariates that will be returned from the data set. Options: 'AGE', 'SEX', 'CHRONICITY', 'LESIONED_HEMISPHERE'
    :param verbose : boolean, default=False
        Enable prints for detailed logging information.
    :param y_var : str, default=None
        Label for variable set as "y". Default is 'normed_motor_scores', can be 'severity' for classification tasks.
    :param remove_demog : int, default=None
        If specified, remove subjs from final list that dont have demographic scores (sex, age, lesioned hem)
    :return: X, C, y, lesionload, sites
    """
    
    df = load_csv(csv_path)
    
    if atlas:
        parc = atlas
    else:
        parc = 'fs86subj'
        
    if chaco_type:
        chacovar = chaco_type
    else:
        chacovar = 'chacovol'
        
    if subset:
        subset_data = subset
    else: 
        subset_data = 'chronic'
        
    if y_var:
        y_var = y_var
    else:
        y_var = 'normed_motor_scores'

        
    #print('FULL DF SHAPE')
    #print(df.shape)
    # format data frame, remove missing data
    df = remove_missing_motor(df)
    #print('DF AFTER REMOVING MISSING MOTOR SCORES')
    #print(df.shape)
    
    # covariate extraction (age, sex, site, etc) 
    
    all_cov_labels = access_elements(df.columns.values,[3,4,5,6,9]) # Age, sex, days post stroke, chronicity, lesioned hem 
 
    if covariates:
        if isinstance(covariates, str):
            covariates = [covariates]
        if not isinstance(covariates, list):
            raise RuntimeError('TypeError: covariates must be str or list, {} not accepted \n'.format(type(covariates)))
        if not set(covariates).issubset(set(all_cov_labels)):
            raise RuntimeError('Warning! Unknown covariates specified: {} \n'
                               'Only the following options are allowed: {} \n'.format(covariates, all_cov_labels))
        covariates_list = covariates
    else:
        covariates_list = []
        
    if remove_demog:
        df = remove_missing_demographics(df,covariates_list)   
    
    #print('DF SHAPE AFTER REMOVAL OF MISSING DEMOGRAPHICS:')
    #print(df.shape)
    ids=df['BIDS_ID']
    
    df_final, ids = get_chronicity_subset(df, subset_data)
   
    
    #print('DF SHAPE AFTER REMOVAL OF ACUTE:')
    #print(df_final.shape)
          
    # find subjects who have motor scores but are missing scans.
    ids_fullpaths_nonemissing, missinglist = find_missing_scans(ids, parc, chacovar)
    df_final = remove_missing_scans(df_final,missinglist)  

    #print('DF SHAPE AFTER REMOVAL OF MISSING SCANS:')
    #print(df_final.shape)
    
    print('Loading data for atlas: {}, ChaCo scores: {}, subset: {}'.format(atlas, chacovar,subset_data))
    # load X data
    X = load_chaco_data(ids_fullpaths_nonemissing, chacovar)
    X = np.array(X)
    
    if verbose and isinstance(covariates, str):
        print('Extracting {} from atlas {} \n'.format(str(covariates_list), atlas))
        
    C = df_final.loc[:,covariates_list].values
    site = df_final["SITE"]
    #make an instance of Label Encoder
    label_encoder = preprocessing.LabelEncoder()
    site = label_encoder.fit_transform(site)        

    # load y data
    if y_var == 'normed_motor_scores':
        y = df_final['NORMED_MOTOR'].values
       # y = np.reshape(y, (len(y),1))
        
    elif y_var =='severity':
        y = df_final['NORMED_MOTOR'].values < 0.424
        y = y.astype(int)
    
    # load lesionload

    llvars = ['M1_CST', 'PMd_CST', 'PMv_CST','S1_CST','SMA_CST','preSMA_CST']
    if ll=='all':
        lesion_load = df_final.loc[:,llvars]
    elif ll=='M1':
        lesion_load=df_final.loc[:,'M1_CST']
    
    print('Final size of data: \n X_data: {} by {} \n Y_data: length {} \n'.format(X.shape[0], X.shape[1], y.shape[0]))
    return X, y, C, lesion_load, site




def calculate_demographics_dataset(X, Y, C, site):
    outer_cv = LeaveOneGroupOut()
    outer_cv_splits = outer_cv.get_n_splits(X, Y, site)

    site_size = np.zeros((outer_cv_splits,1), dtype=object)
    size_mean_NMS = np.zeros((outer_cv_splits,1), dtype=object)
    size_std_NMS = np.zeros((outer_cv_splits,1), dtype=object)
    missing_age = np.zeros((outer_cv_splits,1), dtype=object)
    missing_sex = np.zeros((outer_cv_splits,1), dtype=object)
    missing_lesionedhem = np.zeros((outer_cv_splits,1), dtype=object)
    site_mean_pctF = np.zeros((outer_cv_splits,1), dtype=object)
    site_mean_age = np.zeros((outer_cv_splits,1), dtype=object)

    cv_idx=0
    for train_index, test_index in outer_cv.split(X, Y, site):
        y_train, y_test = Y[train_index], Y[test_index]
        site_train, site_test = site[train_index], site[test_index]
        C_train, C_test = C[train_index], C[test_index]

        site_size[cv_idx]= site_test.shape[0]
        size_mean_NMS[cv_idx] = np.mean(y_test)
        size_std_NMS[cv_idx] = np.std(y_test)
        
        missing_age[cv_idx]=np.sum(np.isnan(C_test[:,1]))
        missing_sex[cv_idx]=np.sum(np.isnan(C_test[:,0]))
        missing_lesionedhem[cv_idx]=np.sum(np.isnan(C_test[:,2]))
        site_mean_pctF[cv_idx]=np.sum(C_test[:,0]==2)/C_test.shape[0]
        site_mean_age[cv_idx]=np.mean(C_test[:,1])

        

        cv_idx+=1
    size_motorscores = np.concatenate((site_size,size_mean_NMS,size_std_NMS,missing_age,missing_sex,missing_lesionedhem,site_mean_pctF,site_mean_age), axis=1)
    return size_motorscores
