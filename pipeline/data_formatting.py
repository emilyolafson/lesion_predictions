
import sys; sys.path
import pandas as pd
import numpy as np 
import pickle
from helper_functions import *
import glob
from sklearn import preprocessing 

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

def find_missing_scans(ids, atlas, chaco_type,nemo_path, nemo_settings):
    # Get full filenames for NeMo outputs:
    # NeMo suffix:

    if chaco_type == 'NA':
        chaco_type = 'chacovol'
    if not (atlas == 'fs86subj' or atlas == 'shen268'): # if atlas == none (this may occur when we're doing just LL, but we still want to load chaco data.)
        atlas = 'fs86subj'
    print(atlas)
    print(chaco_type)
    nemo_suffix = '_{}_nemo_output_{}_{}_{}_mean.pkl'.format(nemo_settings[0], nemo_settings[1],chaco_type,atlas)
    files = glob.glob(os.path.join(nemo_path, '*_{}_nemo_output_{}_{}_{}_mean.pkl'.format(nemo_settings[0], nemo_settings[1], chaco_type, atlas)))

    # check if there is any content between the ID and the lesionmask filename after removing the NeMo bits.
    files_stripped = [file.replace(nemo_path, '').replace(nemo_suffix, '') for file in files]

    i = 0
    length = len(files_stripped)
    
    c=0
    while i < length:
        for id in ids:
            if id in files_stripped[i]:
                if not id == files_stripped[i]: # if there's not an exact character match
                    if len(id)>len(files_stripped[i]):
                        print('Subjects IDs may not be specific enough to find the corresponding NeMo output file. ')
                    else:
                        filebits = files_stripped[i].split(id)[1]
                        if c == 0: # only run once
                            current_diff = filebits
                            c=c+1   
                        else:
                            if not filebits == current_diff:
                                print(filebits)

                                print('Multiple different image file names.')
                                
                else:
                    filebits=[]
                       
        i += 1

    # returns list of files without missing scans, as well as ids of missing subjects
    files_in_dir = glob.glob(os.path.join(nemo_path,'*'+ filebits + nemo_suffix))
    list_of_files_fromIDlist = nemo_path + ids + filebits + nemo_suffix 
    missing_scans = [x for x in list_of_files_fromIDlist if x not in files_in_dir]
    ids_fullpaths_nonemissing = [x for x in list_of_files_fromIDlist if x not in missing_scans]
    endbits_toremove = filebits + nemo_suffix

    missing_scans = [scan.split(endbits_toremove)[0].split(nemo_path)[1] for scan in missing_scans] # get sub ID in table form.
    print('\nThe following subjects are in the .csv file but do not have corresponding ChaCo data: {}\n'.format(missing_scans))
    return ids_fullpaths_nonemissing, missing_scans

def load_chaco_data(ids,chaco_type):
    # This function takes in a list of IDs and a string containing the type of chaco data (either 'chacovol' or 'chacoconn')
    # and returns a matrix of the chaco data. For 'chacoconn' data, the matrix is the upper triangular portion of the adjacency
    # matrix with the diagonal set to 0. For 'chacovol' data, the matrix is the volume data for each subject.
    if chaco_type == 'NA':
        chaco_type = 'chacovol'
        
    for i in range(0,len(ids)):
        
        with open(ids[i], 'r+b') as e:
            
            if chaco_type == 'chacoconn':
                data = pickle.load(e)
                data= data.todense()
                np.fill_diagonal(data, 0)
                nROIs = data.shape[0]
                if i==0:
                    X = data[np.triu_indices(nROIs,k=1)] # take upper triangular
                    continue
                X = np.concatenate((X, data[np.triu_indices(nROIs,k=1)]), axis=0)
                
            elif chaco_type=='chacovol':
                data = pickle.load(e)    
                if i==0:
                    X = data
                    continue
                X = np.concatenate((X, data), axis=0)   
    return X
            
def remove_missing_yvar(df, yvar_colname):

    idx=np.isnan(df[yvar_colname])
    df=df[~idx]

    return df 

def remove_missing_scans(df, missinglist,subid_colname):

    for missingscan in missinglist:
        df = df[df[subid_colname] != missingscan] 

    return df

def remove_missing_demographics(df,covariates):
    
    missing_ids = np.zeros(df.shape[0])

    for cov in covariates:
        
        missing_ids = np.isnan(df[cov]) + missing_ids
        
    missing_ids = missing_ids>0
    df=df[~missing_ids]

    return df

def load_csv(csv_path):
    df = pd.read_csv(csv_path, header =0)
    return df

def get_chronicity_subset(df, subset, subid_colname, chronicity_colname):
    # The get_chronicity_subset function filters a DataFrame df to only include data for a certain subset of stroke subjects, 
    # as specified by the subset parameter. The subset parameter can be either 'chronic' or 'acute', 
    # to indicate whether to include only chronic or acute stroke subjects, respectively. If the subset
    # parameter is set to any other value, the function will return the original DataFrame without filtering.

    # It then uses an if statement to check the value of subset and filter the DataFrame accordingly. 
    # If subset is 'chronic', the function filters the DataFrame to only include rows where the 'CHRONICITY'
    # column has a value of 180. If subset is 'acute', the function filters the DataFrame to only include rows
    # where the 'CHRONICITY' column has a value of 90. In either case, the function then resets the DataFrame's
    # index and saves the filtered DataFrame to the df_final variable.

    # Finally, the function returns the list of subject IDs in the filtered DataFrame, which are stored in the 'BIDS_ID' column.

    if subset == 'chronic':
        print('\nSelecting chronic subjects only.\n')

        df_chronic = df[df[chronicity_colname]==180]
        df_chronic = df_chronic.reset_index(drop=True)
        df_final = df_chronic
        ids = df_chronic[subid_colname]  
    elif subset == 'acute':
        print('\nSelecting acute subjects only. \n')
        df_acute = df[df[chronicity_colname]==90]
        df_acute = df_acute.reset_index(drop=True)
        df_final = df_acute
        ids = df_acute[subid_colname]
        
    elif subset == 'acutechronic': # if acute+chronic, load chronic data here. 
        print('\nSelecting chronic subjects only.\n')

        df_chronic = df[df[chronicity_colname]==180]
        df_chronic = df_chronic.reset_index(drop=True)
        df_final = df_chronic
        ids = df_chronic[subid_colname] 
    else:
        print('\nSelecting all subjects\n')
        ids = df[subid_colname]
        df_final = df

    return df_final, ids

def load_lesion_vol(df_final):
    lesionvolpath = '/home/ubuntu/enigma/lesionvol'
    lesionvol=np.zeros(shape=(df_final.shape[0],1))
    i=0
    for subject in df_final['BIDS_ID']:
        lesionvol[i]=(np.loadtxt(os.path.join(lesionvolpath, subject + '.txt'))[0])
        i=i+1
    df_final['lesionvol']=lesionvol
    
    return lesionvol
        

def create_data_set(csv_path=None, site_colname = None, nemo_path=None,yvar_colname = None, subid_colname=None,chronicity_colname=None,atlas=None, covariates=None, verbose=False, y_var=None,chaco_type=None, subset=None, remove_demog =None, nemo_settings=None, ll=None,return_motor=False):
    print('\n\nLoading .csv...')
    print(csv_path)
    df = load_csv(csv_path)
    print('\nSize of dataset before removing subjects without outcome scores: {} subjects'.format(df.shape[0]))

    df = remove_missing_yvar(df, yvar_colname)
    print('Size of dataset after removing subjects without outcome scores: {} subjects\n'.format(df.shape[0]))

    all_cov_labels = df.columns.values # Age, sex, days post stroke, chronicity, lesioned hem 
    
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
        
    all_ll_options = ['M1', 'all', 'none', 'all_2h', 'slnm']
    
    if ll:
        if isinstance(ll, str):
            ll = ll
        if not set([ll]).issubset(set(all_ll_options)):
            raise RuntimeError('Warning! Unknown lesion load option specified: {} \n'
                               'Only the following options are allowed: {} \n'.format(ll, all_ll_options))
    
    print('\nSize of dataset before removing subjects without non-lesion covariates {}: {} subjects'.format(covariates_list,df.shape[0]))
    df = remove_missing_demographics(df,covariates_list)   
    print('Size of dataset after removing subjects without non-lesion covariates: {} subjects'.format(df.shape[0]))

    # this was for me specifically, just make sure your sex variable is a logical.
    if 'SEX' in covariates_list:
        sex = df['SEX']
        df['SEX'] = sex -1
        df = df[df['SEX'] <= 1]
        

    ids=df[subid_colname]
    print('\nSize of dataset before subsetting for chronic/acute: {} subjects'.format(df.shape[0]))
    df_final, ids = get_chronicity_subset(df, subset, subid_colname, chronicity_colname)
    print('Size of dataset after subsetting for chronic/acute: {} subjects'.format(df_final.shape[0]))

    # find subjects who have motor scores but are missing scans.
    ids_fullpaths_nonemissing, missinglist = find_missing_scans(ids, atlas, chaco_type,nemo_path, nemo_settings)
    df_final = remove_missing_scans(df_final,missinglist,subid_colname)  
    
    X = load_chaco_data(ids_fullpaths_nonemissing, chaco_type)
    X = np.array(X)

    C = df_final.loc[:,covariates_list].values
    
        
    # load y data
    y = df_final[yvar_colname].values
       # y = np.reshape(y, (len(y),1))
    
    
    llvars = ['M1_CST', 'PMd_CST', 'PMv_CST','S1_CST','SMA_CST','preSMA_CST']
    ll_2h_vars =['L_M1_CST', 'L_PMd_CST', 'L_PMv_CST','L_S1_CST','L_SMA_CST','L_preSMA_CST','R_M1_CST', 'R_PMd_CST', 'R_PMv_CST','R_S1_CST','R_SMA_CST','R_preSMA_CST']
    slnm_vars = ['PC1', 'PC2_1', 'PC2_2','PC3_1','PC3_2']
    if ll=='all':
        lesion_load = df_final.loc[:,llvars]
    elif ll=='M1':
        lesion_load=df_final.loc[:,'M1_CST']
    elif ll=='all_2h':
        lesion_load=df_final.loc[:,ll_2h_vars]
    elif ll=='slnm':
        lesion_load = df_final.loc[:,slnm_vars]
    elif ll=='none':
        lesion_load=[]
        
    subIDs = df_final[subid_colname]
    

    return X, y, C, lesion_load, subIDs

