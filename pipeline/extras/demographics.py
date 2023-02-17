import helper_functions 
from helper_functions import *
from imp import reload
import pandas as pd
import data_formatting
from data_formatting import *
reload(helper_functions)
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

def round_with_padding(value, round_digits):
    return format(round(value,round_digits), "."+str(round_digits)+"f")

csv_path='/home/ubuntu/enigma/motor_predictions/Behaviour_Information_ALL_April7_2022_sorted_CST_12_ll_slnm.csv'
nemo_settings='1mm,sdstream'
nemo_path='/home/ubuntu/enigma/lesionmasks/'
results_path='/home/ubuntu/enigma/results/'
analysis_id='analysis_1'
workbench_vis=False
scenesdir='/home/ubuntu/enigma/motor_predictions/wb_files'
hcp_dir='/home/ubuntu/enigma/motor_predictions/wb_files/HCP_S1200_GroupAvg_v1'
wbpath='/home/ubuntu/enigma/motor_predictions/wb_files/workbench_ubuntu/bin_linux64'
chaco_type='chacovol'
atlas='fs86subj'
covariates=['AGE','SEX','DAYS_POST_STROKE']
boxplots=False
lesionload_types='M1,all,all_2h'
crossval_types='1'
verbose=0
nperms=100
save_models=True
ensembles='none'
subid_colname='BIDS_ID'
ensemble_atlas='fs86subj'
site_colname='SITE'
chronicity_colname='CHRONICITY'
yvar_colname='NORMED_MOTOR'
subset='acute'
y_var='normed_motor_scores'
model_specified='ridge'
override_rerunmodels=False
remove_demog=None
ll='M1'
nemo_settings= ['1mm','sdstream']


original_dataset = load_csv(csv_path)



print(original_dataset)


[X, Y, C, lesion_load, site,site_idx, lesionvol] = create_data_set(csv_path, site_colname, nemo_path,yvar_colname, subid_colname,chronicity_colname,atlas, covariates, verbose, y_var,chaco_type, subset, remove_demog , nemo_settings, ll)

siteID = np.unique(site)
totalNo=[]
totalNWomen=[]
totalNMen=[]
medianAge=[]
minAge=[]
maxAge=[]
iqr=[]
ageStrings=[]
motorStrings=[]
sexStrings=[]
dayspostStrings=[]
# get mesaures for each site
for sites in siteID:
    idx = site==sites
    C_site = C[idx]
    Y_site = Y[idx]
    sexString = "{} ({}/{})".format(np.sum(idx), np.sum(C_site[:,1]==0),np.sum(C_site[:,1]==1)) 
    sexStrings.append(sexString)
    q1,q3=np.percentile(C_site[:,0], [25, 75])
    IQRange_age = q3-q1
    ageString = "{0:.1f} ({1:.1f}, {2:.1f}-{3:.1f})".format(np.median(C_site[:,0]), IQRange_age, np.min(C_site[:,0]), np.max(C_site[:,0]))
    ageStrings.append(ageString)
    q1,q3=np.percentile(Y_site, [25, 75])
    IQRange_motor = q3-q1
    motorString = "{0:.2f} ({1:.2f}, {2:.2f}-{3:.2f})".format(np.median(Y_site),IQRange_motor, np.min(Y_site), np.max(Y_site))
    motorStrings.append(motorString)
    q1,q3=np.percentile(C_site[:,2], [25, 75])
    DPS_IQR = q3-q1
    dayspostString = "{0:.1f} ({1:.1f}, {2:.1f}-{3:.1f})".format(np.median(C_site[:,2])/30, DPS_IQR/30, np.min(C_site[:,2])/30, np.max(C_site[:,2])/30)
    dayspostStrings.append(dayspostString)
    print('Number of subjects with Days post stroke == 180: {}'.format(np.sum(C_site[:,2]==180)))

# get full sample measures
q1,q3=np.percentile(C[:,0], [25, 75])
IQRange_age = q3-q1
fullAgeString = "{0:.1f} ({1:.1f}, {2:.1f}-{3:.1f})".format(np.median(C[:,0]), IQRange_age, np.min(C[:,0]), np.max(C[:,0]))
fullsexString = "{} ({}/{})".format(C.shape[0], np.sum(C[:,1]==0),np.sum(C[:,1]==1)) 
q1,q3=np.percentile(Y, [25, 75])
IQRange_motor = q3-q1
fullMotorString = "{0:.1f} ({1:.1f}, {2:.1f}-{3:.1f})".format(np.median(Y), IQRange_motor, np.min(Y), np.max(Y))
q1,q3=np.percentile(C_site[:,2], [25, 75])
DPS_IQR = q3-q1
fullDaysPostString = "{0:.1f} ({1:.1f}, {2:.1f}-{3:.1f})".format(np.median(C[:,2])/30, DPS_IQR/30, np.min(C[:,2])/30, np.max(C[:,2])/30)



sexStrings.append(fullsexString)
ageStrings.append(fullAgeString)
motorStrings.append(fullMotorString)
dayspostStrings.append(fullDaysPostString)
siteID = siteID.tolist()
siteID.append('all sites')

datatable = pd.DataFrame({'Total N.(F/M)':sexStrings, 'Median age (y) (IQR, min-max)':ageStrings, 'Median normed motor score (IQR, min-max)':motorStrings, 'Median time since stroke (days) (IQR, min-max)':dayspostStrings})
#print(datatable.to_latex(index=False))

datatable.index = siteID

s = datatable.style
print(s.to_latex(
    column_format="rrrrrr", position="h", position_float="centering",
    hrules=True, label="table:5", caption="Site breakdown of total sample size (N), number of females (F) and males (M), and information about age (years), normed motor scores, and time since stroke (days). IQR, interquartile range",
    multirow_align="t",multicol_align="c"
)  )



## no min-max:

[X, Y, C, lesion_load, site,site_idx,lesionvol] = create_data_set(csv_path, site_colname, nemo_path,yvar_colname, subid_colname,chronicity_colname,atlas, covariates, verbose, y_var,chaco_type, subset, remove_demog , nemo_settings, ll)
siteID = np.unique(site)
totalNo=[]
totalNWomen=[]
totalNMen=[]
medianAge=[]
minAge=[]
maxAge=[]
iqr=[]
ageStrings=[]
motorStrings=[]
sexStrings=[]
dayspostStrings=[]
lesionvolStrings=[]
# get mesaures for each site
for sites in siteID:
    idx = site==sites
    C_site = C[idx]
    Y_site = Y[idx]
    lesionvol_site = lesionvol[idx]
    sexString = "{} ({}/{})".format(np.sum(idx), np.sum(C_site[:,1]==0),np.sum(C_site[:,1]==1)) 
    sexStrings.append(sexString)
    q1,q3=np.percentile(C_site[:,0], [25, 75])
    IQRange_age = q3-q1
    ageString = "{0:.1f} ({1:.1f})".format(np.median(C_site[:,0]), IQRange_age)
    ageStrings.append(ageString)
    q1,q3=np.percentile(Y_site, [25, 75])
    IQRange_motor = q3-q1
    motorString = "{0:.2f} ({1:.2f})".format(np.median(Y_site),IQRange_motor)
    motorStrings.append(motorString)
    q1,q3=np.percentile(C_site[:,2], [25, 75])
    DPS_IQR = q3-q1
    dayspostString = "{0:.1f} ({1:.1f})".format(np.median(C_site[:,2])/30, DPS_IQR/30)
    dayspostStrings.append(dayspostString)
    q1,q3=np.percentile(lesionvol_site*0.001, [25, 75])
    lesionvolIQR = q3-q1
    lesionvolz ="{0:.2f} ({1:.2f})".format(np.median(lesionvol_site)*0.001,lesionvolIQR)
    lesionvolStrings.append(lesionvolz)

# get full sample measures
q1,q3=np.percentile(C[:,0], [25, 75])
IQRange_age = q3-q1
fullAgeString = "{0:.1f} ({1:.1f})".format(np.median(C[:,0]), IQRange_age)
fullsexString = "{} ({}/{})".format(C.shape[0], np.sum(C[:,1]==0),np.sum(C[:,1]==1)) 
q1,q3=np.percentile(Y, [25, 75])
IQRange_motor = q3-q1
fullMotorString = "{0:.1f} ({1:.1f})".format(np.median(Y), IQRange_motor)
q1,q3=np.percentile(C_site[:,2], [25, 75])
DPS_IQR = q3-q1
fullDaysPostString = "{0:.1f} ({1:.1f})".format(np.median(C[:,2])/30, DPS_IQR/30)
q1,q3=np.percentile(lesionvol*0.001, [25, 75])
lesionvolIQR = q3-q1
fullLesionVol ="{0:.2f} ({1:.2f})".format(np.median(lesionvol)*0.001,lesionvolIQR)


sexStrings.append(fullsexString)
ageStrings.append(fullAgeString)
motorStrings.append(fullMotorString)
dayspostStrings.append(fullDaysPostString)
lesionvolStrings.append(fullLesionVol)
siteID = siteID.tolist()
siteID.append('all sites')

datatable = pd.DataFrame({'Total N.(F/M)':sexStrings, 'Median age (y) (IQR)':ageStrings, 'Median normed motor score (IQR)':motorStrings, 'Median time since stroke (days) (IQR)':dayspostStrings, 'Lesion volume (mm3)':lesionvolStrings})
#print(datatable.to_latex(index=False))

datatable.index = siteID

s = datatable.style
print(s.to_latex(
    column_format="rrrrrrr", position="h", position_float="centering",
    hrules=True, label="table:5", caption="Site breakdown of total sample size (N), number of females (F) and males (M), and information about age (years), normed motor scores, and time since stroke (days). IQR, interquartile range",
    multirow_align="t",multicol_align="c"
)  )



csv_path='/home/ubuntu/enigma/motor_predictions/Behaviour_Information_ALL_April7_2022_sorted_CST_12_ll_slnm.csv'
nemo_settings='1mm,sdstream'
nemo_path='/home/ubuntu/enigma/lesionmasks/'
results_path='/home/ubuntu/enigma/results/'
analysis_id='analysis_1'
workbench_vis=False
scenesdir='/home/ubuntu/enigma/motor_predictions/wb_files'
hcp_dir='/home/ubuntu/enigma/motor_predictions/wb_files/HCP_S1200_GroupAvg_v1'
wbpath='/home/ubuntu/enigma/motor_predictions/wb_files/workbench_ubuntu/bin_linux64'
chaco_type='chacovol'
atlas='fs86subj'
covariates=['AGE','SEX','DAYS_POST_STROKE']
boxplots=False
lesionload_types='M1,all,all_2h'
crossval_types='1'
verbose=0
nperms=100
save_models=True
ensembles='none'
subid_colname='BIDS_ID'
ensemble_atlas='fs86subj'
site_colname='SITE'
chronicity_colname='CHRONICITY'
yvar_colname='NORMED_MOTOR'
subset='none'
y_var='normed_motor_scores'
model_specified='ridge'
override_rerunmodels=False
remove_demog=None
ll='M1'
nemo_settings= ['1mm','sdstream']


original_dataset = load_csv(csv_path)


print(original_dataset)


[X, Y, C, lesion_load, site,site_idx, lesionvol] = create_data_set(csv_path, site_colname, nemo_path,yvar_colname, subid_colname,chronicity_colname,atlas, covariates, verbose, y_var,chaco_type, subset, remove_demog , nemo_settings, ll)

siteID = np.unique(site)
totalNo=[]
totalNWomen=[]
totalNMen=[]
medianAge=[]
minAge=[]
maxAge=[]
iqr=[]
ageStrings=[]
motorStrings=[]
sexStrings=[]
dayspostStrings=[]
lesionvolStrings=[]
# get mesaures for each site
for sites in siteID:
    idx = site==sites
    C_site = C[idx]
    Y_site = Y[idx]
    lesionvol_site = lesionvol[idx]
    sexString = "{} ({}/{})".format(np.sum(idx), np.sum(C_site[:,1]==0),np.sum(C_site[:,1]==1)) 
    sexStrings.append(sexString)
    q1,q3=np.percentile(C_site[:,0], [25, 75])
    IQRange_age = q3-q1
    ageString = "{0:.1f} ({1:.1f})".format(np.median(C_site[:,0]), IQRange_age)
    ageStrings.append(ageString)
    q1,q3=np.percentile(Y_site, [25, 75])
    IQRange_motor = q3-q1
    motorString = "{0:.2f} ({1:.2f})".format(np.median(Y_site),IQRange_motor)
    motorStrings.append(motorString)
    q1,q3=np.percentile(C_site[:,2], [25, 75])
    DPS_IQR = q3-q1
    dayspostString = "{0:.1f} ({1:.1f})".format(np.median(C_site[:,2])/30, DPS_IQR/30)
    dayspostStrings.append(dayspostString)
    q1,q3=np.percentile(lesionvol_site*0.001, [25, 75])
    lesionvolIQR = q3-q1
    lesionvolz ="{0:.2f} ({1:.2f})".format(np.median(lesionvol_site)*0.001,lesionvolIQR)
    lesionvolStrings.append(lesionvolz)

# get full sample measures
q1,q3=np.percentile(C[:,0], [25, 75])
IQRange_age = q3-q1
fullAgeString = "{0:.1f} ({1:.1f})".format(np.median(C[:,0]), IQRange_age)
fullsexString = "{} ({}/{})".format(C.shape[0], np.sum(C[:,1]==0),np.sum(C[:,1]==1)) 
q1,q3=np.percentile(Y, [25, 75])
IQRange_motor = q3-q1
fullMotorString = "{0:.1f} ({1:.1f})".format(np.median(Y), IQRange_motor)
q1,q3=np.percentile(C_site[:,2], [25, 75])
DPS_IQR = q3-q1
fullDaysPostString = "{0:.1f} ({1:.1f})".format(np.median(C[:,2])/30, DPS_IQR/30)
q1,q3=np.percentile(lesionvol*0.001, [25, 75])
lesionvolIQR = q3-q1
q1,q3=np.percentile(lesionvol*0.001, [25, 75])
lesionvolIQR = q3-q1
fullLesionVol ="{0:.2f} ({1:.2f})".format(np.median(lesionvol)*0.001,lesionvolIQR)


sexStrings.append(fullsexString)
ageStrings.append(fullAgeString)
motorStrings.append(fullMotorString)
dayspostStrings.append(fullDaysPostString)
lesionvolStrings.append(fullLesionVol)
siteID = siteID.tolist()
siteID.append('all sites')


datatable = pd.DataFrame({'Total N.(F/M)':sexStrings, 'Median age (y) (IQR)':ageStrings, 'Median normed motor score (IQR)':motorStrings, 'Median time since stroke (days) (IQR)':dayspostStrings, 'Lesion volume (mm3)':lesionvolStrings})
#print(datatable.to_latex(index=False))

datatable.index = siteID

s = datatable.style
print(s.to_latex(
    column_format="rrrrrr", position="h", position_float="centering",
    hrules=True, label="table:5", caption="Site breakdown of total sample size (N), number of females (F) and males (M), and information about age (years), normed motor scores, and time since stroke (days). IQR, interquartile range",
    multirow_align="t",multicol_align="c"
)  )

# compare motor scores in subjects with and without demographic data.
#original_dataset = load_csv(csv_path)
#chronic = original_dataset[original_dataset['CHRONICITY']==180]





# Make figures about lesion load distribution 

#ll='M1'

#[X, Y, C, lesion_load, site,site_idx] = create_data_set(csv_path, site_colname, nemo_path,yvar_colname, subid_colname,chronicity_colname,atlas, covariates, verbose, y_var,chaco_type, subset, remove_demog , nemo_settings, ll)

plt.figure(figsize=(15,13))
plt.hist(lesion_load,bins=100)
plt.xlabel('Lesion load',fontsize=20)
plt.ylabel('Count',fontsize=20)
plt.tick_params(axis='both', which='major', labelsize=20)
#plt.savefig(os.path.join(results_path, analysis_id, 'm1_lesionload.png'),bbox_inches='tight')



ll='all'

#[X, Y, C, lesion_load, site,site_idx] = create_data_set(csv_path, site_colname, nemo_path,yvar_colname, subid_colname,chronicity_colname,atlas, covariates, verbose, y_var,chaco_type, subset, remove_demog , nemo_settings, ll)
fig,ax=plt.subplots(2,3,figsize=(20, 15))
countery=0
counterx=0

for col in lesion_load.columns:
    ax[counterx%2, countery%3].hist(lesion_load[col],bins=100)
    ax[counterx%2, countery%3].set_title(col,fontsize=18)
    ax[counterx%2, countery%3].set_xlabel('Lesion load')
    ax[counterx%2, countery%3].set_ylabel('Count')
    ax[counterx%2, countery%3].tick_params(axis='both', which='major', labelsize=18)
    ax[counterx%2, countery%3].set_xlabel('Lesion load',fontsize=18)
    ax[counterx%2, countery%3].set_ylabel('Count',fontsize=18)
    if countery==2:
        counterx = counterx+1
            
    countery=countery+1
plt.ylabel('Count',fontsize=18)
plt.xlabel('Lesion load',fontsize=18)

#print(analysis_id)
print(os.path.join(results_path, analysis_id, 'all_lesionload.png'))
#plt.savefig(os.path.join(results_path, analysis_id, 'all_lesionload.png'),bbox_inches='tight')



ll='all_2h'

[X, Y, C, lesion_load, site,site_idx] = create_data_set(csv_path, site_colname, nemo_path,yvar_colname, subid_colname,chronicity_colname,atlas, covariates, verbose, y_var,chaco_type, subset, remove_demog , nemo_settings, ll)
fig,ax=plt.subplots(2,6,figsize=(45, 13.333))
countery=0
counterx=0
for col in lesion_load.columns:
    ax[counterx%2, countery%6].hist(lesion_load[col],bins=100)
    ax[counterx%2, countery%6].set_title(col,fontsize=18)
    ax[counterx%2, countery%6].set_xlabel('Lesion load')
    ax[counterx%2, countery%6].set_ylabel('Count')
    ax[counterx%2, countery%6].tick_params(axis='both', which='major', labelsize=18)
    ax[counterx%2, countery%6].set_xlabel('Lesion load',fontsize=18)
    ax[counterx%2, countery%6].set_ylabel('Count',fontsize=18)
    if countery==5:
        counterx = counterx+1
        
    countery=countery+1
    

#print(analysis_id)
#plt.savefig(os.path.join(results_path, analysis_id, 'all2h_lesionload.png'),bbox_inches='tight')