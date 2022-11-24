
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.stats as stats 
import nibabel as nib
import nibabel.processing
import seaborn as sns
import os
import shutil



def create_performance_figures(r2_scores, correlations,label, results_path, analysis_id):
    
    title = ''
    xticklabels = label
    path = results_path + '/' + analysis_id 
    print('Saving figures with filename: {}'.format(path))
    ylabel = 'R-squared'
    path_file = path + '/'+ analysis_id + '_boxplots_rsquared.png'
    box_and_whisker(np.transpose(r2_scores), title, ylabel, xticklabels, path_file)
    
    ylabel = 'Correlations'
    path_file = path + '/'+ analysis_id + '_boxplots_correlations.png'
    box_and_whisker(np.transpose(correlations), title, ylabel, xticklabels, path_file)

def create_performance_figures_loo(r2_scores, correlations,label, results_path, output_path, filename):
    font = {'family' : 'normal',
            'size'   : 15}

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
    
    
def box_and_whisker(data, title, ylabel, xticklabels, path):
    """
    Create a box-and-whisker plot with significance bars.
    Source: https://rowannicholls.github.io/python/graphs/ax_based/boxplots_significance.html
    """
    font = {'family' : 'normal',
            'size'   : 15}
    matplotlib.rc('font', **font)

    # Change figure quality
    plt.rc('figure', dpi=300)

    fig, ax = plt.subplots(ncols=1, figsize =(7, 7))
    
    
    bp = ax.boxplot(data, widths=0.6, patch_artist=True)
    # Graph title
    ax.set_title(title, fontsize=14)
    # Label y-axis
    ax.set_ylabel(ylabel,fontsize=14)
    # Label x-axis ticks
    ax.set_xticklabels(xticklabels,rotation=90,fontsize=14)

    # Hide x-axis major ticks
    ax.tick_params(axis='x', which='major', length=0)
    # Show x-axis minor ticks
    xticks = [0.5] + [x + 0.5 for x in ax.get_xticks()]
    ax.set_xticks(xticks, minor=True)
    # Clean up the appearance
    ax.tick_params(axis='x', which='minor', length=3, width=1)

    # Change the colour of the boxes to Seaborn's 'pastel' palette
    colors = sns.color_palette('Blues')
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    # Colour of the median lines
    plt.setp(bp['medians'], color='k')

    # Check for statistical significance
    significant_combinations = []
    
    # Check from the outside pairs of boxes inwards
    ls = list(range(1, data.shape[1] + 1))
    
    combinations = [(ls[x], ls[x + y]) for y in reversed(ls) for x in range((len(ls) - y))]
    
    for c in combinations:
        data1 = data[:,c[0]-1]
        data2 = data[:,c[1]-1]
        #print(data2.shape)
        # Significance
        U, p = stats.mannwhitneyu(data1, data2, alternative='two-sided')
        if p < 0.05:
            significant_combinations.append([c, p])
            

    # Get info about y-axis
    bottom, top = ax.get_ylim()
    if bottom < 0:
        bottom = 0
    yrange = top - bottom

    # Significance bars
    for i, significant_combination in enumerate(significant_combinations):
        # Columns corresponding to the datasets of interest
        x1 = significant_combination[0][0]
        x2 = significant_combination[0][1]
        # What level is this bar among the bars above the plot?
        level = len(significant_combinations) - i
        # Plot the bar
        bar_height = (yrange * 0.08 * level) + top
        bar_tips = bar_height - (yrange * 0.02)
        plt.plot(
            [x1, x1, x2, x2],
            [bar_tips, bar_height, bar_height, bar_tips], lw=1, c='k')
        # Significance level
        p = significant_combination[1]
        if p < 0.001:
            sig_symbol = '***'
        elif p < 0.01:
            sig_symbol = '**'
        elif p < 0.05:
            sig_symbol = '*'
        text_height = bar_height + (yrange * 0.01)
        plt.text((x1 + x2) * 0.5, text_height, sig_symbol, ha='center', c='k',fontsize=10)

    # Adjust y-axis
    bottom, top = ax.get_ylim()
    if bottom < 0:
        bottom = 0
    yrange = top - bottom
    ax.set_ylim(bottom - 0.1 * yrange, top)
    

    # Annotate sample size below each box
    #for i, dataset in enumerate(data):
        #sample_size = len(dataset)
        #ax.text(i + 1, bottom, fr'n = {sample_size}', ha='center', size='x-small')

    plt.savefig(path,bbox_inches='tight')
    

def create_site_truepred_figures(true, pred,site_size,results_path,site):

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


def generate_wb_files(atlas, results_path, output_path, analysis_id, y_var,chaco_type, subset, model_tested,crossval):
    os.chdir('/home/ubuntu/enigma/motor_predictions/wb_files/workbench_ubuntu')

    atlas_dir = '/home/ubuntu/enigma/motor_predictions/wb_files/'
    image_dir = '/home/ubuntu/enigma/motor_predictions/wb_files/'
    hcp_dir =  '/home/ubuntu/enigma/motor_predictions/wb_files/HCP_S1200_GroupAvg_v1'
    txt_dir = '/home/ubuntu/enigma/results/analysis_1/'
    workbench_dir =  '/home/ubuntu/enigma/motor_predictions/wb_files/workbench_ubuntu'
    

    
    textfiles = ['meanfeatureweight_allperms_50', 'meanfeatureweight_allperms_90', 'meanfeatureweight_allperms_99']
    
    for file in textfiles:
        
        textfile = results_path + output_path + '/{}_{}_{}_{}_{}_crossval{}_{}.txt'.format(atlas, y_var, chaco_type, subset, model_tested[0],crossval, file)
        
        # load text file in fs86 parc (86,1) vector/shen268 parc (268,1)
        scalar = np.genfromtxt(textfile, dtype = "float32", delimiter = ',', usecols = 0)
        print(scalar.shape)
        
        # freesurfer86 region
        if scalar.shape[0]==86:
            # fs86 parcellation - surface files
            atlas_file = nib.load(atlas_dir + 'fs86_dil1_allsubj_mode.nii.gz').get_fdata()
            nodes = np.unique(atlas_file)
            nodes = np.delete(nodes,0)
            data = np.zeros(atlas_file.shape, dtype=np.float32)
            for i,n in enumerate(nodes):
                data[atlas_file == n] = scalar[i]
                
            sample_img = nib.load(atlas_dir + 'fs86_dil1_allsubj_mode.nii.gz')
            save_file = results_path + '/'+ analysis_id + '/{}_{}_{}_{}_{}_crossval{}_{}_surfacefile.nii.gz'.format(atlas, y_var, chaco_type, subset, model_tested[0],crossval, file)
            
            # store nifti header info for saving file
            save_img = nib.Nifti1Image(data, sample_img.affine, sample_img.header)
            
            save_img.set_data_dtype(data.dtype)
            
            nib.save(save_img, save_file) 
            
            filename = save_file
            surf_prefix = filename.replace('.nii.gz', '')
            
            for hemi in ['L', 'R']: 
                os.chdir(workbench_dir + '/bin_linux64')
                cmd = './wb_command -volume-to-surface-mapping '+  filename + ' ' + hcp_dir + '/S1200.' + hemi + '.midthickness_MSMAll.32k_fs_LR.surf.gii '+ surf_prefix + hemi + '.shape.gii -enclosing'
                os.system(cmd)
                #cmd = './wb_command -metric-dilate ' + surf_prefix + hemi + '.shape.gii' + ' ' + hcp_dir + '/S1200.' + hemi + '.midthickness_MSMAll.32k_fs_LR.surf.gii 20 ' + surf_prefix + hemi + '_filled.shape.gii -nearest'
                #os.system(cmd)

            os.remove(surf_prefix + '.nii.gz')
            
            
            # fs86 parcellation - subcortical volume ------------
            os.chdir('/home/ubuntu/enigma/motor_predictions/wb_files/workbench_ubuntu')

            roivol = nib.load(atlas_dir + "fs86_dil1_allsubj_mode.nii.gz")
            Vroi = roivol.get_fdata()
            Vnew = np.zeros(Vroi.shape)
            roidata = np.genfromtxt( textfile, dtype = "float32", delimiter = ',', usecols = 0)

            for i,v in enumerate(np.unique(Vroi[Vroi>0])):
                Vnew[Vroi == v] = roidata[i]
            imgnew = nib.Nifti1Image(Vnew, affine = roivol.affine, header = roivol.header)
            
            nib.save(imgnew, "subcortical_volumes.nii.gz")
            datapath =  "subcortical_volumes.nii.gz"
            newdata = nib.load("subcortical_volumes.nii.gz")
            
            cc400 = nib.load(atlas_dir + "fs86_dil1_allsubj_mode.nii.gz")
            atlas2 = nib.load(atlas_dir + "fs86_dil1_allsubj_mode_subcort.nii.gz")
            
            atlas1=nibabel.processing.resample_from_to(atlas2, cc400, order=0)
            
            V400 = cc400.get_fdata()
            V1 = atlas1.get_fdata()
            
            subcortvals = np.unique(V400[(V400>0) * (V1>0)])
            V400_subcort = V400 * np.isin(V400,subcortvals)
                   
            imgdata = newdata.get_fdata()            
            Vnew = np.double(imgdata*(V400_subcort>0))
            
            cc400.header.set_data_dtype(np.float64)  # without this the visualization breaks. LOVE IT!
            
            newdata_subcort=nib.Nifti1Image(Vnew, affine=cc400.affine, header=cc400.header)

            save_file = results_path + '/'+ analysis_id + '/{}_{}_{}_{}_{}_crossval{}_{}_subcortical_file.nii.gz'.format(atlas, y_var, chaco_type, subset, model_tested[0],crossval, file)

            nib.save(newdata_subcort, save_file)

        elif scalar.shape[0]==268:
            # fs86 parcellation - surface files
            atlas_file = nib.load(atlas_dir + 'shen268_MNI1mm_dil1.nii.gz').get_fdata()
            nodes = np.unique(atlas_file)
            nodes = np.delete(nodes,0)
            data = np.zeros(atlas_file.shape, dtype=np.float32)
            for i,n in enumerate(nodes):
                data[atlas_file == n] = scalar[i]
                
            sample_img = nib.load(atlas_dir + 'shen268_MNI1mm_dil1.nii.gz')
            save_file = results_path + '/'+ analysis_id + '/{}_{}_{}_{}_{}_crossval{}_{}_surfacefile.nii.gz'.format(atlas, y_var, chaco_type, subset, model_tested[0],crossval, file)
            
            # store nifti header info for saving file
            save_img = nib.Nifti1Image(data, sample_img.affine, sample_img.header)
            
            save_img.set_data_dtype(data.dtype)
            
            nib.save(save_img, save_file) 
            
            filename = save_file
            surf_prefix = filename.replace('.nii.gz', '')
            
            for hemi in ['L', 'R']: 
                os.chdir(workbench_dir + '/bin_linux64')
                cmd = './wb_command -volume-to-surface-mapping '+  filename + ' ' + hcp_dir + '/S1200.' + hemi + '.midthickness_MSMAll.32k_fs_LR.surf.gii '+ surf_prefix + hemi + '.shape.gii -enclosing'
                os.system(cmd)
                #cmd = './wb_command -metric-dilate ' + surf_prefix + hemi + '.shape.gii' + ' ' + hcp_dir + '/S1200.' + hemi + '.midthickness_MSMAll.32k_fs_LR.surf.gii 20 ' + surf_prefix + hemi + '_filled.shape.gii -nearest'
                #os.system(cmd)

            os.remove(surf_prefix + '.nii.gz')
            
            
            # fs86 parcellation - subcortical volume ------------
            os.chdir('/home/ubuntu/enigma/motor_predictions/wb_files/workbench_ubuntu')

            roivol = nib.load(atlas_dir + "shen268_MNI1mm_dil1.nii.gz")
            Vroi = roivol.get_fdata()
            Vnew = np.zeros(Vroi.shape)
            roidata = np.genfromtxt( textfile, dtype = "float32", delimiter = ',', usecols = 0)

            for i,v in enumerate(np.unique(Vroi[Vroi>0])):
                Vnew[Vroi == v] = roidata[i]
            imgnew = nib.Nifti1Image(Vnew, affine = roivol.affine, header = roivol.header)
            
            nib.save(imgnew, "subcortical_volumes.nii.gz")
            datapath =  "subcortical_volumes.nii.gz"
            newdata = nib.load("subcortical_volumes.nii.gz")
            
            cc400 = nib.load(atlas_dir + "shen268_MNI1mm_dil1.nii.gz")
            atlas2 = nib.load(atlas_dir + "shen268_MNI1mm_dil1_subcort.nii")
            
            atlas1=nibabel.processing.resample_from_to(atlas2, cc400, order=0)
            
            V400 = cc400.get_fdata()
            V1 = atlas1.get_fdata()
            
            subcortvals = np.unique(V400[(V400>0) * (V1>0)])
            V400_subcort = V400 * np.isin(V400,subcortvals)
                   
            imgdata = newdata.get_fdata()            
            Vnew = np.double(imgdata*(V400_subcort>0))
            
            cc400.header.set_data_dtype(np.float64)  # without this the visualization breaks. LOVE IT!
            
            newdata_subcort=nib.Nifti1Image(Vnew, affine=cc400.affine, header=cc400.header)

            save_file = results_path + '/'+ analysis_id + '/{}_{}_{}_{}_{}_crossval{}_{}_subcortical_file.nii.gz'.format(atlas, y_var, chaco_type, subset, model_tested[0],crossval, file)

            nib.save(newdata_subcort, save_file)
            
        
def generate_wb_figures_setup(hcpdir, scenesdir):
    # update MNI volumetric template path (rel path stored in wb)
    # Read in the file
    with open(scenesdir + 'subcort_scene.scene', "r") as f:
        scenefile = f.read()
        scenefile = scenefile.replace("HCP_S1200_GroupAvg_v1/","")
        scenefile = scenefile.replace('S1200_AverageT1w_restore.nii.gz',hcpdir + 'S1200_AverageT1w_restore.nii.gz')
    # Write the file out again
    with open(scenesdir + 'subcort_scene_edit.scene', 'w') as f:
        f.write(scenefile)
        
    # update midthickness surface template path (rel path stored in wb)
    # Read in the file
    with open(scenesdir + 'landscape_surfaces.scene', "r") as f:
        scenefile = f.read()
        scenefile = scenefile.replace("HCP_S1200_GroupAvg_v1/","")

        scenefile = scenefile.replace("S1200.L.inflated_MSMAll.32k_fs_LR.surf.gii",hcpdir + "S1200.L.inflated_MSMAll.32k_fs_LR.surf.gii")
        scenefile = scenefile.replace("S1200.R.inflated_MSMAll.32k_fs_LR.surf.gii",hcpdir + "S1200.R.inflated_MSMAll.32k_fs_LR.surf.gii")
        scenefile = scenefile.replace('shen268_normed_motor_scores_chacovol_chronic_ridge_crossval1_meanfeatureweight_allperms_50_surfacefileL.shape.gii', 'surfL.gii')
        
        scenefile = scenefile.replace('shen268_normed_motor_scores_chacovol_chronic_ridge_crossval1_meanfeatureweight_allperms_50_surfacefileR.shape.gii', 'surfR.gii')
    # Write the file out again
    with open(scenesdir + 'landscape_surfaces_edit.scene', 'w') as f:
        f.write(scenefile)
        
    with open(scenesdir + 'row_surfaces.scene', "r") as f:
        scenefile = f.read()
        scenefile = scenefile.replace("HCP_S1200_GroupAvg_v1/","")

        scenefile = scenefile.replace("S1200.L.inflated_MSMAll.32k_fs_LR.surf.gii",hcpdir + "S1200.L.inflated_MSMAll.32k_fs_LR.surf.gii")
        scenefile = scenefile.replace("S1200.R.inflated_MSMAll.32k_fs_LR.surf.gii",hcpdir + "S1200.R.inflated_MSMAll.32k_fs_LR.surf.gii")
        scenefile = scenefile.replace('surfmetadataL.shape.gii', 'surfL.gii')
        
        scenefile = scenefile.replace('surfmetadataR.shape.gii', 'surfR.gii')
    # Write the file out again
    with open(scenesdir + 'row_surfaces_edit.scene', 'w') as f:
        f.write(scenefile)

        
    #these are the scene templates that will be used to create figures for the shen/fs feature weights.


def generate_wb_figures(atlas, results_path, analysis_id, y_var,chaco_type, subset, model_tested,crossval,scenesdir, wbpath):
    
    textfiles = ['meanfeatureweight_allperms_50', 'meanfeatureweight_allperms_90', 'meanfeatureweight_allperms_99']

    for file in textfiles:

        subcortical_file = results_path + '/'+ analysis_id + '/{}_{}_{}_{}_{}_crossval{}_{}_subcortical_file.nii.gz'.format(atlas, y_var, chaco_type, subset, model_tested[0],crossval, file)
        surface_fileR = results_path + '/'+ analysis_id + '/{}_{}_{}_{}_{}_crossval{}_{}_surfacefileR.shape.gii'.format(atlas, y_var, chaco_type, subset, model_tested[0],crossval, file)
        surface_fileL = results_path + '/'+ analysis_id + '/{}_{}_{}_{}_{}_crossval{}_{}_surfacefileL.shape.gii'.format(atlas, y_var, chaco_type, subset, model_tested[0],crossval, file)
        
        # because workbench is sooo intuitive, palette/visualization settings are stored in the nifti/gifti metadata!
        # have to rewrite the nifti files with nifti header info derived from manually setting the palette in workbench.
        niftimeta = nib.load(scenesdir + 'niftimetadata.nii.gz')

        newsubcortfile = nib.Nifti1Image(nib.load(subcortical_file).get_fdata(), niftimeta.affine, niftimeta.header)
        nib.save(newsubcortfile, subcortical_file)
        
        #workbench changes the dataarray metadata.
        # keep the metadata for reference gifti, replace with data from actual feature file
        giftimetaR = nib.load(scenesdir + 'surfmetadataR.shape.gii') # reference gifti
        giftimetaR.darrays[0].data = nib.load(surface_fileR).darrays[0].data
        newgifti = nib.gifti.gifti.GiftiImage(header=giftimetaR.header, extra=None, file_map = giftimetaR.file_map, labeltable=giftimetaR.labeltable, darrays=giftimetaR.darrays, meta = giftimetaR.meta, version='1.0')
        nib.save(newgifti, surface_fileR)
        giftimetaL = nib.load(scenesdir + 'surfmetadataL.shape.gii') # reference gifti
        giftimetaL.darrays[0].data = nib.load(surface_fileL).darrays[0].data
        newgifti = nib.gifti.gifti.GiftiImage(header=giftimetaL.header, extra=None, file_map = giftimetaL.file_map, labeltable=giftimetaL.labeltable, darrays=giftimetaL.darrays, meta = giftimetaL.meta, version='1.0')
        nib.save(newgifti, surface_fileL)

        # make copy of the scenes file that we modify for each figure.
        shutil.copy(scenesdir+'subcort_scene_edit.scene', results_path+ '/'+  analysis_id +'/subcortical_scene.scene') 
        shutil.copy(scenesdir+'landscape_surfaces_edit.scene', results_path+ '/'+  analysis_id +'/surfaces_scene.scene') 
        shutil.copy(scenesdir+'row_surfaces_edit.scene', results_path+ '/'+  analysis_id +'/rowsurfaces_scene.scene') 

        # replace volume/surface files with specific results files.
        with open(results_path+ '/'+  analysis_id +'/subcortical_scene.scene', "r") as f:
            scenefile = f.read()
            scenefile = scenefile.replace('subcortical_volumes.nii.gz',subcortical_file)
        with open(results_path+ '/'+  analysis_id +'/subcortical_scene.scene', 'w') as f:
            f.write(scenefile)
            
        with open(results_path+ '/'+  analysis_id +'/surfaces_scene.scene', "r") as f:
            scenefile = f.read()
            scenefile = scenefile.replace('surfL.gii',surface_fileL)
            scenefile = scenefile.replace('surfR.gii',surface_fileR)
        with open(results_path+ '/'+  analysis_id +'/surfaces_scene.scene', 'w') as f:
            f.write(scenefile)
        
        with open(results_path+ '/'+  analysis_id +'/rowsurfaces_scene.scene', "r") as f:
            scenefile = f.read()
            scenefile = scenefile.replace('surfL.gii',surface_fileL)
            scenefile = scenefile.replace('surfR.gii',surface_fileR)
        with open(results_path+ '/'+  analysis_id +'/rowsurfaces_scene.scene', 'w') as f:
            f.write(scenefile)
                
        # subcortical scene
        
        figurefile = results_path + '/'+ analysis_id + '/{}_{}_{}_{}_{}_crossval{}_{}_subcortical_fig.png'.format(atlas, y_var, chaco_type, subset, model_tested[0],crossval, file)
        scenefile = results_path+ '/'+  analysis_id +'/subcortical_scene.scene'
        print('Generating workbench figures:\n {}'.format(figurefile))
        os.system('bash {}/wb_command -show-scene {} 1 {} 1500 300'.format(wbpath, scenefile, figurefile))
            
        # surface scene2
        
        figurefile = results_path + '/'+ analysis_id + '/{}_{}_{}_{}_{}_crossval{}_{}_surfaces_fig.png'.format(atlas, y_var, chaco_type, subset, model_tested[0],crossval, file)
        scenefile = results_path+ '/'+  analysis_id +'/surfaces_scene.scene'
        print('Generating workbench figures:\n {}'.format(figurefile))
        os.system('bash {}/wb_command -show-scene {} 1 {} 1300 900'.format(wbpath, scenefile, figurefile))
        
        figurefile = results_path + '/'+ analysis_id + '/{}_{}_{}_{}_{}_crossval{}_{}_rowsurfaces_fig.png'.format(atlas, y_var, chaco_type, subset, model_tested[0],crossval, file)
        scenefile = results_path+ '/'+  analysis_id +'/rowsurfaces_scene.scene'
        print('Generating workbench figures:\n {}'.format(figurefile))
        os.system('bash {}/wb_command -show-scene {} 1 {} 1300 250'.format(wbpath, scenefile, figurefile))
        
        
def generate_smatt_ll_figures(results_path,analysis_id, output_path, atlas, y_var, chaco_type, subset, model_tested, crossval):
    
    rootname_truepred = results_path + output_path + '/{}_{}_{}_{}_{}_crossval{}'.format(atlas, y_var, chaco_type, subset, model_tested[0],crossval)
    meanfeatureweights = np.loadtxt(rootname_truepred +'_meanfeatureweight_allperms.txt')
    
    # Change the colour of the boxes to match the SMATT figure.
    m1 = (89/255, 196/255, 89/255) # green
    s1 = (228/255, 212/255, 74/255) # yellow
    pmd = (219/255, 100/255, 221/255) # pink
    pmv = (236/255, 152/255, 81/255) # orange
    sma = (76/255, 73/255, 231/255) # blue
    psma = (221/255, 52/255, 50/255) # red
    
    xticklabels = ['M1','PMd','PMv','S1','SMA','preSMA']

    title = ''
    path = results_path + '/' + analysis_id 
    print('Saving figures with filename: {}'.format(path))
    ylabel = 'Feature importance (haufe-transformed)'
    path_file = path + '/'+ analysis_id + '_smatt_featureimpt.png'
    
    colors = (m1, pmd, pmv, s1, sma, psma)
    data = meanfeatureweights
   
    fig, ax = plt.subplots(ncols=1, figsize =(7, 7))
    
    bp = ax.bar([0, 1, 2, 3, 4, 5], data)
    # Graph title
    ax.set_title(title, fontsize=14)
    # Label y-axis
    ax.set_ylabel(ylabel,fontsize=14)
    # Label x-axis ticks
    ax.set_xticklabels(xticklabels,rotation=90,fontsize=14)

    # Hide x-axis major ticks
    ax.tick_params(axis='x', which='major', length=0)
    # Show x-axis minor ticks
    xticks = [0.5] + [x + 0.5 for x in ax.get_xticks()]
    ax.set_xticks(range(len(data)))
    # Clean up the appearance
    ax.tick_params(axis='x', which='minor', length=3, width=1)

    for patch, color in zip(bp, colors):
       patch.set_facecolor(color)       
       
    plt.savefig(path_file, bbox_inches ='tight')