import numpy as np
import nibabel as nib
import os
import shutil
from PIL import Image, ImageFont, ImageDraw
import nibabel.processing

def plot_workbench(textfile,factor):
    test=2
    atlas_dir='/home/ubuntu/enigma/motor_predictions/wb_files'
    hcp_dir='/home/ubuntu/enigma/motor_predictions/wb_files/HCP_S1200_GroupAvg_v1'
    wbpath='/home/ubuntu/enigma/motor_predictions/wb_files/workbench_ubuntu/bin_linux64'
    scenesdir='/home/ubuntu/enigma/motor_predictions/wb_files'

    results_dir = os.path.dirname(textfile)

    scalar = np.genfromtxt(textfile)

    if scalar.shape[0]==268:
        print('test 2')
        scalar=scalar/factor
        # It first loads the scalar data from the text file, and then loads the atlas file for the shen268 parcellation. 
        atlas_file = nib.load(os.path.join(atlas_dir, 'shen268_MNI1mm_dil1.nii.gz')).get_fdata()
        subcorticalshen = np.loadtxt(os.path.join(atlas_dir, 'shen_subcorticalROIs.txt'))
        
        scalar = scalar[0:268]
        # It then creates an array of nodes by extracting the unique values from the atlas file and removing the value 0. 
        # The code then creates an empty array of data with the same shape as the atlas file.    
        # It then iterates over the nodes, and if the node is not in the list of subcortical regions, it assigns the corresponding
        # value in the scalar array to the data array.    
        nodes = np.unique(atlas_file)
        nodes = np.delete(nodes,0)
        data = np.zeros(atlas_file.shape, dtype=np.float32)
        for i,n in enumerate(nodes):
            if n in subcorticalshen:
                continue
            data[atlas_file == n] = scalar[i]
            
        sample_img = nib.load(os.path.join(atlas_dir, 'shen268_MNI1mm_dil1.nii.gz'))
        save_file = os.path.join(results_dir ,'temp_surfacefile_betas.nii.gz')
        # store nifti header info for saving file
        save_img = nib.Nifti1Image(data, sample_img.affine, sample_img.header)
        
        save_img.set_data_dtype(data.dtype)
        
        nib.save(save_img, save_file) 
        
        filename = save_file
        surf_prefix = filename.replace('.nii.gz', '')
        
        for hemi in ['L', 'R']: 
            os.chdir(wbpath)
            cmd = './wb_command' + ' -volume-to-surface-mapping '+  filename + ' ' + hcp_dir + '/S1200.' + hemi + '.midthickness_MSMAll.32k_fs_LR.surf.gii '+ surf_prefix + hemi + '.shape.gii -enclosing'
            os.system(cmd)


        os.remove(surf_prefix + '.nii.gz')
        # fs86 parcellation - subcortical volume ------------
        os.chdir('/home/ubuntu/enigma/motor_predictions/wb_files/workbench_ubuntu')

        roivol = nib.load(os.path.join(atlas_dir, 'shen268_MNI1mm_dil1.nii.gz'))
        Vroi = roivol.get_fdata()
        Vnew = np.zeros(Vroi.shape)
        roidata = np.genfromtxt( textfile, dtype = "float32", delimiter = ',', usecols = 0)
        roidata = roidata/factor
        
        roidata=roidata[0:268]
        print(roidata.shape)
        
        for i,v in enumerate(np.unique(Vroi[Vroi>0])):
            Vnew[Vroi == v] = roidata[i]
        imgnew = nib.Nifti1Image(Vnew, affine = roivol.affine, header = roivol.header)
        
        nib.save(imgnew, "subcortical_volumes.nii.gz")
        datapath =  "subcortical_volumes.nii.gz"
        newdata = nib.load("subcortical_volumes.nii.gz")
        
        cc400 = nib.load(os.path.join(atlas_dir, 'shen268_MNI1mm_dil1.nii.gz'))
        atlas2 = nib.load(os.path.join(atlas_dir, 'shen268_MNI1mm_dil1_subcort.nii'))
        
        atlas1=nibabel.processing.resample_from_to(atlas2, cc400, order=0)
        
        V400 = cc400.get_fdata()
        V1 = atlas1.get_fdata()
        
        subcortvals = np.unique(V400[(V400>0) * (V1>0)])
        V400_subcort = V400 * np.isin(V400,subcortvals)
                
        imgdata = newdata.get_fdata()            
        Vnew = np.double(imgdata*(V400_subcort>0))
        
        cc400.header.set_data_dtype(np.float64)  # without this the visualization breaks. LOVE IT!
        
        newdata_subcort=nib.Nifti1Image(Vnew, affine=cc400.affine, header=cc400.header)

        save_file = os.path.join(results_dir ,'temp_subcortical_betas.nii.gz')
        print('saving')
        nib.save(newdata_subcort, save_file)
        
        
    # freesurfer86 region
    if scalar.shape[0]==86:
        print('test 3')

        scalar=scalar/factor
        # fs86 parcellation - surface files
        atlas_file = nib.load(os.path.join(atlas_dir,'fs86_dil1_allsubj_mode.nii.gz')).get_fdata()
        nodes = np.unique(atlas_file)
        
        nodes = np.delete(nodes,0)
        data = np.zeros(atlas_file.shape, dtype=np.float32)
        for i,n in enumerate(nodes):
            if n<18:
                continue
            data[atlas_file == n] = scalar[i]
            
        sample_img = nib.load(os.path.join(atlas_dir,'fs86_dil1_allsubj_mode.nii.gz'))
        save_file = os.path.join(results_dir ,'temp_surfacefile_betas.nii.gz')
        
        # store nifti header info for saving file
        save_img = nib.Nifti1Image(data, sample_img.affine, sample_img.header)
        
        save_img.set_data_dtype(data.dtype)
        
        nib.save(save_img, save_file) 
        
        filename = save_file
        surf_prefix = filename.replace('.nii.gz', '')
        
        for hemi in ['L', 'R']:
            os.chdir(wbpath) 
            cmd = './wb_command' + ' -volume-to-surface-mapping '+  filename + ' ' + hcp_dir + '/S1200.' + hemi + '.midthickness_MSMAll.32k_fs_LR.surf.gii '+ surf_prefix + hemi + '.shape.gii -enclosing'
            os.system(cmd)

            #cmd = './wb_command -metric-dilate ' + surf_prefix + hemi + '.shape.gii' + ' ' + hcp_dir + '/S1200.' + hemi + '.midthickness_MSMAll.32k_fs_LR.surf.gii 20 ' + surf_prefix + hemi + '_filled.shape.gii -nearest'
            #os.system(cmd)

        os.remove(surf_prefix + '.nii.gz')
        
        
        # fs86 parcellation - subcortical volume ------------
        os.chdir('/home/ubuntu/enigma/motor_predictions/wb_files/workbench_ubuntu')

        roivol = nib.load(os.path.join(atlas_dir,'fs86_dil1_allsubj_mode.nii.gz'))
        Vroi = roivol.get_fdata()
        Vnew = np.zeros(Vroi.shape)
        roidata = np.genfromtxt( textfile, dtype = "float32", delimiter = ',', usecols = 0)
        roidata=roidata/factor
        for i,v in enumerate(np.unique(Vroi[Vroi>0])):
            Vnew[Vroi == v] = roidata[i]
        imgnew = nib.Nifti1Image(Vnew, affine = roivol.affine, header = roivol.header)
        
        nib.save(imgnew, "subcortical_volumes.nii.gz")
        datapath =  "subcortical_volumes.nii.gz"
        newdata = nib.load("subcortical_volumes.nii.gz")
        
        cc400 = nib.load(os.path.join(atlas_dir,'fs86_dil1_allsubj_mode.nii.gz'))
        atlas2 = nib.load(os.path.join(atlas_dir,'fs86_dil1_allsubj_mode_subcort.nii.gz'))
        
        atlas1=nibabel.processing.resample_from_to(atlas2, cc400, order=0)
        
        V400 = cc400.get_fdata()
        V1 = atlas1.get_fdata()
        
        subcortvals = np.unique(V400[(V400>0) * (V1>0)])
        V400_subcort = V400 * np.isin(V400,subcortvals)
            
        imgdata = newdata.get_fdata()            
        Vnew = np.double(imgdata*(V400_subcort>0))
        
        cc400.header.set_data_dtype(np.float64)  # without this the visualization breaks. LOVE IT!
        
        newdata_subcort=nib.Nifti1Image(Vnew, affine=cc400.affine, header=cc400.header)
        save_file = os.path.join(results_dir ,'temp_subcortical_betas.nii.gz')

        nib.save(newdata_subcort, save_file)
    

    subcortical_file =os.path.join(results_dir ,'temp_subcortical_betas.nii.gz')
    surface_fileL =os.path.join(results_dir ,'temp_surfacefile_betasL.shape.gii')
    surface_fileR = os.path.join(results_dir ,'temp_surfacefile_betasR.shape.gii')
    
    # because workbench is sooo intuitive, palette/visualization settings are stored in the nifti/gifti metadata!
    # have to rewrite the nifti files with nifti header info derived from manually setting the palette in workbench.
    niftimeta = nib.load(os.path.join(scenesdir, 'niftimetadata_POSNEG.nii.gz'))
    newsubcortfile = nib.Nifti1Image(nib.load(subcortical_file).get_fdata(), niftimeta.affine, niftimeta.header)
    nib.save(newsubcortfile, subcortical_file)
    
    giftimetaR = nib.load(os.path.join(scenesdir, 'surfmetadataL_posneg.shape.gii')) # reference gifti
    giftimetaR.darrays[0].data = nib.load(surface_fileR).darrays[0].data
    newgifti = nib.gifti.gifti.GiftiImage(header=giftimetaR.header, extra=None, file_map = giftimetaR.file_map, labeltable=giftimetaR.labeltable, darrays=giftimetaR.darrays, meta = giftimetaR.meta, version='1.0')
    surface_fileRpos = surface_fileR
    nib.save(newgifti, surface_fileRpos)
    giftimetaL = nib.load(os.path.join(scenesdir,'surfmetadataR_posneg.shape.gii')) # reference gifti
    giftimetaL.darrays[0].data = nib.load(surface_fileL).darrays[0].data
    newgifti = nib.gifti.gifti.GiftiImage(header=giftimetaL.header, extra=None, file_map = giftimetaL.file_map, labeltable=giftimetaL.labeltable, darrays=giftimetaL.darrays, meta = giftimetaL.meta, version='1.0')
    surface_fileLpos =surface_fileL
    nib.save(newgifti, surface_fileLpos)
    
    shutil.copy(os.path.join(scenesdir,'subcort_scene_edit.scene'), os.path.join(results_dir,'subcortical_scene.scene'))
    shutil.copy(os.path.join(scenesdir,'landscape_surfaces_edit.scene'), os.path.join(results_dir,'surfaces_scene_pos.scene')) 
    shutil.copy(os.path.join(scenesdir,'dorsal_surface_edit.scene'), os.path.join(results_dir,'dorsalsurfaces_scene_pos.scene')) 
    
    # replace volume/surface files with specific results files.
    with open(os.path.join(results_dir, 'subcortical_scene.scene'), "r") as f:
        scenefile = f.read()
        scenefile = scenefile.replace('subcortical_volumes.nii.gz',subcortical_file)
    with open(os.path.join(results_dir, 'subcortical_scene.scene'), 'w') as f:
        f.write(scenefile)
    
    
    with open(os.path.join(results_dir, 'surfaces_scene_pos.scene'), "r") as f:
        scenefile = f.read()
        scenefile = scenefile.replace('surfL.gii',surface_fileLpos)
        scenefile = scenefile.replace('surfR.gii',surface_fileRpos)
    with open(os.path.join(results_dir, 'surfaces_scene_pos.scene'), 'w') as f:
        f.write(scenefile)  
    
    with open(os.path.join(results_dir, 'dorsalsurfaces_scene_pos.scene'), "r") as f:
        scenefile = f.read()
        scenefile = scenefile.replace('surfL.gii',surface_fileLpos)
        scenefile = scenefile.replace('surfR.gii',surface_fileRpos)
    with open(os.path.join(results_dir, 'dorsalsurfaces_scene_pos.scene'), 'w') as f:
        f.write(scenefile)   
        
    # subcortical scene
    
    figurefile=os.path.join(results_dir ,'temp_subcortical_fig.png')
    scenefile = os.path.join(results_dir,'subcortical_scene.scene')
    print('Generating workbench figures:\n {}'.format(figurefile))
    os.system('bash {}/wb_command -show-scene {} 1 {} 1500 300'.format(wbpath, scenefile, figurefile))
    
    # surface scene2
    figurefile=os.path.join(results_dir ,'temp_surfaces_fig.png')
    scenefile = os.path.join(results_dir,'surfaces_scene_pos.scene')
    print('Generating workbench figures:\n {}'.format(figurefile))
    os.system('bash {}/wb_command -show-scene {} 1 {} 1300 900'.format(wbpath, scenefile, figurefile))

    figurefile=os.path.join(results_dir ,'temp_dorsalsurfaces_fig.png')
    scenefile = os.path.join(results_dir,'dorsalsurfaces_scene_pos.scene')
    print('Generating workbench figures:\n {}'.format(figurefile))
    os.system('bash {}/wb_command -show-scene {} 1 {} 10000 1300'.format(wbpath, scenefile, figurefile))
    
    image1 = make_black_white(crop_dorsal(Image.open(os.path.join(results_dir ,'temp_dorsalsurfaces_fig.png'))))
    image2 = make_black_white(Image.open(os.path.join(results_dir ,'temp_surfaces_fig.png')))
    sub = Image.open(os.path.join(results_dir ,'temp_subcortical_fig.png'))
    
    brains = Image.fromarray(np.hstack(make_same_height([image2, image1])))
    figure = Image.fromarray(np.vstack(make_same_width([brains, sub])))
    figure.save(os.path.join(results_dir ,'final_figure.png'))
    
    return figure



def make_black_white(picture):
    # Get the size of the image
    width, height = picture.size

    # Process every pixel
    for x in range(0,width):
        for y in range(0,height):
            current_color = picture.getpixel((x,y))
            if current_color ==(0,0,0):
                picture.putpixel( (x,y), (255,255,255))

    return picture

def make_white_black(picture):
    # Get the size of the image
    width, height = picture.size

    # Process every pixel
    for x in range(0,width):
        for y in range(0,height):
            current_color = picture.getpixel((x,y))
            if current_color ==(255,255,255):
                picture.putpixel( (x,y), (0,0,0))

    return picture
def change_height_proportionally(img, width):
    """Change height of image proportional to given width."""
    wpercent = width / img.size[0]
    proportional_height = int(img.size[1] * wpercent)
    return img.resize((width, proportional_height), Image.ANTIALIAS)


def change_width_proportionally(img, height):
    """Change width of image proportional to given height."""
    hpercent = height / img.size[1]
    proportional_width = int(img.size[0] * hpercent)
    return img.resize((proportional_width, height), Image.ANTIALIAS)


def make_same_width(image_list):
    """Make all images in input list the same width."""
    imgs = [i for i in image_list]
    min_width = min([i.size[0] for i in imgs])
    resized = [change_height_proportionally(img, min_width) for img in imgs]
    return [np.asarray(i) for i in resized]


def make_same_height(image_list):
    """Make all images in input list the same height."""
    imgs = [i for i in image_list]
    min_height = min([i.size[1] for i in imgs])
    resized = [change_width_proportionally(img, min_height) for img in imgs]
    return [np.asarray(i) for i in resized]


def add_text(img):
    """Add text annotation to hardcoded locations."""
    font = ImageFont.truetype(
        "/usr/share/fonts/truetype/freefont/FreeMonoBold.ttf",
        size=24,
        encoding="unic")
    draw = ImageDraw.Draw(img)
    draw.text((30, 30), "A", (0, 0, 0), font=font)
    draw.text((30, 490), "B", (0, 0, 0), font=font)
    draw.text((30, 950), "C", (0, 0, 0), font=font)
    draw.text((30, 1430), "D", (0, 0, 0), font=font)
    draw.text((510, 30), "E", (0, 0, 0), font=font)
    draw.text((510, 950), "F", (0, 0, 0), font=font)
    
def add_whitespace_between_horiz(image_list, width):
    #eo
    # adds whitespace horizontally two or more figures. builds first->last left->right
    whitespace = np.ones([10000,width,3],dtype=np.uint8)*255
    whitespace = Image.fromarray(whitespace)
    for i in range(0, len(image_list)):
        if i==0:
            imgleft = image_list[i]
            continue
        imgright = image_list[i]
        imgrightwhite = Image.fromarray(np.hstack(make_same_height([whitespace, imgright])))
        imgleft = Image.fromarray(np.hstack(make_same_height([imgleft,imgrightwhite])))
        
    return imgleft
    
def add_whitespace_between_vert(image_list, height):
    #eo
    # adds whitespace vertically between two or more figures. builds first->last top->down
    whitespace = np.ones([height,10000,3],dtype=np.uint8)*255
    whitespace = Image.fromarray(whitespace)
    for i in range(0, len(image_list)):
        if i==0:
            imgtop = image_list[i]
            continue
        imgbottom = image_list[i]
        imgbottomwhite = Image.fromarray(np.vstack(make_same_width([whitespace, imgbottom])))
        imgtop = Image.fromarray(np.vstack(make_same_width([imgtop,imgbottomwhite])))
        
    return imgtop

def add_blackspace_between_vert(image_list, height):
    #eo
    # adds whitespace vertically between two or more figures. builds first->last top->down
    whitespace = np.ones([height,10000,3],dtype=np.uint8)*0
    whitespace = Image.fromarray(whitespace)
    for i in range(0, len(image_list)):
        if i==0:
            imgtop = image_list[i]
            continue
        imgbottom = image_list[i]
        imgbottomwhite = Image.fromarray(np.vstack(make_same_width([whitespace, imgbottom])))
        imgtop = Image.fromarray(np.vstack(make_same_width([imgtop,imgbottomwhite])))
        
    return imgtop

def add_blackspace_between_horiz(image_list, width):
    #eo
    # adds whitespace horizontally two or more figures. builds first->last left->right
    whitespace = np.ones([10000,width,3],dtype=np.uint8)*0
    whitespace = Image.fromarray(whitespace)
    for i in range(0, len(image_list)):
        if i==0:
            imgleft = image_list[i]
            continue
        imgright = image_list[i]
        imgrightwhite = Image.fromarray(np.hstack(make_same_height([whitespace, imgright])))
        imgleft = Image.fromarray(np.hstack(make_same_height([imgleft,imgrightwhite])))
        
    return imgleft

def add_whitespace_below(image, height):
    #eo
    # adds whitespace horizontally two or more figures. builds first->last left->right
    whitespace = np.ones([height,10000,3],dtype=np.uint8)*255
    whitespace = Image.fromarray(whitespace)

    img = Image.fromarray(np.vstack(make_same_width([image, whitespace])))
    return img

def add_whitespace_above(image, height):
    #eo
    # adds whitespace horizontally two or more figures. builds first->last left->right
    whitespace = np.ones([height,10000,3],dtype=np.uint8)*255
    whitespace = Image.fromarray(whitespace)

    img = Image.fromarray(np.vstack(make_same_width([whitespace, image])))
    return img
                          
def crop_dorsal(image1):
    # Cropped image of above dimension
    # (It will not change original image)
    im1 = image1.crop((3300, 0, 3900, 1300))
    im2 = image1.crop((6100, 0, 6700, 1300))

    return Image.fromarray(np.hstack([im1, im2]))
    
def crop_medial_lateral(image2):
    print('cropping medial/lateral')
    im1 = image2.crop((230, 0, 580, 250))
    im2 = image2.crop((730, 0, 1070, 250))

    return Image.fromarray(np.hstack([im1, im2]))