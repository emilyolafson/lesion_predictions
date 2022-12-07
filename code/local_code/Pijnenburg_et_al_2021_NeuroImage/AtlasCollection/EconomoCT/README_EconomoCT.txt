Version 1; 15-01-2021; Free to use under a Creative Commons Attribution-NonCommercial-Share A like 4.0 International License.

Package content:
README_EconomoCT							Data version, licensing information and list of brain regions, including for each region: region number, digital atlas label, the corresponding original region
'label' folder								Containing individual manually drawn labels, used in atlas digitisation
lh.atlas.gcs & rh.atlas.gcs						Surface-based atlas
lh.fsaverage.EconomoCT.label.gii & rh.fsaverage.EconomoCT.label.gii 	Surface-based atlas, applied to fsaverage and converted to .gii surface mesh using mris_convert 
EconomoCT.nii.gz							Colin27 volume-based atlas
EconomoCT_NCBI152.nii.gz							NCBI152 volume-based atlas
lh.colortable.txt & rh.colortable.txt					Colortable files for the surface-based atlas
EconomoCTColorLUT.txt							Lookup table, including ROI number and color code for each region

NOTE: 
Anatomically non-contigious areas within each cortical type were segmented as separate labels. For analyses, labels may be merged within each cortical type. 

List of Brain Regions
#	Digital label			Original Region
1	CT1				Cortical Type 1
2	CT1_2_fro_inf			Cortical Type 1/2, Inferior Frontal 
3	CT1_2_mof			Cortical Type 1/2, Medial Orbitofrontal
4	CT1_2_pref_ins_postc		Cortical Type 1/2, Precentral, Insula and Postcentral
5	CT1_2_temp			Cortical Type 1/2, Temporal
6	CT2_fro				Cortical Type 2, Frontal
7	CT2_par				Cortical Type 2, Parietal
8	CT3_fro				Cortical Type 3, Frontal
9	CT3_par				Cortical Type 3, Parietal
10	CT4_occ				Cortical Type 4, Occipital
11	CT4_dmpf			Cortical Type 4, Dorsomedial Prefrontal
12	CT5_cen				Cortical Type 5, Central Sulcus
13	CT5_ins				Cortical Type 5, Insula
14	CT5_med				Cortical Type 5, Medial
15	CT5_occ				Cortical Type 5, Occipital