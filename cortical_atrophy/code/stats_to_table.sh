 for j in *; do for i in $j/*; do asegstats2table -i $i/aseg.stats --tablefile "$i"_aseg_stats.txt ; done ; done
aparcstats2table --skip --subjects R*/* --hemi rh --tablefile aparc_rh_stats.txt
