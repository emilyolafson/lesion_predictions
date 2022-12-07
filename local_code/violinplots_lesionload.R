library(R.matlab)
library(ggplot2)
library(ggsignif)
library(ggthemes)

# ChaCo - data-driven
r2scores_fs86subj_chacovol_1<-rowMeans(read.table("/Users/emilyolafson/GIT/ENIGMA/enigma_disconnections/results/r2scores_fs86subj_chacovol_1.txt",sep = ","))
r2scores_fs86subj_chacovol_5<-rowMeans(read.table("/Users/emilyolafson/GIT/ENIGMA/enigma_disconnections/results/r2scores_fs86subj_chacovol_5.txt",sep = ",") )
corrs_fs86subj_chacovol_1<-rowMeans(read.table("/Users/emilyolafson/GIT/ENIGMA/enigma_disconnections/results/corrs_fs86subj_chacovol_1.txt",sep = ",") )
corrs_fs86subj_chacovol_5<-rowMeans(read.table("/Users/emilyolafson/GIT/ENIGMA/enigma_disconnections/results/corrs_fs86subj_chacovol_5.txt",sep = ",") )
r2scores_shen268_chacovol_1<-rowMeans(read.table("/Users/emilyolafson/GIT/ENIGMA/enigma_disconnections/results/r2scores_shen268_chacovol_1.txt",sep = ",") )
r2scores_shen268_chacovol_5<-rowMeans(read.table("/Users/emilyolafson/GIT/ENIGMA/enigma_disconnections/results/r2scores_shen268_chacovol_5.txt",sep = ",") )
corrs_shen268_chacovol_1<-rowMeans(read.table("/Users/emilyolafson/GIT/ENIGMA/enigma_disconnections/results/corrs_shen268_chacovol_1.txt",sep = ",") )
corrs_shen268_chacovol_5<-rowMeans(read.table("/Users/emilyolafson/GIT/ENIGMA/enigma_disconnections/results/corrs_shen268_chacovol_5.txt",sep = ",") )

# CST template - all ROIs
r2scores_fs86subj_chacovol_1_cstall<-rowMeans(read.table("/Users/emilyolafson/GIT/ENIGMA/enigma_disconnections/results/r2scores_fs86subj_chacovol_1_lesionload.txt",sep = ","))
r2scores_fs86subj_chacovol_5_cstall<-rowMeans(read.table("/Users/emilyolafson/GIT/ENIGMA/enigma_disconnections/results/r2scores_fs86subj_chacovol_5_lesionload.txt",sep = ","))
corrs_fs86subj_chacovol_1_cstall<-rowMeans(read.table("/Users/emilyolafson/GIT/ENIGMA/enigma_disconnections/results/corrs_fs86subj_chacovol_1_lesionload.txt",sep = ","))
corrs_fs86subj_chacovol_5_cstall<-rowMeans(read.table("/Users/emilyolafson/GIT/ENIGMA/enigma_disconnections/results/corrs_fs86subj_chacovol_5_lesionload.txt",sep = ","))
r2scores_shen268_chacovol_1_cstall<-rowMeans(read.table("/Users/emilyolafson/GIT/ENIGMA/enigma_disconnections/results/r2scores_shen268_chacovol_1_lesionload.txt",sep = ","))
r2scores_shen268_chacovol_5_cstall<-rowMeans(read.table("/Users/emilyolafson/GIT/ENIGMA/enigma_disconnections/results/r2scores_shen268_chacovol_5_lesionload.txt",sep = ","))
corrs_shen268_chacovol_1_cstall<-rowMeans(read.table("/Users/emilyolafson/GIT/ENIGMA/enigma_disconnections/results/corrs_shen268_chacovol_1_lesionload.txt",sep = ","))
corrs_shen268_chacovol_5_cstall<-rowMeans(read.table("/Users/emilyolafson/GIT/ENIGMA/enigma_disconnections/results/corrs_shen268_chacovol_5_lesionload.txt",sep = ","))

# CST template - only M1
r2scores_fs86subj_chacovol_1_cst<-rowMeans(read.table("/Users/emilyolafson/GIT/ENIGMA/enigma_disconnections/results/r2scores_fs86subj_chacovol_1_lesionload_cst.txt",sep = ","))
r2scores_fs86subj_chacovol_5_cst<-rowMeans(read.table("/Users/emilyolafson/GIT/ENIGMA/enigma_disconnections/results/r2scores_fs86subj_chacovol_5_lesionload_cst.txt",sep = ","))
corrs_fs86subj_chacovol_1_cst<-rowMeans(read.table("/Users/emilyolafson/GIT/ENIGMA/enigma_disconnections/results/corrs_fs86subj_chacovol_1_lesionload_cst.txt",sep = ","))
corrs_fs86subj_chacovol_5_cst<-rowMeans(read.table("/Users/emilyolafson/GIT/ENIGMA/enigma_disconnections/results/corrs_fs86subj_chacovol_5_lesionload_cst.txt",sep = ","))
r2scores_shen268_chacovol_1_cst<-rowMeans(read.table("/Users/emilyolafson/GIT/ENIGMA/enigma_disconnections/results/r2scores_shen268_chacovol_1_lesionload_cst.txt",sep = ","))
r2scores_shen268_chacovol_5_cst<-rowMeans(read.table("/Users/emilyolafson/GIT/ENIGMA/enigma_disconnections/results/r2scores_shen268_chacovol_5_lesionload_cst.txt",sep = ","))
corrs_shen268_chacovol_1_cst<-rowMeans(read.table("/Users/emilyolafson/GIT/ENIGMA/enigma_disconnections/results/corrs_shen268_chacovol_1_lesionload_cst.txt",sep = ","))
corrs_shen268_chacovol_5_cst<-rowMeans(read.table("/Users/emilyolafson/GIT/ENIGMA/enigma_disconnections/results/corrs_shen268_chacovol_5_lesionload_cst.txt",sep = ","))

#ensemble
corrs_shen268_chacovol_1_ensemble<-rowMeans(read.table("/Users/emilyolafson/GIT/ENIGMA/enigma_disconnections/results/corrs_shen268_chacovol_1_ensemble.txt",sep = ","))
corrs_shen268_chacovol_5_ensemble<-rowMeans(read.table("/Users/emilyolafson/GIT/ENIGMA/enigma_disconnections/results/corrs_shen268_chacovol_5_ensemble.txt",sep = ","))
corrs_fs86subj_chacovol_1_ensemble<-rowMeans(read.table("/Users/emilyolafson/GIT/ENIGMA/enigma_disconnections/results/corrs_fs86subj_chacovol_1_ensemble.txt",sep = ","))
corrs_fs86subj_chacovol_5_ensemble<-rowMeans(read.table("/Users/emilyolafson/GIT/ENIGMA/enigma_disconnections/results/corrs_fs86subj_chacovol_5_ensemble.txt",sep = ","))

len=24

GeomSplitViolin <- ggproto("GeomSplitViolin", GeomViolin, draw_group = function(self, data, ..., draw_quantiles = NULL){
  data <- transform(data, xminv = x - violinwidth * (x - xmin), xmaxv = x + violinwidth * (xmax - x))
  grp <- data[1,'group']
  newdata <- plyr::arrange(transform(data, x = if(grp%%2==1) xminv else xmaxv), if(grp%%2==1) y else -y)
  newdata <- rbind(newdata[1, ], newdata, newdata[nrow(newdata), ], newdata[1, ])
  newdata[c(1,nrow(newdata)-1,nrow(newdata)), 'x'] <- round(newdata[1, 'x']) 
  if (length(draw_quantiles) > 0 & !scales::zero_range(range(data$y))) {
    stopifnot(all(draw_quantiles >= 0), all(draw_quantiles <= 1))
    quantiles <- ggplot2:::create_quantile_segment_frame(data, draw_quantiles)
    aesthetics <- data[rep(1, nrow(quantiles)), setdiff(names(data), c("x", "y")), drop = FALSE]
    aesthetics$alpha <- rep(1, nrow(quantiles))
    both <- cbind(quantiles, aesthetics)
    quantile_grob <- GeomPath$draw_panel(both, ...)
    ggplot2:::ggname("geom_split_violin", grid::grobTree(GeomPolygon$draw_panel(newdata, ...), quantile_grob))
  }
  else {
    ggplot2:::ggname("geom_split_violin", GeomPolygon$draw_panel(newdata, ...))
  }
})

geom_split_violin <- function (mapping = NULL, data = NULL, stat = "ydensity", position = "identity",  ..., draw_quantiles = NULL,
                               trim = TRUE, scale = "area", na.rm = FALSE, show.legend = NA, inherit.aes = TRUE) {
  layer(data = data, mapping = mapping, stat = stat, geom = GeomSplitViolin, position = position, show.legend = show.legend, 
        inherit.aes = inherit.aes, params = list(trim = trim, scale = scale, draw_quantiles = draw_quantiles, na.rm = na.rm, ...))
}


clus=character()
clus[1]<-"1_KFold_M1"
clus[2]<-"2_GroupKFold_M1"
clus[3]<-"3_KFold_AllCST"
clus[4]<-"4_GroupKFold_AllCST"
clus[5]<-"5_KFold_ChaCo"
clus[6]<-"6_GroupKFold_ChaCo"

# Create vector with stroke & control labels for each [23, 24] x 5 (sessions)
len1=24
len2=24
Atlas<-NULL
x<-NULL
for(i in 1:6){
  Atlas=c(Atlas, rep('FreeSurfer-86 region', len1), rep('Shen 268-region', len2))
  x=c(x, rep(clus[i], 48))
}

y<-NULL
y1=r2scores_fs86subj_chacovol_1_cst
y2=r2scores_shen268_chacovol_1_cst
y3=r2scores_fs86subj_chacovol_5_cst
y4=r2scores_shen268_chacovol_5_cst
y5=r2scores_fs86subj_chacovol_1_cstall
y6=r2scores_shen268_chacovol_1_cstall
y7=r2scores_fs86subj_chacovol_5_cstall
y8=r2scores_shen268_chacovol_5_cstall
y9=r2scores_fs86subj_chacovol_1[1:24]
y10=r2scores_shen268_chacovol_1[1:24]
y11=r2scores_fs86subj_chacovol_5[1:24]
y12=r2scores_shen268_chacovol_5[1:24]
yall=c(y1, y2,y3,y4, y5, y6, y7, y8, y9, y10, y11, y12)
y=yall

my_data = data.frame(y,x,Atlas)


pdf(file='/Users/emilyolafson/GIT/ENIGMA/enigma_disconnections/figures/r2_chacovol_lesionload_cst.pdf', width=10, height=5)

p <- ggplot(my_data, aes(x, y, fill = Atlas)) +theme_classic()
p<- p+ geom_boxplot(binaxis='y', binwidth=0.01,stackdir='center',width = 0.5,position=position_dodge(1))
p + ggtitle("") +scale_fill_manual(values=c("#2b7fed", "#2bbfec")) + ylab('Explained variance')+xlab('')

dev.off()

mydata_f86 = my_data[my_data$Atlas == "FreeSurfer-86 region",]
output_fs <- pairwise.wilcox.test(mydata_f86$y, mydata_f86$x, p.adjust.method = "BH", paired =TRUE)
mydata_shen = my_data[my_data$Atlas == "Shen 268-region",]
output_shen <- pairwise.wilcox.test(mydata_shen$y, mydata_shen$x, p.adjust.method = "BH", paired =TRUE)

res <- wilcox.test(mydata_shen$y ~mydata_shen$x,  paired = TRUE)


install.packages('plot.matrix')
library('plot.matrix')
par(mar=c(10.1, 10.1, 10.1, 10.1)) # adapt margins

library(wesanderson)
colnamesz = c("M1 - KFold", "M1 - GroupKFold", "All-CST - KFold", "All-CST - GroupKFold", "ChaCo - KFold")
colnames(output_fs$p.value)<-colnamesz
rownamesz  = c("M1 - GroupKFold", "All-CST - KFold", "All-CST - GroupKFold", "ChaCo - KFold", "ChaCo - GroupKFold")
rownames(output_fs$p.value)<-rownamesz

plot(output_fs$p.value,col=wes_palette("GrandBudapest2", 5, type = "continuous"),digits=3, 
     text.cell=list(cex=0.5), las=2,xlab='', ylab='')


plot(output_fs$p.value<0.05,col=wes_palette("Zissou1", 2, type = "continuous"),
      las=2,xlab='', ylab='')



clus=character()
clus[1]<-"1_KFold_M1"
clus[2]<-"2_GroupKFold_M1"
clus[3]<-"3_KFold_AllCST"
clus[4]<-"4_GroupKFold_AllCST"
clus[5]<-"5_KFold_M1"
clus[6]<-"6_GroupKFold_M1"

# Create vector with stroke & control labels for each [23, 24] x 5 (sessions)
len1=24
len2=24
Atlas<-NULL
x<-NULL
for(i in 1:6){
  Atlas=c(Atlas, rep('FreeSurfer-86 region', len1), rep('Shen 268-region', len2))
  x=c(x, rep(clus[i], 48))
}


y1=corrs_fs86subj_chacovol_1_cst
y2=corrs_shen268_chacovol_1_cst
y3=corrs_fs86subj_chacovol_5_cst
y4=corrs_shen268_chacovol_5_cst
y5=corrs_fs86subj_chacovol_1_cstall
y6=corrs_shen268_chacovol_1_cstall
y7=corrs_fs86subj_chacovol_5_cstall
y8=corrs_shen268_chacovol_5_cstall
y9=corrs_fs86subj_chacovol_1[1:24]
y10=corrs_shen268_chacovol_1[1:24]
y11=corrs_fs86subj_chacovol_5[1:24]
y12=corrs_shen268_chacovol_5[1:24]
yall=c(y1, y2,y3,y4, y5, y6, y7, y8, y9, y10, y11, y12)
y=yall


my_data = data.frame(y,x,Atlas)
pairwise.wilcox.test(my_data$y, my_data$x, p.adjust.method = "BH")


pdf(file='/Users/emilyolafson/GIT/ENIGMA/enigma_disconnections/figures/corelation_chacovol_lesionload_cst.pdf', width=10, height=5)

p <- ggplot(my_data, aes(x, y, fill = Atlas)) +theme_classic()
p<- p+ geom_boxplot(binaxis='y', binwidth=0.01,stackdir='center',width = 0.5,position=position_dodge(1))
p + ggtitle("") +scale_fill_manual(values=c("#2b7fed", "#2bbfec")) + ylab('Perason correlation')+xlab('')

dev.off()






## Ensemble
clus=character()
clus[1]<-"1_KFold_ChaCo"
clus[2]<-"2_GroupKFold_ChaCo"
clus[3]<-"3_KFold_ChaCo_ClinVars"
clus[4]<-"4_GroupKFold_ChaCo_ClinVars"


# Create vector with stroke & control labels for each [23, 24] x 5 (sessions)
len1=24
len2=24
Atlas<-NULL
x<-NULL
for(i in 1:4){
  Atlas=c(Atlas, rep('FreeSurfer-86 region', len1), rep('Shen 268-region', len2))
  x=c(x, rep(clus[i], 48))
}

y<-NULL
y1=corrs_fs86subj_chacovol_1[1:24]
y2=corrs_shen268_chacovol_1[1:24]
y3=corrs_fs86subj_chacovol_5[1:24]
y4=corrs_shen268_chacovol_5[1:24]
y5=corrs_fs86subj_chacovol_1_ensemble
y6=corrs_shen268_chacovol_1_ensemble
y7=corrs_fs86subj_chacovol_5_ensemble
y8=corrs_shen268_chacovol_5_ensemble

yall=c(y1, y2,y3,y4, y5, y6, y7, y8)
y=yall

my_data = data.frame(y,x,Atlas)

p <- ggplot(my_data, aes(x, y, fill = Atlas)) +theme_classic()
p<- p+ geom_boxplot(binaxis='y', binwidth=0.01,stackdir='center',width = 0.5,position=position_dodge(1))
p + ggtitle("") +scale_fill_manual(values=c("#2b7fed", "#2bbfec")) + ylab('Perason correlation')+xlab('')





# tests 1: FS86 vs SHEN268
pvals<-NULL
# a. KFold, ChaCoVol
result <- wilcox.test(my_data[(my_data$crossval=='KFold') & (my_data$chacotype=='ChaCoVol') & (my_data$atlas=='FreeSurfer-86 region'),]$y, my_data[(my_data$crossval=='KFold') & (my_data$chacotype=='ChaCoVol') & (my_data$atlas=='Shen 268-region'),]$y, paired=TRUE) 
pvals <-c(pvals,  result$p.value)
