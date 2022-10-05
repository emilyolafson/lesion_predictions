library(R.matlab)
library(ggplot2)
library(ggsignif)
library(ggthemes)

r2scores_fs86subj_chacovol_1<-rowMeans(read.table("/Users/emilyolafson/GIT/ENIGMA/enigma_disconnections/results/r2scores_fs86subj_chacovol_1.txt",sep = ","))
null_r2scores_fs86subj_chacovol_1<-read.table("/Users/emilyolafson/GIT/ENIGMA/enigma_disconnections/results/null_r2scores_fs86subj_chacovol_1.txt",sep = ",") 
r2scores_fs86subj_chacoconn_1<-rowMeans(read.table("/Users/emilyolafson/GIT/ENIGMA/enigma_disconnections/results/r2scores_fs86subj_chacoconn_1.txt",sep = ",")) 

r2scores_fs86subj_chacovol_5<-rowMeans(read.table("/Users/emilyolafson/GIT/ENIGMA/enigma_disconnections/results/r2scores_fs86subj_chacovol_5.txt",sep = ",") )
null_r2scores_fs86subj_chacovol_5<-read.table("/Users/emilyolafson/GIT/ENIGMA/enigma_disconnections/results/null_r2scores_fs86subj_chacovol_5.txt",sep = ",") 
r2scores_fs86subj_chacoconn_5<-rowMeans(read.table("/Users/emilyolafson/GIT/ENIGMA/enigma_disconnections/results/r2scores_fs86subj_chacoconn_5.txt",sep = ",")) 

corrs_fs86subj_chacovol_1<-rowMeans(read.table("/Users/emilyolafson/GIT/ENIGMA/enigma_disconnections/results/corrs_fs86subj_chacovol_1.txt",sep = ",") )
null_corrs_fs86subj_chacovol_1<-read.table("/Users/emilyolafson/GIT/ENIGMA/enigma_disconnections/results/null_corrs_fs86subj_chacovol_1.txt",sep = ",") 
corrs_fs86subj_chacoconn_1<-rowMeans(read.table("/Users/emilyolafson/GIT/ENIGMA/enigma_disconnections/results/corrs_fs86subj_chacoconn_1.txt",sep = ",") )

corrs_fs86subj_chacovol_5<-rowMeans(read.table("/Users/emilyolafson/GIT/ENIGMA/enigma_disconnections/results/corrs_fs86subj_chacovol_5.txt",sep = ",") )
null_corrs_fs86subj_chacovol_5<-read.table("/Users/emilyolafson/GIT/ENIGMA/enigma_disconnections/results/null_corrs_fs86subj_chacovol_5.txt",sep = ",")  
corrs_fs86subj_chacoconn_5<-rowMeans(read.table("/Users/emilyolafson/GIT/ENIGMA/enigma_disconnections/results/corrs_fs86subj_chacoconn_5.txt",sep = ",") )

r2scores_shen268_chacovol_1<-rowMeans(read.table("/Users/emilyolafson/GIT/ENIGMA/enigma_disconnections/results/r2scores_shen268_chacovol_1.txt",sep = ",") )
null_r2scores_shen268_chacovol_1<-read.table("/Users/emilyolafson/GIT/ENIGMA/enigma_disconnections/results/null_r2scores_shen268_chacovol_1.txt",sep = ",") 
r2scores_shen268_chacoconn_1<-rowMeans(read.table("/Users/emilyolafson/GIT/ENIGMA/enigma_disconnections/results/r2scores_shen268_chacoconn_1.txt",sep = ",") )

r2scores_shen268_chacovol_5<-rowMeans(read.table("/Users/emilyolafson/GIT/ENIGMA/enigma_disconnections/results/r2scores_shen268_chacovol_5.txt",sep = ",") )
null_r2scores_shen268_chacovol_5<-read.table("/Users/emilyolafson/GIT/ENIGMA/enigma_disconnections/results/null_r2scores_shen268_chacovol_5.txt",sep = ",") 
r2scores_shen268_chacoconn_5<-rowMeans(read.table("/Users/emilyolafson/GIT/ENIGMA/enigma_disconnections/results/r2scores_shen268_chacoconn_5.txt",sep = ",") )

corrs_shen268_chacovol_1<-rowMeans(read.table("/Users/emilyolafson/GIT/ENIGMA/enigma_disconnections/results/corrs_shen268_chacovol_1.txt",sep = ",") )
null_corrs_shen268_chacovol_1<-read.table("/Users/emilyolafson/GIT/ENIGMA/enigma_disconnections/results/null_corrs_shen268_chacovol_1.txt",sep = ",") 
corrs_shen268_chacoconn_1<-rowMeans(read.table("/Users/emilyolafson/GIT/ENIGMA/enigma_disconnections/results/corrs_shen268_chacoconn_1.txt",sep = ","))

corrs_shen268_chacovol_5<-rowMeans(read.table("/Users/emilyolafson/GIT/ENIGMA/enigma_disconnections/results/corrs_shen268_chacovol_5.txt",sep = ",") )
null_corrs_shen268_chacovol_5<-read.table("/Users/emilyolafson/GIT/ENIGMA/enigma_disconnections/results/null_corrs_shen268_chacovol_5.txt",sep = ",") 
corrs_shen268_chacoconn_5<-rowMeans(read.table("/Users/emilyolafson/GIT/ENIGMA/enigma_disconnections/results/corrs_shen268_chacoconn_5.txt",sep = ",") )





len=25

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
clus[1]<-"1_KFold"
clus[2]<-"2_GroupKFold"
clus[3] <-"3_KFold_CONN"
clus[4]<-"4_GroupKFold_CONN"


# Create vector with stroke & control labels for each [23, 24] x 5 (sessions)
len1=25
len2=25
Condition<-NULL
x<-NULL
for(i in 1:4){
  Condition=c(Condition, rep('FreeSurfer-86 region', len1), rep('Shen 268-region', len2))
  x=c(x, rep(clus[i], 50))
}

y<-NULL
y1=r2scores_fs86subj_chacovol_1
y2=r2scores_shen268_chacovol_1
y3=r2scores_fs86subj_chacovol_5
y4=r2scores_shen268_chacovol_5

y5=r2scores_fs86subj_chacoconn_1
y6=r2scores_shen268_chacoconn_1
y7=r2scores_fs86subj_chacoconn_5
y8=r2scores_shen268_chacoconn_5
yall=c(y1, y2,y3,y4,y5,y6,y7,y8)
y=yall

my_data = data.frame(y,x,Condition)


pdf(file='/Users/emilyolafson/GIT/ENIGMA/enigma_disconnections/figures/r2_chacovolchacovonn.pdf', width=10, height=5)

p <- ggplot(my_data, aes(x, y, fill = Condition)) 
p<- p+ geom_boxplot(binaxis='y', binwidth=0.01,stackdir='center',width = 0.5,position=position_dodge(1))
p + ggtitle("") +scale_fill_manual(values=c("#2b7fed", "#2bbfec")) 

dev.off()

## Correlation

y<-NULL
y1=corrs_fs86subj_chacovol_1
y2=corrs_shen268_chacovol_1
y3=corrs_fs86subj_chacovol_5
y4=corrs_shen268_chacovol_5

y5=corrs_fs86subj_chacoconn_1
y6=corrs_shen268_chacoconn_1
y7=corrs_fs86subj_chacoconn_5
y8=corrs_shen268_chacoconn_5
yall=c(y1, y2,y3,y4,y5,y6,y7,y8)
y=yall

my_data = data.frame(y,x,Condition)
pdf(file='/Users/emilyolafson/GIT/ENIGMA/enigma_disconnections/figures/corrs_chacovolchacovonn.pdf', width=10, height=5)

p <- ggplot(my_data, aes(x, y, fill = Condition)) 
p<- p+ geom_boxplot(binaxis='y', binwidth=0.01,stackdir='center',width = 0.5,position=position_dodge(1))
p + ggtitle("") +scale_fill_manual(values=c("#2b7fed", "#2bbfec")) 

dev.off()



#KW Test ----------
# r2 values.
clus=character()
clus[1]<-"ChaCoVol"
clus[2]<-"ChaCoVol"
clus[3] <-"ChaCoConn"
clus[4]<-"ChaCoConn"

clus2=character()
clus2[1]<-"KFold"
clus2[2]<-"GroupKFold"
clus2[3] <-"KFold"
clus2[4]<-"GroupKFold"

# Create vector with stroke & control labels for each [23, 24] x 5 (sessions)
len1=25
len2=25
atlas<-NULL
chacotype<-NULL
crossval<-NULL
for(i in 1:4){
  atlas=c(atlas, rep('FreeSurfer-86 region', len1), rep('Shen 268-region', len2))
  chacotype=c(chacotype, rep(clus[i], 50))
  crossval = c(crossval, rep(clus2[i],50))
}

y<-NULL
y1=r2scores_fs86subj_chacovol_1
y2=r2scores_shen268_chacovol_1
y3=r2scores_fs86subj_chacovol_5
y4=r2scores_shen268_chacovol_5

y5=r2scores_fs86subj_chacoconn_1
y6=r2scores_shen268_chacoconn_1
y7=r2scores_fs86subj_chacoconn_5
y8=r2scores_shen268_chacoconn_5
yall=c(y1, y2,y3,y4,y5,y6,y7,y8)
y=yall

my_data = data.frame(y,chacotype,crossval,atlas)

# tests 1: FS86 vs SHEN268
pvals<-NULL
# a. KFold, ChaCoVol
result <- wilcox.test(my_data[(my_data$crossval=='KFold') & (my_data$chacotype=='ChaCoVol') & (my_data$atlas=='FreeSurfer-86 region'),]$y, my_data[(my_data$crossval=='KFold') & (my_data$chacotype=='ChaCoVol') & (my_data$atlas=='Shen 268-region'),]$y, paired=TRUE) 
pvals <-c(pvals,  result$p.value)

# b. GroupKFold, ChaCoVol
result<-wilcox.test(my_data[(my_data$crossval=='GroupKFold') & (my_data$chacotype=='ChaCoVol') & (my_data$atlas=='FreeSurfer-86 region'),]$y, my_data[(my_data$crossval=='GroupKFold') & (my_data$chacotype=='ChaCoVol') & (my_data$atlas=='Shen 268-region'),]$y, paired=TRUE) 
pvals <-c(pvals,  result$p.value)

# c KFold, ChaCoConn
result <- wilcox.test(my_data[(my_data$crossval=='KFold') & (my_data$chacotype=='ChaCoConn') & (my_data$atlas=='FreeSurfer-86 region'),]$y, my_data[(my_data$crossval=='KFold') & (my_data$chacotype=='ChaCoConn') & (my_data$atlas=='Shen 268-region'),]$y, paired=TRUE) 
pvals <-c(pvals,  result$p.value)

# d. GroupKFold, ChaCoConn
result<-wilcox.test(my_data[(my_data$crossval=='GroupKFold') & (my_data$chacotype=='ChaCoConn') & (my_data$atlas=='FreeSurfer-86 region'),]$y, my_data[(my_data$crossval=='GroupKFold') & (my_data$chacotype=='ChaCoConn') & (my_data$atlas=='Shen 268-region'),]$y, paired=TRUE) 
pvals <-c(pvals,  result$p.value)


# tests 2: KFold vs GroupKFold
pvals2<-NULL
# a. FS86, ChaCoVol
result <- wilcox.test(my_data[(my_data$crossval=='KFold') & (my_data$chacotype=='ChaCoVol') & (my_data$atlas=='FreeSurfer-86 region'),]$y, my_data[(my_data$crossval=='GroupKFold') & (my_data$chacotype=='ChaCoVol') & (my_data$atlas=='Shen 268-region'),]$y, paired=TRUE) 
pvals2 <-c(pvals2,  result$p.value)

# b. FS86, ChaCoConn
result<-wilcox.test(my_data[(my_data$crossval=='KFold') & (my_data$chacotype=='ChaCoConn') & (my_data$atlas=='FreeSurfer-86 region'),]$y, my_data[(my_data$crossval=='GroupKFold') & (my_data$chacotype=='ChaCoConn') & (my_data$atlas=='Shen 268-region'),]$y, paired=TRUE) 
pvals2 <-c(pvals2,  result$p.value)

# c. Shen268, ChaCoVol
result <- wilcox.test(my_data[(my_data$crossval=='KFold') & (my_data$chacotype=='ChaCoVol') & (my_data$atlas=='Shen 268-region'),]$y, my_data[(my_data$crossval=='GroupKFold') & (my_data$chacotype=='ChaCoVol') & (my_data$atlas=='FreeSurfer-86 region'),]$y, paired=TRUE) 
pvals2 <-c(pvals2,  result$p.value)

# d. Shen268, ChaCoConn
result<-wilcox.test(my_data[(my_data$crossval=='KFold') & (my_data$chacotype=='ChaCoConn') & (my_data$atlas=='Shen 268-region'),]$y, my_data[(my_data$crossval=='GroupKFold') & (my_data$chacotype=='ChaCoConn') & (my_data$atlas=='FreeSurfer-86 region'),]$y, paired=TRUE) 
pvals2 <-c(pvals2,  result$p.value)


# tests 3: ChaCoConn Vs ChaCoVol
pvals3<-NULL
# a. GroupKFold, FS86
result <- wilcox.test(my_data[(my_data$crossval=='GroupKFold') & (my_data$chacotype=='ChaCoConn') & (my_data$atlas=='FreeSurfer-86 region'),]$y, my_data[(my_data$crossval=='GroupKFold') & (my_data$chacotype=='ChaCoVol') & (my_data$atlas=='FreeSurfer-86 region'),]$y, paired=TRUE) 
pvals3 <-c(pvals3,  result$p.value)

# b. KFold, FS86
result<-wilcox.test(my_data[(my_data$crossval=='KFold') & (my_data$chacotype=='ChaCoConn') & (my_data$atlas=='FreeSurfer-86 region'),]$y, my_data[(my_data$crossval=='KFold') & (my_data$chacotype=='ChaCoVol') & (my_data$atlas=='FreeSurfer-86 region'),]$y, paired=TRUE) 
pvals3 <-c(pvals3,  result$p.value)

# c.GroupKFold, Shen268
result <- wilcox.test(my_data[(my_data$crossval=='GroupKFold') & (my_data$chacotype=='ChaCoConn') & (my_data$atlas=='Shen 268-region'),]$y, my_data[(my_data$crossval=='GroupKFold') & (my_data$chacotype=='ChaCoVol') & (my_data$atlas=='Shen 268-region'),]$y, paired=TRUE) 
pvals3 <-c(pvals3,  result$p.value)

# d. KFold, Shen268
result<-wilcox.test(my_data[(my_data$crossval=='KFold') & (my_data$chacotype=='ChaCoConn') & (my_data$atlas=='Shen 268-region'),]$y, my_data[(my_data$crossval=='KFold') & (my_data$chacotype=='ChaCoVol') & (my_data$atlas=='Shen 268-region'),]$y, paired=TRUE) 
pvals3 <-c(pvals3,  result$p.value)

pvals_r2_all <- c(pvals, pvals2, pvals3)
pvals_r2_all_FDR <-p.adjust(pvals_r2_all, method = 'fdr', n = length(pvals_r2_all))
# list of tests
tests = c("FS86 vs Shen:  KFold, ChaCoVol", "FS86 vs Shen:  GroupKFold, ChaCoVol", "FS86 vs Shen:  KFold, ChaCoConn",  "FS86 vs Shen:  GroupKFold, ChaCoConn", 
          "KFold vs GroupKFold: FS86, ChaCoVol", "KFold vs GroupKFold: FS86, ChaCoConn", "KFold vs GroupKFold: Shen268, ChaCoVol", "KFold vs GroupKFold: Shen268, ChaCoConn",
          "ChaCoConn Vs ChaCoVol: GroupKFold, FS86","ChaCoConn Vs ChaCoVo: KFold, FS86", "ChaCoConn Vs ChaCoVo: GroupKFold, Shen268", "ChaCoConn Vs ChaCoVo: KFold, Shen268")

r2_comparisons <- cbind(tests, pvals_r2_all_FDR)


#KW Test ----------
# r2 values.
clus=character()
clus[1]<-"ChaCoVol"
clus[2]<-"ChaCoVol"
clus[3] <-"ChaCoConn"
clus[4]<-"ChaCoConn"

clus2=character()
clus2[1]<-"KFold"
clus2[2]<-"GroupKFold"
clus2[3] <-"KFold"
clus2[4]<-"GroupKFold"

# Create vector with stroke & control labels for each [23, 24] x 5 (sessions)
len1=25
len2=25
atlas<-NULL
chacotype<-NULL
crossval<-NULL
for(i in 1:4){
  atlas=c(atlas, rep('FreeSurfer-86 region', len1), rep('Shen 268-region', len2))
  chacotype=c(chacotype, rep(clus[i], 50))
  crossval = c(crossval, rep(clus2[i],50))
}

y<-NULL
y1=corrs_fs86subj_chacovol_1
y2=corrs_shen268_chacovol_1
y3=corrs_fs86subj_chacovol_5
y4=corrs_shen268_chacovol_5

y5=corrs_fs86subj_chacoconn_1
y6=corrs_shen268_chacoconn_1
y7=corrs_fs86subj_chacoconn_5
y8=corrs_shen268_chacoconn_5
yall=c(y1, y2,y3,y4,y5,y6,y7,y8)
y=yall

my_data = data.frame(y,chacotype,crossval,atlas)

# tests 1: FS86 vs SHEN268
pvals<-NULL
# a. KFold, ChaCoVol
result <- wilcox.test(my_data[(my_data$crossval=='KFold') & (my_data$chacotype=='ChaCoVol') & (my_data$atlas=='FreeSurfer-86 region'),]$y, my_data[(my_data$crossval=='KFold') & (my_data$chacotype=='ChaCoVol') & (my_data$atlas=='Shen 268-region'),]$y, paired=TRUE) 
pvals <-c(pvals,  result$p.value)

# b. GroupKFold, ChaCoVol
result<-wilcox.test(my_data[(my_data$crossval=='GroupKFold') & (my_data$chacotype=='ChaCoVol') & (my_data$atlas=='FreeSurfer-86 region'),]$y, my_data[(my_data$crossval=='GroupKFold') & (my_data$chacotype=='ChaCoVol') & (my_data$atlas=='Shen 268-region'),]$y, paired=TRUE) 
pvals <-c(pvals,  result$p.value)

# c KFold, ChaCoConn
result <- wilcox.test(my_data[(my_data$crossval=='KFold') & (my_data$chacotype=='ChaCoConn') & (my_data$atlas=='FreeSurfer-86 region'),]$y, my_data[(my_data$crossval=='KFold') & (my_data$chacotype=='ChaCoConn') & (my_data$atlas=='Shen 268-region'),]$y, paired=TRUE) 
pvals <-c(pvals,  result$p.value)

# d. GroupKFold, ChaCoConn
result<-wilcox.test(my_data[(my_data$crossval=='GroupKFold') & (my_data$chacotype=='ChaCoConn') & (my_data$atlas=='FreeSurfer-86 region'),]$y, my_data[(my_data$crossval=='GroupKFold') & (my_data$chacotype=='ChaCoConn') & (my_data$atlas=='Shen 268-region'),]$y, paired=TRUE) 
pvals <-c(pvals,  result$p.value)


# tests 2: KFold vs GroupKFold
pvals2<-NULL
# a. FS86, ChaCoVol
result <- wilcox.test(my_data[(my_data$crossval=='KFold') & (my_data$chacotype=='ChaCoVol') & (my_data$atlas=='FreeSurfer-86 region'),]$y, my_data[(my_data$crossval=='GroupKFold') & (my_data$chacotype=='ChaCoVol') & (my_data$atlas=='Shen 268-region'),]$y, paired=TRUE) 
pvals2 <-c(pvals2,  result$p.value)

# b. FS86, ChaCoConn
result<-wilcox.test(my_data[(my_data$crossval=='KFold') & (my_data$chacotype=='ChaCoConn') & (my_data$atlas=='FreeSurfer-86 region'),]$y, my_data[(my_data$crossval=='GroupKFold') & (my_data$chacotype=='ChaCoConn') & (my_data$atlas=='Shen 268-region'),]$y, paired=TRUE) 
pvals2 <-c(pvals2,  result$p.value)

# c. Shen268, ChaCoVol
result <- wilcox.test(my_data[(my_data$crossval=='KFold') & (my_data$chacotype=='ChaCoVol') & (my_data$atlas=='Shen 268-region'),]$y, my_data[(my_data$crossval=='GroupKFold') & (my_data$chacotype=='ChaCoVol') & (my_data$atlas=='FreeSurfer-86 region'),]$y, paired=TRUE) 
pvals2 <-c(pvals2,  result$p.value)

# d. Shen268, ChaCoConn
result<-wilcox.test(my_data[(my_data$crossval=='KFold') & (my_data$chacotype=='ChaCoConn') & (my_data$atlas=='Shen 268-region'),]$y, my_data[(my_data$crossval=='GroupKFold') & (my_data$chacotype=='ChaCoConn') & (my_data$atlas=='FreeSurfer-86 region'),]$y, paired=TRUE) 
pvals2 <-c(pvals2,  result$p.value)


# tests 3: ChaCoConn Vs ChaCoVol
pvals3<-NULL
# a. GroupKFold, FS86
result <- wilcox.test(my_data[(my_data$crossval=='GroupKFold') & (my_data$chacotype=='ChaCoConn') & (my_data$atlas=='FreeSurfer-86 region'),]$y, my_data[(my_data$crossval=='GroupKFold') & (my_data$chacotype=='ChaCoVol') & (my_data$atlas=='FreeSurfer-86 region'),]$y, paired=TRUE) 
pvals3 <-c(pvals3,  result$p.value)

# b. KFold, FS86
result<-wilcox.test(my_data[(my_data$crossval=='KFold') & (my_data$chacotype=='ChaCoConn') & (my_data$atlas=='FreeSurfer-86 region'),]$y, my_data[(my_data$crossval=='KFold') & (my_data$chacotype=='ChaCoVol') & (my_data$atlas=='FreeSurfer-86 region'),]$y, paired=TRUE) 
pvals3 <-c(pvals3,  result$p.value)

# c.GroupKFold, Shen268
result <- wilcox.test(my_data[(my_data$crossval=='GroupKFold') & (my_data$chacotype=='ChaCoConn') & (my_data$atlas=='Shen 268-region'),]$y, my_data[(my_data$crossval=='GroupKFold') & (my_data$chacotype=='ChaCoVol') & (my_data$atlas=='Shen 268-region'),]$y, paired=TRUE) 
pvals3 <-c(pvals3,  result$p.value)

# d. KFold, Shen268
result<-wilcox.test(my_data[(my_data$crossval=='KFold') & (my_data$chacotype=='ChaCoConn') & (my_data$atlas=='Shen 268-region'),]$y, my_data[(my_data$crossval=='KFold') & (my_data$chacotype=='ChaCoVol') & (my_data$atlas=='Shen 268-region'),]$y, paired=TRUE) 
pvals3 <-c(pvals3,  result$p.value)


pvals_corr_all <- c(pvals, pvals2, pvals3)
pvals_corr_all_FDR <-p.adjust(pvals_corr_all, method = 'fdr', n = length(pvals_corr_all))

# list of tests
tests = c("FS86 vs Shen:  KFold, ChaCoVol", "FS86 vs Shen:  GroupKFold, ChaCoVol", "FS86 vs Shen:  KFold, ChaCoConn",  "FS86 vs Shen:  GroupKFold, ChaCoConn", 
          "KFold vs GroupKFold: FS86, ChaCoVol", "KFold vs GroupKFold: FS86, ChaCoConn", "KFold vs GroupKFold: Shen268, ChaCoVol", "KFold vs GroupKFold: Shen268, ChaCoConn",
          "ChaCoConn Vs ChaCoVol: GroupKFold, FS86","ChaCoConn Vs ChaCoVo: KFold, FS86", "ChaCoConn Vs ChaCoVo: GroupKFold, Shen268", "ChaCoConn Vs ChaCoVo: KFold, Shen268")

corr_comparisons <- cbind(tests, pvals_corr_all_FDR)





