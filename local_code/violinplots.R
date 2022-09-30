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
clus[1]<-"KFold"
clus[2]<-"GroupKFold"

# Create vector with stroke & control labels for each [23, 24] x 5 (sessions)
len1=25
len2=25
Condition<-NULL
x<-NULL
for(i in 1:2){
  Condition=c(Condition, rep('FreeSurfer-86 region', len1), rep('Shen 268-region', len2))
  x=c(x, rep(clus[i], 50))
}

y<-NULL
y1=r2scores_fs86subj_chacovol_1
y2=r2scores_shen268_chacovol_1
y3=r2scores_fs86subj_chacovol_5
y4=r2scores_shen268_chacovol_5
yall=c(y1, y2,y3,y4)
y=yall

my_data = data.frame(y,x,Condition)
#pdf(file='/Users/emilyolafson/GIT/dynamic-brainstates/results/shen268/state1_ar.pdf', width=10, height=8)

p <- ggplot(my_data, aes(x, y, fill = Condition)) 
p<- p+ geom_boxplot(binaxis='y', binwidth=0.01,stackdir='center',width = 0.5,position=position_dodge(1))
p+theme_classic(base_size = 22,base_line_size = 1) + ggtitle("") #+ylim(1.2,3) #+ylim(0.1, 0.4) #+≈

dev.off()

## Correlation

y<-NULL
y1=corrs_fs86subj_chacovol_1
y2=corrs_shen268_chacovol_1
y3=corrs_fs86subj_chacovol_5
y4=corrs_shen268_chacovol_5
yall=c(y1, y2,y3,y4)
y=yall

my_data = data.frame(y,x,Condition)
#pdf(file='/Users/emilyolafson/GIT/dynamic-brainstates/results/shen268/state1_ar.pdf', width=10, height=8)

p <- ggplot(my_data, aes(x, y, fill = Condition)) 
p<- p+ geom_boxplot(binaxis='y', binwidth=0.01,stackdir='center',width = 0.5,position=position_dodge(1))
p+theme_classic(base_size = 22,base_line_size = 1) + ggtitle("") #+ylim(1.2,3) #+ylim(0.1, 0.4) #+≈



## CHACOCONN--------
# R2
y<-NULL
y1=r2scores_fs86subj_chacoconn_1
y2=r2scores_shen268_chacoconn_1
y3=r2scores_fs86subj_chacoconn_5
y4=r2scores_shen268_chacoconn_5
yall=c(y1, y2,y3,y4)
y=yall

my_data = data.frame(y,x,Condition)
#pdf(file='/Users/emilyolafson/GIT/dynamic-brainstates/results/shen268/state1_ar.pdf', width=10, height=8)

p <- ggplot(my_data, aes(x, y, fill = Condition)) 
p<- p+ geom_boxplot(binaxis='y', binwidth=0.01,stackdir='center',width = 0.5,position=position_dodge(1))
p+theme_classic(base_size = 22,base_line_size = 1) + ggtitle("") #+ylim(1.2,3) #+ylim(0.1, 0.4) #+≈


## Correlation

y<-NULL
y1=corrs_fs86subj_chacoconn_1
y2=corrs_shen268_chacoconn_1
y3=corrs_fs86subj_chacoconn_5
y4=corrs_shen268_chacoconn_5
yall=c(y1, y2,y3,y4)
y=yall

my_data = data.frame(y,x,Condition)
#pdf(file='/Users/emilyolafson/GIT/dynamic-brainstates/results/shen268/state1_ar.pdf', width=10, height=8)

p <- ggplot(my_data, aes(x, y, fill = Condition)) 
p<- p+ geom_boxplot(binaxis='y', binwidth=0.01,stackdir='center',width = 0.5,position=position_dodge(1))
p+theme_classic(base_size = 22,base_line_size = 1) + ggtitle("") #+ylim(1.2,3) #+ylim(0.1, 0.4) #+≈



