if (!requireNamespace("BiocManager", quietly = TRUE))
library(devtools)
library(plotly)
library(ggfortify)
library(ggplot2)
library(sva)
library(RColorBrewer)
library(gridExtra)
library(neuroCombat)
library(cowplot)
install.packages("invgamma")
library(invgamma)
library(wesanderson)

n <- 30
qual_col_pals = brewer.pal.info[brewer.pal.info$category == 'qual',]
col_vector = unlist(mapply(brewer.pal, qual_col_pals$maxcolors, rownames(qual_col_pals)))

thickness <- read.table('/Users/emilyolafson/GIT/ENIGMA/enigma_disconnections/cortical_atrophy/data/FREESURFER_ENIGMA/enigma_thickness_datasetvar.csv',sep = ",", header=TRUE)

og_colnames <- colnames(thickness)
# setup combat
Subject <- thickness$Subject
dataset <- thickness$dataset
lesionvol <- thickness$LesionVolume
age <- thickness$age
thickness$sex[thickness$sex =="3.0"]=NA
thickness$sex[thickness$sex ==""]=NA
sex <- as.factor(thickness$sex)
tss <- thickness$time_since_stroke
motor_score <- thickness$motor_score

nas <- as.logical(rowSums(is.na(cbind(age, sex, tss, motor_score,lesionvol))))
thickness <- thickness[!nas,]


subject <- thickness$Subject
age <- thickness$age
tss <- thickness$time_since_stroke
sex <- as.factor(thickness$sex)
dxstatus <- as.factor(thickness$disease_status)
motor <- thickness$motor_score
lesionvol <- thickness$LesionVolume

batchvar <- as.vector(thickness$site)
dataset <- as.vector(thickness$dataset)
thickness <- thickness[, 3:73]
thicknesst <-t(thickness)

finaltable <- cbind(thickness,batchvar,dataset,sex,age,motor,lesionvol)

residualized_regionwise_data=matrix(, nrow = 710, ncol = 71)

for (i in 1:71){
  residualized_regionwise_data[,i] <- residuals(lm(unlist(thickness[i]) ~ age + sex + tss +motor +lesionvol))
}

d2<-data.frame(residualized_regionwise_data, batchvar)
clrs <- wes_palette("Zissou1", 28, type = "continuous")

# PCA
pcavar <- prcomp(d2[1:70] , center = TRUE,scale. = TRUE)
dev.on()
pdf(width=20, height=6.4,'/Users/emilyolafson/GIT/ENIGMA/enigma_disconnections/cortical_atrophy/figures/PCA_3PCs_age_sex_tss_motor_lesionvol_regressed_enigma_raw.pdf')

p1<- autoplot(pcavar, x=1, y=2, data = d2,size=3, colour = 'batchvar',alpha=0.5)+theme_classic()+scale_color_manual(values = c(col_vector))+theme(text = element_text(size=30)) + theme(legend.position="none", legend.title = element_blank())
p2<- autoplot(pcavar, x=2, y=3, data = d2,size=3,colour = 'batchvar',alpha=0.5)+theme_classic()+scale_color_manual(values = c(col_vector))+theme(text = element_text(size=30))+ theme(legend.position="none",legend.title = element_blank())
p3<- autoplot(pcavar, x=1, y=3, data = d2,size=3,colour = 'batchvar',alpha=0.5)+theme_classic()+scale_color_manual(values = c(col_vector))+theme(text = element_text(size=30))+ theme(legend.position="right",legend.direction = "vertical",legend.title = element_blank(),legend.text =  element_text(size=10,margin = margin(t = 5)))
plot_grid(p1, p2, p3, ncol=3, rel_widths = c(1,1,1.3),label_size = 30)

dev.off()


# COMBAT
mod =cbind(sex,age,tss,motor,lesionvol)

outputn <- neuroCombat(dat=thicknesst,batch=batchvar,mod=mod)
output <- outputn$dat.combat


# visualize combat method.
pdf(width=10, height=10,'/Users/emilyolafson/GIT/ENIGMA/enigma_disconnections/cortical_atrophy/figures/enigma_empirical_estimated_param_gamma.pdf')

ests <- outputn$estimates
gammahats <- ests$gamma.star
sites <- rownames(gammahats)

clrs <- wes_palette("Zissou1", 28, type = "continuous")

myplots <- list()  # new empty list
for(n in 1:28){
  x <-  seq(-3, 2, length=70)
  gammahat <- as.double(gammahats[n,])
  y <-  dnorm(seq(-3, 2, length=70), mean=ests$gamma.bar[n], sd=sqrt(ests$t2[n]))
  Sites <-sites[n]
 
  df<-data.frame(x,y, gammahat,Sites)
  p1 <- ggplot(df, aes(x=x,color=Sites))+ theme_classic()+
    geom_line(aes(x,y,color=Sites), data = NULL, linetype = "solid",size =0.7)+
    scale_color_manual(values = clrs[n],labels = c('HCP_AGING', 'HCP_YA'))+
    theme(legend.position="none")+
    geom_density(aes(x=gammahat,color=Sites), data = NULL, linetype = "dotted",size =0.7)+
    theme(text = element_text(size=8),plot.title = element_text(hjust = 0.5))+
    ylab('Density')+xlab('Gamma')+
    ggtitle(sites[n])
  
  
  myplots[[n]] <- p1  # add each plot into plot list
  
}
do.call(grid.arrange,  c(myplots,ncol=4))
dev.off()


# visualize combat method.
pdf(width=10, height=10,'/Users/emilyolafson/GIT/ENIGMA/enigma_disconnections/cortical_atrophy/figures/enigma_empirical_estimated_param_delta.pdf')

ests <- outputn$estimates
deltahats <- ests$delta.hat
sites <- rownames(deltahats)

clrs <- wes_palette("Zissou1", 28, type = "continuous")

myplots <- list()  # new empty list
for(n in 1:28){
  x <-  seq(-3,4, length=70)
  deltahat <- as.double(deltahats[n,])
  y <-  dinvgamma(x=seq(-3, 4, length=70), shape=ests$a.prior[n],rate=ests$b.prior[n])
  Sites <-sites[n]
  
  df<-data.frame(x,y, deltahat,Sites)
  p1 <- ggplot(df, aes(x=x,color=Sites))+ theme_classic()+
    geom_line(aes(x,y,color=Sites), data = NULL, linetype = "solid",size =0.7)+
    scale_color_manual(values = clrs[n],labels = c('HCP_AGING', 'HCP_YA'))+
    theme(legend.position="none")+
    geom_density(aes(x=deltahat,color=Sites), data = NULL, linetype = "dotted",size =0.7)+
    theme(text = element_text(size=8),plot.title = element_text(hjust = 0.5))+
    ylab('Density')+xlab('Delta')+
    ggtitle(sites[n])
  
  
  myplots[[n]] <- p1  # add each plot into plot list
}

do.call(grid.arrange,  c(myplots,ncol=4))
dev.off()




# format output
outputthickness <-as.numeric(t(output))
outputthickness <- matrix(outputthickness, ncol = 71, byrow = FALSE)

residualized_regionwise_data=matrix(, nrow = 710, ncol = 71)

for (i in 1:71){
  residualized_regionwise_data[,i] <- residuals(lm(outputthickness[,i] ~ age + sex + tss +motor+lesionvol))
}
clrs <- wes_palette("Zissou1", 28, type = "continuous")

d2<-data.frame(residualized_regionwise_data, batchvar)
pcavar <- prcomp(d2[1:71] , center = TRUE,scale. = TRUE)
dev.on()
pdf(width=20, height=6.4,'/Users/emilyolafson/GIT/ENIGMA/enigma_disconnections/cortical_atrophy/figures/PCA_3PCs_age_sex_tss_motor_lesionvol_regressed_enigma_ComBat.pdf')

p1<- autoplot(pcavar, x=1, y=2, data = d2,size=3, colour = 'batchvar',alpha=0.5)+theme_classic()+scale_color_manual(values = c(col_vector))+theme(text = element_text(size=30)) + theme(legend.position="none",legend.title = element_blank())
p2<- autoplot(pcavar, x=2, y=3, data = d2,size=3,colour = 'batchvar',alpha=0.5)+theme_classic()+scale_color_manual(values = c(col_vector))+theme(text = element_text(size=30))+ theme(legend.position="none",legend.title = element_blank())
p3<- autoplot(pcavar, x=1, y=3, data = d2,size=3,colour = 'batchvar',alpha=0.5)+theme_classic()+scale_color_manual(values = c(col_vector))+theme(text = element_text(size=30))+ theme(legend.position="right",legend.direction = "vertical",legend.title = element_blank(),legend.text =  element_text(size=10,margin = margin(t = 5)))
plot_grid(p1, p2, p3, ncol=3, rel_widths = c(1,1,1.3),label_size = 30)

dev.off()



# format data
combatted_table <- data.frame(age, sex, dataset, batchvar, lesionvol)
combatted_table<-cbind(subject, outputthickness, combatted_table)

colnames(combatted_table) <- og_colnames[2:77]


write.table(combatted_table,'/Users/emilyolafson/GIT/ENIGMA/enigma_disconnections/cortical_atrophy/data/FREESURFER_ENIGMA/enigma_thickness_datasetvar_COMBAT.csv',sep = ",",row.names = FALSE)

