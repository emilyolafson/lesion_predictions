if (!requireNamespace("BiocManager", quietly = TRUE))
  install.packages("BiocManager")
library(devtools)
install_github("vqv/ggbiplot")
library(plotly)
library(ggfortify)
library(ggplot2)
library(sva)
require(MASS)
library(cowplot)
library(RColorBrewer)
library(wesanderson)
install.packages("invgamma")
library(invgamma)
thickness <- read.table('/Users/emilyolafson/GIT/ENIGMA/data/FREESURFER_ENIGMA/hcp_aging_ya_thickness_datasetvar.csv',sep = ",", header=TRUE)
og_colnames <- colnames(thickness)
# setup combat
Subject <- thickness$Subject
dataset <- thickness$dataset

age <- thickness$age
thickness$sex[thickness$sex =="3.0"]=NA
thickness$sex[thickness$sex ==""]=NA
sex <- as.factor(thickness$sex)

nasex <- as.logical(rowSums(is.na(cbind(age, sex))))

thickness <- thickness[!nasex,]
subject <- thickness$Subject
age <- thickness$age
sex <- as.factor(thickness$sex)
dxstatus <- as.factor(thickness$disease_status)
batchvar <- as.vector(thickness$dataset)
dataset <- as.vector(thickness$dataset)
thickness <- thickness[, 3:72]
thicknesst <-t(thickness)

residualized_regionwise_data=matrix(, nrow = 1722, ncol = 70)

for (i in 1:70){
  residualized_regionwise_data[,i] <- residuals(lm(unlist(thickness[i]) ~ age + sex))
}

d2<-data.frame(residualized_regionwise_data, batchvar)

# PCA
pcavar <- prcomp(d2[1:70] , center = TRUE,scale. = TRUE)
dev.on()
pdf(width=8, height=4,'/Users/emilyolafson/GIT/ENIGMA/enigma_disconnections/cortical_atrophy/figures/PCA_3PCs_age_sex_regressed_hcpaging_hcpya_raw.pdf')

p1<- autoplot(pcavar, x=1, y=2, data = d2, colour = 'batchvar',alpha=0.5)+theme_classic()+scale_color_brewer(palette = "Set1")+theme(text = element_text(size=15)) + theme(legend.position="top",legend.direction = "vertical",legend.title = element_blank())
p2<- autoplot(pcavar, x=2, y=3, data = d2,colour = 'batchvar',alpha=0.5)+theme_classic()+scale_color_brewer(palette = "Set1")+theme(text = element_text(size=15))+ theme(legend.position="top",legend.direction = "vertical",legend.title = element_blank())
p3<- autoplot(pcavar, x=1, y=3, data = d2,colour = 'batchvar',alpha=0.5)+theme_classic()+scale_color_brewer(palette = "Set1")+theme(text = element_text(size=15))+ theme(legend.position="top",legend.direction = "vertical",legend.title = element_blank())
plot_grid(p1, p2, p3, nrow=1, label_size = 12)
dev.off()

# ComBat

mod =cbind(sex,age)

outputn <- neuroCombat(dat=thicknesst,batch=batchvar,mod=mod)
output <- outputn$dat.combat

# visualize combat method.
pdf(width=10, height=10,'/Users/emilyolafson/GIT/ENIGMA/enigma_disconnections/cortical_atrophy/figures/HCP_aging_ya_empirical_estimated_params.pdf')

ests <- outputn$estimates
gammahat <- ests$gamma.star
x <- c(seq(-1.5, 2, length=70),seq(-1.5, 2, length=70))
gammahat <- c(as.double(gammahat[1,]),as.double(gammahat[2,]))
y <- c(dnorm(seq(-1.5, 2, length=70), mean=ests$gamma.bar[1], sd=sqrt(ests$t2[1])), dnorm(seq(-1.5,2, length=70), mean=ests$gamma.bar[2], sd=sqrt(ests$t2[2])))
Sites <- c(rep('HCP_AGING', 70), rep('HCP_YA',70))
df<-data.frame(x,y, gammahat,Sites)
clrs = c("#397eb8", "#e42021")
p1 <- ggplot(df, aes(x=x,color=Sites))+ theme_classic()+
  geom_line(aes(x,y,color=Sites), data = NULL, linetype = "solid",size =0.7)+
  scale_color_manual(values = clrs,labels = c('HCP_AGING', 'HCP_YA'))+
  theme(legend.position="none")+
  geom_density(aes(x=gammahat,color=Sites), data = NULL, linetype = "dotted",size =0.7)+
  theme(text = element_text(size=15))+

  ylab('Density')+xlab('Gamma')


ests <- outputn$estimates
deltahat <- ests$delta.hat
x <- c(seq(-1.5, 2, length=70),seq(-1.5, 2, length=70))
deltahat <- c(as.double(deltahat[1,]),as.double(deltahat[2,]))
y <- c(dinvgamma(x=seq(-1.5, 2, length=70), shape=ests$a.prior[1], rate=ests$b.prior[1]), dinvgamma(x=,seq(-1.5,2, length=70), shape=ests$a.prior[2], rate=ests$b.prior[2]))
Sites <- c(rep('HCP_AGING', 70), rep('HCP_YA',70))
df<-data.frame(x,y, deltahat,Sites)
clrs = c("#397eb8", "#e42021")
p2 <- ggplot(df, aes(x=x,color=Sites))+ theme_classic()+
  geom_line(aes(x,y,color=Sites), data = NULL, linetype = "solid",size =0.7)+
  scale_color_manual(values = clrs,labels = c('HCP_AGING', 'HCP_YA'))+
  theme(legend.position='bottom')+
  
  geom_density(aes(x=deltahat,color=Sites), data = NULL, linetype = "dotted",size =0.7)+
  theme(text = element_text(size=15))+
  
  ylab('Density')+xlab('Delta')

plot_grid(p1, p2, nrow=2, rel_widths = c(1,1),label_size = 20)

dev.off()


# format outputs
outputthickness <-as.numeric(t(output))
outputthickness <- matrix(outputthickness, ncol = 70, byrow = FALSE)

residualized_regionwise_data=matrix(, nrow = 1722, ncol = 70)

for (i in 1:70){
  residualized_regionwise_data[,i] <- residuals(lm(outputthickness[,i] ~ age + sex))
}

d2<-data.frame(residualized_regionwise_data, batchvar)
pcavar <- prcomp(d2[1:70] , center = TRUE,scale. = TRUE)
dev.on()
pdf(width=10, height=4,'/Users/emilyolafson/GIT/ENIGMA/enigma_disconnections/cortical_atrophy/figures/PCA_3PCs_age_sex_regressed_hcpaging_hcpya_ComBat.pdf')

p1<- autoplot(pcavar, x=1, y=2, data = d2, colour = 'batchvar',alpha=0.5)+theme_classic()+scale_color_brewer(palette = "Set1")+theme(text = element_text(size=15)) + theme(legend.position="top",legend.direction = "vertical",legend.title = element_blank())
p2<- autoplot(pcavar, x=2, y=3, data = d2,colour = 'batchvar',alpha=0.5)+theme_classic()+scale_color_brewer(palette = "Set1")+theme(text = element_text(size=15))+ theme(legend.position="top",legend.direction = "vertical",legend.title = element_blank())
p3<- autoplot(pcavar, x=1, y=3, data = d2,colour = 'batchvar',alpha=0.5)+theme_classic()+scale_color_brewer(palette = "Set1")+theme(text = element_text(size=15))+ theme(legend.position="top",legend.direction = "vertical",legend.title = element_blank())
plot_grid(p1, p2, p3, nrow=1,  label_size = 12)

dev.off()



# format data
combatted_table <- data.frame(age, sex, dataset, batchvar)
combatted_table<-cbind(subject, outputthickness, combatted_table)

colnames(combatted_table) <- og_colnames[2:76]

# COMBAT

write.table(combatted_table,'/Users/emilyolafson/GIT/ENIGMA/data/FREESURFER_ENIGMA/hcp_aging_ya_thickness_datasetvar_COMBAT.csv',sep = ",",row.names = FALSE)



