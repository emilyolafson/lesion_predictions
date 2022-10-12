if (!requireNamespace("BiocManager", quietly = TRUE))
  install.packages("BiocManager")
library(devtools)
install_github("vqv/ggbiplot")
library(plotly)
library(ggfortify)
library(ggplot2)
library(sva)
library(RColorBrewer)
n <- 30
qual_col_pals = brewer.pal.info[brewer.pal.info$category == 'qual',]
col_vector = unlist(mapply(brewer.pal, qual_col_pals$maxcolors, rownames(qual_col_pals)))


thickness <- read.table('/Users/emilyolafson/GIT/ENIGMA/data/FREESURFER_ENIGMA/enigma_thickness_datasetvar.csv',sep = ",", header=TRUE)
og_colnames <- colnames(thickness)
# setup combat
Subject <- thickness$Subject
dataset <- thickness$dataset

age <- thickness$age
thickness$sex[thickness$sex =="3.0"]=NA
thickness$sex[thickness$sex ==""]=NA
sex <- as.factor(thickness$sex)
tss <- thickness$time_since_stroke

nas <- as.logical(rowSums(is.na(cbind(age, sex, tss))))
thickness <- thickness[!nas,]


subject <- thickness$Subject
age <- thickness$age
tss <- thickness$time_since_stroke
sex <- as.factor(thickness$sex)
dxstatus <- as.factor(thickness$disease_status)
motor <- thickness$motor_score

batchvar <- as.vector(thickness$site)
dataset <- as.vector(thickness$dataset)
thickness <- thickness[, 3:72]
thicknesst <-t(thickness)

finaltable <- cbind(thickness,batchvar,dataset,sex,age,motor)

residualized_regionwise_data=matrix(, nrow = 724, ncol = 70)

for (i in 1:70){
  residualized_regionwise_data[,i] <- residuals(lm(unlist(thickness[i]) ~ age + sex + tss ))
}

d2<-data.frame(residualized_regionwise_data, batchvar)

# PCA
pcavar <- prcomp(d2[1:70] , center = TRUE,scale. = TRUE)
dev.on()
pdf(width=20, height=6.4,'/Users/emilyolafson/GIT/ENIGMA/enigma_disconnections/cortical_atrophy/PCA_3PCs_age_sex_tss_regressed_enigma_raw.pdf')

p1<- autoplot(pcavar, x=1, y=2, data = d2,size=3, colour = 'batchvar',alpha=0.5)+theme_classic()+scale_color_manual(values = c(col_vector))+theme(text = element_text(size=30)) + theme(legend.position="none", legend.title = element_blank())
p2<- autoplot(pcavar, x=2, y=3, data = d2,size=3,colour = 'batchvar',alpha=0.5)+theme_classic()+scale_color_manual(values = c(col_vector))+theme(text = element_text(size=30))+ theme(legend.position="none",legend.title = element_blank())
p3<- autoplot(pcavar, x=1, y=3, data = d2,size=3,colour = 'batchvar',alpha=0.5)+theme_classic()+scale_color_manual(values = c(col_vector))+theme(text = element_text(size=30))+ theme(legend.position="right",legend.direction = "vertical",legend.title = element_blank(),legend.text =  element_text(size=10,margin = margin(t = 5)))
plot_grid(p1, p2, p3, ncol=3, rel_widths = c(1,1,1.3),label_size = 30)

dev.off()

mod =cbind(sex,age,tss)
# COMBAT

output <- ComBat(
  dat=thicknesst,
  batch=batchvar,
  mod = mod,
  par.prior = TRUE,
  prior.plots = TRUE,
  mean.only = FALSE,
  ref.batch = NULL,
  BPPARAM = bpparam("SerialParam")
)

outputthickness <-as.numeric(t(output))
outputthickness <- matrix(outputthickness, ncol = 70, byrow = FALSE)

residualized_regionwise_data=matrix(, nrow = 724, ncol = 70)

for (i in 1:70){
  residualized_regionwise_data[,i] <- residuals(lm(outputthickness[,i] ~ age + sex + tss))
}

d2<-data.frame(residualized_regionwise_data, batchvar)
pcavar <- prcomp(d2[1:70] , center = TRUE,scale. = TRUE)
dev.on()
pdf(width=20, height=6.4,'/Users/emilyolafson/GIT/ENIGMA/enigma_disconnections/cortical_atrophy/PCA_3PCs_age_sex_tss_regressed_enigma_ComBat.pdf')

p1<- autoplot(pcavar, x=1, y=2, data = d2,size=3, colour = 'batchvar',alpha=0.5)+theme_classic()+scale_color_manual(values = c(col_vector))+theme(text = element_text(size=30)) + theme(legend.position="none",legend.title = element_blank())
p2<- autoplot(pcavar, x=2, y=3, data = d2,size=3,colour = 'batchvar',alpha=0.5)+theme_classic()+scale_color_manual(values = c(col_vector))+theme(text = element_text(size=30))+ theme(legend.position="none",legend.title = element_blank())
p3<- autoplot(pcavar, x=1, y=3, data = d2,size=3,colour = 'batchvar',alpha=0.5)+theme_classic()+scale_color_manual(values = c(col_vector))+theme(text = element_text(size=30))+ theme(legend.position="right",legend.direction = "vertical",legend.title = element_blank(),legend.text =  element_text(size=10,margin = margin(t = 5)))
plot_grid(p1, p2, p3, ncol=3, rel_widths = c(1,1,1.3),label_size = 30)

dev.off()



# format data
combatted_table <- data.frame(age, sex, dataset, batchvar)
combatted_table<-cbind(subject, outputthickness, combatted_table)

colnames(combatted_table) <- og_colnames[2:76]


write.table(combatted_table,'/Users/emilyolafson/GIT/ENIGMA/data/FREESURFER_ENIGMA/enigma_thickness_datasetvar_COMBAT.csv',sep = ",",row.names = FALSE)

