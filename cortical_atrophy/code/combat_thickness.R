if (!requireNamespace("BiocManager", quietly = TRUE))
  install.packages("BiocManager")
library(devtools)
install_github("vqv/ggbiplot")
library(plotly)
library(ggfortify)
library(ggplot2)
library(sva)

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

finaltable <- cbind(thickness,batchvar,dataset,sex,age)
# pca of the variables?
pcavar <- prcomp(finaltable[3:72], center = TRUE,scale. = TRUE)

autoplot(pcavar, data = finaltable, colour = 'dataset')+theme_classic()+scale_color_brewer(palette = "Set1")

mod =cbind(sex,age)

# COMBAT

output <- ComBat(
  dat=thicknesst,
  batch=batchvar,
  mod = mod,
  parametric=FALSE,
  par.prior = TRUE,
  prior.plots = TRUE,
  mean.only = FALSE,
  ref.batch = NULL,
  BPPARAM = bpparam("SerialParam")
)

outputt <-as.numeric(t(output))
d <- matrix(outputt, ncol = 70, byrow = FALSE)

combatted_table <- data.frame(age, sex, dataset, batchvar)
combatted_table<-cbind(subject, d, combatted_table)

colnames(combatted_table) <- og_colnames[2:76]
pcavar <- prcomp(combatted_table[2:71], center = TRUE,scale. = TRUE)
autoplot(pcavar, data = finaltable, colour = 'dataset')+theme_classic()+scale_color_brewer(palette = "Set1")

tmp <- pcavar$x[,1]
cor(tmp, combatted_table$age)

# COMBAT

write.table(combatted_table,'/Users/emilyolafson/GIT/ENIGMA/data/FREESURFER_ENIGMA/hcp_aging_ya_enigma_thickness_datasetvar_COMBAT.csv',sep = ",",row.names = FALSE)

