library(plotly)
library(ggfortify)
library(ggplot2)
library(sva)
library(RColorBrewer)
library(wesanderson)

thickness <- read.table('/Users/emilyolafson/GIT/ENIGMA/enigma_disconnections/cortical_atrophy/data/FREESURFER_ENIGMA/enigma_thickness_datasetvar.csv',sep = ",", header=TRUE)
og_colnames <- colnames(thickness)
# setup combat
Subject <- thickness$Subject
dataset <- thickness$dataset

age <- thickness$age
thickness$sex[thickness$sex =="3.0"]=NA
thickness$sex[thickness$sex ==""]=NA
sex <- as.factor(thickness$sex)
tss <- thickness$time_since_stroke
motor <- thickness$motor_score

nas <- as.logical(rowSums(is.na(cbind(age, sex, tss, motor))))
thickness <- thickness[!nas,]


subject <- thickness$Subject
age <- thickness$age
tss <- thickness$time_since_stroke
sex <- as.factor(thickness$sex)
dxstatus <- as.factor(thickness$disease_status)
motor <- thickness$motor_score

site <- as.vector(thickness$site)
dataset <- as.vector(thickness$dataset)
thickness <- thickness[, 3:72]
thicknesst <-t(thickness)

finaltable <- data.frame(cbind(subject, thickness,site,dataset,sex,age,motor))
residualized_regionwise_data=matrix(, nrow = 710, ncol = 70)

for (i in 1:70){
  residualized_regionwise_data[,i] <- residuals(lm(unlist(thickness[i]) ~ age + sex + tss +motor))
}


## boxplot age and ct
theme = theme_set(theme_minimal())
theme = theme_update(legend.position="top", legend.title=element_blank(), panel.grid.major.x=element_blank())

x<-NULL
x2<-NULL
cortical_thickness<-NULL
site<-NULL
counter=0
for(i in 1:710){
  x=c(x, seq(counter+1, counter+70, 1))
  x2=c(x2, rep(as.character(i),70))
  cortical_thickness=c(cortical_thickness, as.double(residualized_regionwise_data[i,]))
  site = c(site, rep(finaltable[i,72],70))
  counter=counter+70
}

y <-cortical_thickness
datatable <- data.frame(x,x2,y, site)
names(wes_palettes)

pdf(width=13, height=6.4,'/Users/emilyolafson/GIT/ENIGMA/enigma_disconnections/cortical_atrophy/figures/enigma_CT.pdf')

ggplot(data = datatable,aes(x = x, y = cortical_thickness,color=site)) +theme_classic()+scale_color_manual("Sites", values= wes_palette("Zissou1", 28, type = "continuous"))+geom_boxplot(outlier.colour = NULL,outlier.alpha = 0.2,outlier.size=.1) +theme(axis.ticks.x = element_blank(), axis.text.x = element_blank(),text = element_text(size=20)) + xlab("Subjects grouped by site") + ylab("Cortical thickness")
dev.off()









