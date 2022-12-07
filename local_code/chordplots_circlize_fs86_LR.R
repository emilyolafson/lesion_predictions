install.packages("circlize")
library(circlize)

# left and right split

# Restart circular layout parameters
circos.clear()

circos.initialize()
par(cex =0.9)

m<-data.matrix(read.table('/Users/emilyolafson/GIT/ENIGMA/enigma_disconnections/figures/fs86_1_chacoconn_matrix_LR.csv', sep=","))
rownames(m)<- c("R Visual", "R Somatomotor", "R Dorsal \nAttention", "R Ventral \n Attention", "R Limbic",
                 "R Frontoparietal", "R Default Mode", "R Subcortical \n Structures","R Cerebellum",
                 "L Cerebellum","L Subcortical \n Structures", "L Default \n Mode", "L Frontoparietal",
                 "L Limbic","L Ventral \n Attention", "L Dorsal \n Attention",  "L Somatomotor",   "L Visual")

colnames(m) <-  c( "R Visual", "R Somatomotor", "R Dorsal \nAttention", "R Ventral \n Attention", "R Limbic",
                   "R Frontoparietal", "R Default Mode", "R Subcortical \n Structures",  "R Cerebellum",
                   "L Cerebellum","L Subcortical \n Structures", "L Default \n Mode", "L Frontoparietal",
                   "L Limbic","L Ventral \n Attention", "L Dorsal \n Attention",  "L Somatomotor",   "L Visual")

rangecolors <- hcl.colors(9, "Reds")
rangecolors[9] <- "#FFFFFF"

cols <- colorRamp2(seq(from = -0.25, to = -1, length.out = 9), rangecolors)

circos.par(start.degree = 90)

colors_rows <- c(rev(c("#4D39B5", "#4FBB74", "#AAE8DE", "#79C7EF", "#E2C845", "#DE526A", "#D656C1",  "#F3E392", "#6A5C84")), c("#4D39B5", "#4FBB74", "#AAE8DE", "#79C7EF", "#E2C845", "#DE526A", "#D656C1",  "#F3E392", "#6A5C84"))
#pdf('/Users/emilyolafson/GIT/ENIGMA/enigma_disconnections/figures/chorddiagram_stdCV_LRsplit_reds.pdf', width=10, height=10)

chordDiagram(m, col=cols,transparency = 0,  reduce = 0, grid.col = colors_rows, link.zindex =m,
             annotationTrack =  c("name", "grid"),link.lwd = 0.2,link.lty = 1, link.sort = TRUE, link.decreasing = TRUE,
             scale = TRUE, self.link = 1,order =c( "R Visual", "R Somatomotor", "R Dorsal \nAttention", "R Ventral \n Attention", "R Limbic",
                                                            "R Frontoparietal", "R Default Mode", "R Subcortical \n Structures", "R Cerebellum",
                                                            "L Cerebellum","L Subcortical \n Structures", "L Default \n Mode", "L Frontoparietal",
                                                            "L Limbic","L Ventral \n Attention", "L Dorsal \n Attention",  "L Somatomotor",   "L Visual"))



#dev.off()






