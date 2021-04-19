if (!requireNamespace("haploR", quietly = TRUE))
  install.packages("haploR", dependencies = TRUE,repos = "http://cran.us.r-project.org")
library(haploR)

x <- queryHaploreg(file="LD/leadSNPs.txt", ldThresh = 0.6, verbose = TRUE)
onlyIDs <- x[, c("chr","query_snp_rsid","rsID","ref","alt")]
positions <- vector()
for (item in onlyIDs$rsID){
  x <- queryRegulome(c(item))
  positions <- append(positions, x$guery_coordinates)
}
print(positions)
onlyIDs$positions <- positions
# Bed file
write.table(onlyIDs,file='LD/LDsnps.bed',sep='\t',quote=F,col.names=F,row.names=F)

# #Excell file
# require(openxlsx)
# write.xlsx(x=onlyIDs, file="LD/LDsnps.xlsx")
