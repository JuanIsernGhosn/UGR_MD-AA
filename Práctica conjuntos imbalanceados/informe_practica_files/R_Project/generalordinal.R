dataset <- RWeka::read.arff("../.")
    
ordinal_classification <- function(data, className){
    classes <- unique(data[,className])
    
    
}