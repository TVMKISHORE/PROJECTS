#----------------------------------------------------------------------------------------------------------
#XGB documentation
#https://github.com/dmlc/xgboost/blob/2f3958a45597750197de605bfe784763976650c1/R-package/vignettes/xgboostPresentation.Rmd
#----------------------------------------------------------------------------------------------------------

#Loading training data
train <- read.csv("C:/Users/Admin/Downloads/Black Friday/train_oSwQCTC/train.csv",stringsAsFactors = T)
test <- read.csv("C:/Users/Admin/Downloads/Black Friday/test_HujdGe7/test.csv",stringsAsFactors = T)


train['Product_ID_num']=substr(train$Product_ID,start=2,stop=9)
train$Product_ID_num=as.numeric(train$Product_ID_num)
library(dplyr)
train<-train%>% arrange(Product_ID_num) 


#Just conside imp features from Boruta package
#[1] "Product_ID"         "Gender"             "Age"                "Occupation"        
#[5] "City_Category"      "Product_Category_1" "Product_Category_2" "Product_Category_3"

#train['Product_ID_num']=substr(train$Product_ID,start=4,stop=7)
train$Stay_In_Current_City_Years <- NULL
train$Marital_Status <-NULL
train$User_ID <- NULL
train$Product_ID <- NULL



#-----------------------------------------------------------------------------------------
#Impute missing data and transforming
#Imputation method: all missing fields as different category
#-----------------------------------------------------------------------------------------
train_random_samp$Product_Category_2[is.na(train_random_samp$Product_Category_2)] <- 99
train_random_samp$Product_Category_3[is.na(train_random_samp$Product_Category_3)] <- 99

train_random_samp$Occupation=as.factor(train_random_samp$Occupation)
train_random_samp$Product_Category_1=as.factor(train_random_samp$Product_Category_1)
train_random_samp$Product_Category_2=as.factor(train_random_samp$Product_Category_2)
train_random_samp$Product_Category_3=as.factor(train_random_samp$Product_Category_3)
train_random_samp$Product_ID_num=as.numeric(train_random_samp$Product_ID_num)

trainSet=train_random_samp




#---------------------------------------------------------------
# Building Model: GBM
#---------------------------------------------------------------
XGB <- xgboost(data = trainSet[,predictors], 
               label = trainSet[,outcomeName], 
               eta = 0.003,                      # Learning rate
               max_depth = 4,    
               min_child_weight=4,  
               colsample_bytree=.5,  #ratio of columns when constructing tree Default is :1
               subsample = 0.60,
               set.seed=2126,
               objective = "reg:linear",  #Linear Regression
               nthread = 4,
               nround=300 
               
)
