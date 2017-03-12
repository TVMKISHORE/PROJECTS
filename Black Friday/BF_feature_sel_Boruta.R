# This module is to select the important features using BORUTRA package

#Since we have 18k observations in of data , take randon samples and run Boruta to figure out 
#feature impartance. I took 3 random samples and if they appeared twice as important then 
#I selected as important

#You may refer the below links for documentstion.
#-----------------------------------------------------------------------
#https://www.analyticsvidhya.com/blog/2016/12/introduction-to-feature-selection-methods-with-an-example-or-how-to-select-the-right-variables/
#https://www.analyticsvidhya.com/blog/2016/03/select-important-variables-boruta-package/
#-----------------------------------------------------------------------
#------------------------------------------------------------------------------------------
#Below is the preprocessing of test data
#------------------------------------------------------------------------------------------
wrapdata <- read.csv("C:/Users/Admin/Downloads/Black Friday/train_oSwQCTC/train.csv")
wrapdata_nona=wrapdata[complete.cases(wrapdata),]   #Remove NA data


#----------------------------------------
# Random sample above data set
#----------------------------------------
library(caret)
index <- createDataPartition(wrapdata_nona$Purchase, p=0.025, list=FALSE)
train_random_samp <- wrapdata_nona[ index,]
#testSet <- train_master[-index,]


install.packages("Boruta")
library(Boruta)

set.seed(123)
boruta.train <- Boruta(Purchase~.-User_ID, data = train_random_samp, doTrace = 2)
print(boruta.train)
save(boruta.train,file="boruta.train.Rdata")
save(boruta.train,file="boruta.train.Rdata1")
save(boruta.train,file="boruta.train.Rdata2")
load("boruta.train.Rdata2")
#Confirmed 1 attributes: City_Category.
#Confirmed 1 attributes: Product_ID.

#--------------------------------------------------------------Output--1 run
#Boruta performed 99 iterations in 1.251226 hours.
#7 attributes confirmed important: Age, City_Category, Gender, Product_Category_1,
#Product_Category_2 and 2 more.
#1 attributes confirmed unimportant: Marital_Status.
#2 tentative attributes left: Occupation, Stay_In_Current_City_Years.
#--------------------------------------------------------------Output--2 run
#Boruta performed 99 iterations in 28.31469 mins.
#6 attributes confirmed important: City_Category, Occupation, Product_Category_1,
#Product_Category_2, Product_Category_3 and 1 more.
#3 attributes confirmed unimportant: Age, Marital_Status, Stay_In_Current_City_Years.
#1 tentative attributes left: Gender.
#--------------------------------------------------------------




#--------------------------------------------------------------
# Plotting Importance 
#--------------------------------------------------------------
plot(boruta.train, xlab = "", xaxt = "n")
lz<-lapply(1:ncol(boruta.train$ImpHistory),function(i)
  boruta.train$ImpHistory[is.finite(boruta.train$ImpHistory[,i]),i])
names(lz) <- colnames(boruta.train$ImpHistory)
Labels <- sort(sapply(lz,median))
axis(side = 1,las=2,labels = names(Labels),
     at = 1:ncol(boruta.train$ImpHistory), cex.axis = 0.7)



#--------------------------------------------------------------
#The tentative attributes will be classified as confirmed or rejected by comparing the 
#median Z score of the attributes with the median Z score of the best shadow attribute
#--------------------------------------------------------------
final.boruta <- TentativeRoughFix(boruta.train)
print(final.boruta)



#--------------------------------------------------------------
# Let's obtain the list of confirmed attributes
#--------------------------------------------------------------
getSelectedAttributes(final.boruta, withTentative = F)
#--------------------------------------------------------------Output-1 run
#[1] "Product_ID"                 "Gender"                     "Age"                       
#[4] "City_Category"              "Stay_In_Current_City_Years" "Product_Category_1"        
#[7] "Product_Category_2"         "Product_Category_3" 
#--------------------------------------------------------------Output-2 run
#[1] "Product_ID"         "Occupation"         "City_Category"      "Product_Category_1"
#[5] "Product_Category_2" "Product_Category_3"
#--------------------------------------------------------------Output-3 run
#[1] "Product_ID"         "Gender"             "Age"                "Occupation"        
#[5] "City_Category"      "Product_Category_1" "Product_Category_2" "Product_Category_3"
#--------------------------------------------------------------Output-3 run

#--------------------------------------------------------------
#We'll create a data frame of the final result derived from Boruta.
#--------------------------------------------------------------
boruta.df <- attStats(final.boruta)
class(boruta.df)
#[1] "data.frame"
print(boruta.df)
#The below table will be displayed
#--------------------------------- 1 run
#meanImp  medianImp     minImp     maxImp  normHits  decision
#Product_ID                  52.190561  52.248994 46.3756717  57.725726 1.0000000 Confirmed
#Gender                       5.149241   5.129824  2.5381116   9.032258 1.0000000 Confirmed
#Age                          5.536240   5.571999  2.9788691   8.352965 1.0000000 Confirmed-Rejected
#Occupation                   1.386393   1.449824 -0.9489723   4.362514 0.3636364  Rejected
#City_Category                7.596889   7.592234  4.3390562  11.465242 1.0000000 Confirmed
#Stay_In_Current_City_Years   1.599082   1.563789 -1.6819840   4.413184 0.4343434 Confirmed-Rejected
#Marital_Status               1.196674   1.340376 -1.4340542   2.909970 0.1212121  Rejected-Rejected
#Product_Category_1         108.569056 108.879714 95.6149007 122.057269 1.0000000 Confirmed
#Product_Category_2          51.506660  51.653254 48.1897171  54.969918 1.0000000 Confirmed
#Product_Category_3          41.789487  41.819250 38.2498179  45.079508 1.0000000 Confirmed
#---------------------------------
#--------------------------------- 2 run
#meanImp  medianImp     minImp      maxImp   normHits  decision
#Product_ID                 47.7596881 47.7876113 41.6239320  52.2761244 1.00000000 Confirmed
#Gender                      2.3781985  2.4365136 -0.3111092   5.2204310 0.58585859  Rejected
#Age                         0.7376624  0.8031276 -1.7141199   3.6317044 0.09090909  Rejected
#Occupation                  3.0650044  3.0702015  0.1244097   6.0047473 0.77777778 Confirmed
#City_Category               4.9368554  4.9940516  1.6393802   8.6559766 0.96969697 Confirmed
#Stay_In_Current_City_Years -0.6791123 -0.7723847 -2.1828753   0.8402396 0.00000000  Rejected
#Marital_Status              0.3240183  0.1524639 -1.1773032   2.7197036 0.04040404  Rejected
#Product_Category_1         96.0642240 95.3636576 81.3448079 111.9129477 1.00000000 Confirmed
#Product_Category_2         46.8080480 47.1206487 42.6952946  50.4611152 1.00000000 Confirmed
#Product_Category_3         37.2669038 37.4686441 32.5953173  40.3712184 1.00000000 Confirmed


#confirmed IVs after Boruta run

#[1] "Product_ID"         "Gender"             "Age"                "Occupation"        
#[5] "City_Category"      "Product_Category_1" "Product_Category_2" "Product_Category_3"

