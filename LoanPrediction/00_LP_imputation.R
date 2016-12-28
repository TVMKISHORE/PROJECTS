#---------------------------------------------------------------------------------------------
#5 Powerful R Packages used for imputing missing values
#---------------------------------------------------------------------------------------------
#Resource
#https://www.analyticsvidhya.com/blog/2016/03/tutorial-powerful-packages-imputing-missing-values/
#---------------------------
#List of R Packages
#MICE
#Amelia
#missForest
#Hmisc
#mi
#---------------------------------------------------------------------------------------------

#install.packages('dplyr')   To perform set operation on Data Frames
install.packages("missForest")
library(missForest)

#install package and load library
install.packages("Hmisc")
library(Hmisc)

install.packages('xgboost')
library(xgboost)

#-------------------------------------
#loading data and subsetting the Na and noNa
#-------------------------------------
#Load Train Data
#To replace all the spaces and nulls as NA while loading
train <- read.csv("C:/Users/Admin/Downloads/Loan Prediction III/train_u6lujuX_CVtuZ9i.csv", na.strings=c("","NA"))
train_nona=train[complete.cases(train),]
write.csv(train_nona,file='trainf.csv')
train_na=train[!complete.cases(train),]

#Load test data
#To replace all the spaces and nulls as NA while loading
test <- read.csv("C:/Users/Admin/Downloads/Loan Prediction III/test_Y3wMUE5_7gLdaTN.csv", na.strings=c("","NA"))
test_nona=test[complete.cases(test),]
test_na=test[!complete.cases(test),]


#---------------------------
# combining tets and train set for imputation 
#---------------------------
#combine test and train data
test_train=rbind(train_nona[,1:12],test)
rownames(test_train) <- 1:nrow(test_train)  # To rebuild Index 



#--------------------------------------
# adding dummy variables Exclude "Loan_ID", "Gender", "Education" 
#-----------------------------------
#cols=c("Married","Dependents","Self_Employed","ApplicantIncome","CoapplicantIncome",
#       "LoanAmount","Loan_Amount_Term","Credit_History","Property_Area" )

cols=c("Dependents","Self_Employed","Property_Area" )


temp_d=model.matrix(~ Dependents - 1, data=test_train)
temp=merge(test_train,temp_d,by=0,all='TRUE')
temp$Row.names<-NULL
temp$Dependents<-NULL
test_train=temp


rownames(test_train) <- 1:nrow(test_train)
temp_d=model.matrix(~ Self_Employed - 1, data=test_train)
temp=merge(test_train,temp_d,by= 0,all='TRUE')
temp$Row.names<-NULL
temp$Self_Employed<-NULL
test_train=temp


rownames(test_train) <- 1:nrow(test_train)
temp_d=model.matrix(~ Property_Area - 1, data=test_train)
temp=merge(test_train,temp_d,by= 0,all='TRUE')
temp$Row.names<-NULL
temp$Property_Area<-NULL
test_train=temp

#----------------------------
#Total necessary unavailable columns
#----------------------------
#LoanAmount    Loan_Amount_Term Credit_History   Self_EmployedNo Self_Employe 
#NA's   :5       NA's   :6        NA's   :29       NA's   :23      NA's   :23      

#Dependents0      Dependents1   Dependents2     Dependents3+     
#NA's   :10       NA's   :10    NA's   :10      NA's   :10       
#----------------------------
#------------------------------------------------------------------
# perform imputation and create mutiple data sets 
# Replace NA values with mode/mean of these datasets
#----------------------------------------------


var.names=c('Dependents0','Dependents1','Dependents2','Dependents3_above',
            'Self_EmployedYes','Self_EmployedNo','ApplicantIncome',
            'CoapplicantIncome','LoanAmount','Loan_Amount_Term','Credit_History',
            'Property_AreaRural','Property_AreaUrban','Property_AreaSemiurban')

rownames(test_train) <- 1:nrow(test_train)
var.names=c('Dependents1','CoapplicantIncome','CoapplicantIncome','LoanAmount')
(fmla <- as.formula(paste(" ~ ", paste(var.names, collapse=" +"))))
impute_arg <- aregImpute(formula = fmla, n.impute=4, data=test_train)
impute_data <-  impute(test_train,11,impute_arg$imputed$Dependents1,Catogorical='True')

rownames(impute_data) <- 1:nrow(impute_data)
var.names=c('Dependents2','CoapplicantIncome','CoapplicantIncome','LoanAmount')
(fmla <- as.formula(paste(" ~ ", paste(var.names, collapse=" +"))))
impute_arg <- aregImpute(formula = fmla, n.impute=4, data=impute_data)
impute_data <-  impute(impute_data,12,impute_arg$imputed$Dependents2,Catogorical='True')

rownames(test_train) <- 1:nrow(test_train)
var.names=c('Loan_Amount_Term','Property_AreaRural')
(fmla <- as.formula(paste(" ~ ", paste(var.names, collapse=" +"))))
impute_arg <- aregImpute(formula = fmla, n.impute=4, data=impute_data)
impute_data <-  impute(impute_data,8,impute_arg$imputed$Loan_Amount_Term,Catogorical='False')

rownames(test_train) <- 1:nrow(test_train)
var.names=c('Credit_History','ApplicantIncome','CoapplicantIncome','LoanAmount')
(fmla <- as.formula(paste(" ~ ", paste(var.names, collapse=" +"))))
impute_arg <- aregImpute(formula = fmla, n.impute=4, data=impute_data)
impute_data <-  impute(impute_data,9,impute_arg$imputed$Credit_History,Catogorical='True')


rownames(test_train) <- 1:nrow(test_train)
var.names=c('LoanAmount','CoapplicantIncome','CoapplicantIncome','LoanAmount')
(fmla <- as.formula(paste(" ~ ", paste(var.names, collapse=" +"))))
impute_arg <- aregImpute(formula = fmla, n.impute=4, data=impute_data)
impute_data <-  impute(impute_data,7,impute_arg$imputed$LoanAmount,Catogorical='False')

#-----------------------
#now split data back to  test_data and train data and save for python processing
#-----------------------
rownames(test_train) <- 1:nrow(test_train)
what_tolook='Loan_ID'
what_toget=c("Property_AreaRural","Property_AreaSemiurban","Dependents1","Dependents2","Loan_Amount_Term.x",
             "Credit_History.x","CoapplicantIncome.x","LoanAmount.x","ApplicantIncome.x","Loan_Status")
train_final=lookup_table(train,impute_data,what_tolook,what_toget)

what_toget=c("Loan_ID","Property_AreaRural","Property_AreaSemiurban","Dependents1","Dependents2","Loan_Amount_Term.y",
             "Credit_History.y","CoapplicantIncome.y","LoanAmount.y","ApplicantIncome.y")
test_final=lookup_table(test,impute_data,what_tolook,what_toget)

write.csv(train_final,file='trainf.csv')

write.csv(test_final,file='testf.csv')





#----------------------------------
#exploration
#----------------------------------
# Missing & most important features exploration
#----------------------------------
#Loan_Amount_Term
tapply(train_nona$Loan_Amount_Term,train_nona$Gender,mean)
tapply(train_nona$Loan_Amount_Term,train_nona$Dependents,mean)
tapply(train_nona$Loan_Amount_Term,train_nona$Married,mean)
tapply(train_nona$Loan_Amount_Term,train_nona$Education,mean)
tapply(train_nona$Loan_Amount_Term,train_nona$Self_Employed,mean)
Credit_History
LoanAmount 
tapply(train_nona$LoanAmount,train_nona$Gender,mean)
tapply(train_nona$LoanAmount,train_nona$Dependents,mean)
tapply(train_nona$LoanAmount,train_nona$Married,mean)
tapply(train_nona$LoanAmount,train_nona$Education,mean)
tapply(train_nona$LoanAmount,train_nona$Self_Employed,mean)

#CoapplicantIncome 
tapply(train_nona$CoapplicantIncome,train_nona$Gender,mean)
tapply(train_nona$CoapplicantIncome,train_nona$Dependents,mean)
tapply(train_nona$CoapplicantIncome,train_nona$Married,mean)
tapply(train_nona$CoapplicantIncome,train_nona$Education,mean)
tapply(train_nona$CoapplicantIncome,train_nona$Self_Employed,mean)

trn_aggdata1 <-aggregate(train_nona, by=list(train_nona$Married,train_nona$Gender,train_nona$Dependents,train_nona$Property_Area),FUN=mean)
trn_aggdata2 <-aggregate(train_nona, by=list(train_nona$Married,train_nona$Gender,train_nona$Dependents,train_nona$Property_Area),FUN=length)
tst_aggdata1 <-aggregate(test_nona, by=list(test_nona$Married,test_nona$Gender,test_nona$Dependents,test_nona$Property_Area),FUN=mean)
tst_aggdata2 <-aggregate(test_nona, by=list(train_nona$Married,train_nona$Gender,train_nona$Dependents,train_nona$Property_Area),FUN=length)
#temp=subset(train_na, Married=='No' & Gender=='Male'& Dependents=='1' & Property_Area=='Urban')
#temp=subset(train_na, Married=='No' & Gender=='Male'& Dependents=='2' & Property_Area=='Rural')
#temp=subset(train_na, Married=='No' & Gender=='Male'& Dependents=='2' & Property_Area=='Semiurban')

#Use the below to impute the test set
#aggdata1 <- aggregate(train_nona, by=list(train_nona$Gender,train_nona$Education),FUN=mean)  
#aggdata2 <- aggregate(train_nona, by=list(train_nona$Gender,train_nona$Education),FUN=length)  





#-------------------------------------
#Multiple imputation using MICE package
#--------------------------------------
#Have imputed 5 different data sets for the missing values
#Testes accuracy score will all the data sets
#Only customer credit score is imputed with mode imputation
#-----------------------------------------
#load mice package
library(mice)
#load test and train data
train <- read.csv("C:/Users/Admin/Downloads/Loan Prediction III/train_u6lujuX_CVtuZ9i.csv", na.strings=c("","NA"))
test <- read.csv("C:/Users/Admin/Downloads/Loan Prediction III/test_Y3wMUE5_7gLdaTN.csv", na.strings=c("","NA"))
#combine both
cols=c("ApplicantIncome","CoapplicantIncome","LoanAmount","Loan_Amount_Term","Credit_History")
temp=rbind(train[cols],test[cols],axis=1)
total_data=temp[cols]

#impute the data
tempData <- mice(total_data,m=5,maxit=50,meth='pmm',seed=500) #meth='pmm'
#replace imputed values in the data set
dataset1 <- complete(tempData,1)
dataset2 <- complete(tempData,2)
dataset3 <- complete(tempData,3)
dataset4 <- complete(tempData,4)



#Seperate train and test sets
#lookup_table <- function(df1,df2,what_tolook,what_toget)
what_tolook=0   #Index

#Build columns to join
what_togettst=c("Loan_ID","ApplicantIncome.y","CoapplicantIncome.y","LoanAmount.y","Loan_Amount_Term.y","Credit_History.x")
what_togettrn=c("Loan_ID","ApplicantIncome.y","CoapplicantIncome.y","LoanAmount.y","Loan_Amount_Term.y","Credit_History.x","Loan_Status")

#Rename Columns after join
what_togettrn_rename=c("Loan_ID","ApplicantIncome","CoapplicantIncome","LoanAmount","Loan_Amount_Term","Credit_History","Loan_Status")
what_togettst_rename=c("Loan_ID","ApplicantIncome","CoapplicantIncome","LoanAmount","Loan_Amount_Term","Credit_History")

#Lookup test and train data sets and get the imputed values 
trainf1=lookup_table(train,dataset3,what_tolook,what_togettrn)
testf1=lookup_table(test,dataset3,what_tolook,what_togettst)


trainf1=lookup_table(train,dataset2,what_tolook,what_togettrn)
testf1=lookup_table(test,dataset2,what_tolook,what_togettst)

trainf1=lookup_table(train,dataset3,what_tolook,what_togettrn)
testf1=lookup_table(test,dataset3,what_tolook,what_togettst)


trainf1=lookup_table(train_nona,dataset4,what_tolook,what_togettrn)
testf1=lookup_table(test,dataset4,what_tolook,what_togettst)

#rename columns to a generic name
names(trainf1)<-what_togettrn_rename
names(testf1)<-what_togettst_rename

write.csv(trainf1,file='trainf1_cat3.csv')
write.csv(testf1,file='testf1_cat3.csv')

#save to csv file and pass to python

#A different way of imputation is to consider imputed catogorical variables as a different catogory 
#lets replace NA credit history as catogory type 3
testf1$Credit_History[is.na(testf1$Credit_History)]<-3
trainf1$Credit_History[is.na(trainf1$Credit_History)]<-3

#-------------------------------------------------
# Train sample engineering follows
#-------------------------------------------------
# Train set from imputation -- trainf1
install.packages('dplyr')
library(dplyr)
bad_data=subset(train_nona, Loan_Status=="N" & Credit_History==1 & CoapplicantIncome+ApplicantIncome > 4500 )
test_Gooddat=setdiff(train_nona,bad_data)
write.csv(test_Gooddat,file='trainf2.csv')
test_Gooddat=setdiff(train_nona,bad_data)
write.csv(test_Gooddat,file='trainf2.csv')


bad_data1=subset(trainf1, actual==0 & predected==1)
bad_data2=subset(trainf1, actual==1 & predected==0)
test_Gooddat=setdiff(trainf1,bad_data1)
test_Gooddat=setdiff(test_Gooddat,bad_data2)
write.csv(test_Gooddat,file='trainf_Badfree.csv')







#----------------------------------------------------------------------------------------
#To add a feature  EMIs is: EMI = [P x R x (1+R)^N]/[(1+R)^N  where R=9 
#----------------------------------------------------------------------------------------
#This is the formula to caluculate:
#E = P×r×(1 + r)n/((1 + r)n - 1)
#E is EMI
#where P is Priniple Loan Amount
#r is rate of interest calualted in monthly basis it should be = Rate of Annual interest/12/100
#if its 10% annual ,then its 10/12/100=0.00833
#n is tenture in number of months
#Eg: For 100000 at 10% annual interest for a period of 12 months
#it comes  to 100000*0.00833*(1 + 0.00833)12/((1 + 0.00833)12 - 1) = 8792
#------------------------------------------------------------------------------------------

#To replace all the spaces and nulls as NA while loading
train <- read.csv("C:/Users/Admin/Downloads/Loan Prediction III/train_u6lujuX_CVtuZ9i.csv", na.strings=c("","NA"))
train_nona=train[complete.cases(train),]
write.csv(train_nona,file='trainf.csv')
train_na=train[!complete.cases(train),]

train_nona['EMI']=(train_nona$LoanAmount*(.00833*(1.00833^((train_nona$Loan_Amount_Term)/12))/((1.00833)^(train_nona$Loan_Amount_Term)/12)))
write.csv(train_nona,file='trainf_Emiadded.csv')


testf1['EMI']=(testf1$LoanAmount*(.00833*(1.00833^((testf1$Loan_Amount_Term)/12))/((1.00833)^(testf1$Loan_Amount_Term)/12)))
write.csv(testf1,file='testf1.csv')


#----------------------------------------------------------------------------------------
#some useful functions
#----------------------------------------------------------------------------------------
#To delete all the variables 
rm(list=ls(all=TRUE))

