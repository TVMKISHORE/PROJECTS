#You may refer the below links for documentstion.
#-----------------------------------------------------------------------
#https://www.analyticsvidhya.com/blog/2016/12/introduction-to-feature-selection-methods-with-an-example-or-how-to-select-the-right-variables/
#https://www.analyticsvidhya.com/blog/2016/03/select-important-variables-boruta-package/
#-----------------------------------------------------------------------
#------------------------------------------------------------------------------------------
#Below is the preprocessing of test data
#------------------------------------------------------------------------------------------
wrapdata <- read.csv("C:/Users/Admin/Downloads/Big Mart Sales III/Train_UWu5bXk.csv", na.strings=c("","NA","NaN"))
wrapdata$Outlet_Size[is.na(wrapdata$Outlet_Size)]<-'Small'  # imputation (Missing depends on missing. Identified by exploration on Excel)
wrapdata$Item_Fat_Content=as.character(wrapdata$Item_Fat_Content)
wrapdata[c(row.names(wrapdata[ which(wrapdata$Item_Type=='Household' | wrapdata$Item_Type == 'Health and Hygiene'), ])),]$Item_Fat_Content="Non-Con"
wrapdata[c(which(wrapdata$Item_Fat_Content=='Low Fat')),]$Item_Fat_Content='LF'
wrapdata[c(which(wrapdata$Item_Fat_Content=='Regular')),]$Item_Fat_Content='Reg'
wrapdata$Item_Fat_Content=as.factor(wrapdata$Item_Fat_Content)
wrapdata['prod_code_text']=substr(wrapdata$Item_Identifier,start=1,stop=3)
wrapdata['prod_code_num']=substr(wrapdata$Item_Identifier,start=4,stop=6)
wrapdata['outlet_age']=2016-wrapdata$Outlet_Establishment_Year
wrapdata_nona=wrapdata[complete.cases(wrapdata),]   #Remove NA data


install.packages("Boruta")
library(Boruta)

set.seed(123)
boruta.train <- Boruta(Item_Outlet_Sales~.-Item_Identifier, data = wrapdata_nona, doTrace = 2)
print(boruta.train)

#--------------------------------------------------------------Output
#Boruta performed 99 iterations in 1.00042 hours.
#6 attributes confirmed important: Item_MRP, Item_Visibility, Outlet_Establishment_Year,
#Outlet_Identifier, Outlet_Location_Type and 1 more.
#2 attributes confirmed unimportant: Item_Fat_Content, Outlet_Size.
#2 tentative attributes left: Item_Type, Item_Weight.
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
#--------------------------------------------------------------Output
#[1] "Item_Visibility"           "Item_MRP"                  "Outlet_Identifier"        
#[4] "Outlet_Establishment_Year" "Outlet_Location_Type"      "Outlet_Type"              


#--------------------------------------------------------------
#We'll create a data frame of the final result derived from Boruta.
#--------------------------------------------------------------
boruta.df <- attStats(final.boruta)
class(boruta.df)
#[1] "data.frame"
print(boruta.df)
#The below table will be displayed
#meanImp   medianImp   minImp    maxImp   normHits    decision
#--------------------------------------------------------------









#-------------------------------------------------------------------------------------------------
#                               Second run Boruta package
#-------------------------------------------------------------------------------------------------
#--------------------------------
# The below featurs are already found Good from the above first run
fea_col=
  c("Item_Visibility",
    "Item_MRP",
    "Outlet_Identifier",
    "Outlet_Establishment_Year",
    "Outlet_Location_Type",
    "Outlet_Type"     )
wrapdata_nona[,fea_col]<-NULL  # above features can be removed 



set.seed(123)
boruta.train <- Boruta(Item_Outlet_Sales~.-Item_Identifier, data = wrapdata_nona, doTrace = 2)
print(boruta.train)


getSelectedAttributes(boruta.train, withTentative = F)
#--------------------------------------------------------------Output
#[1] "Item_Weight"      "Item_Fat_Content" "Item_Type"        "Outlet_Size"      "prod_code_text"  
#[6] "prod_code_num"    "outlet_age"   



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
#Conclusion 
##Since Boruta is computationally expensive , I ran the package twice.
##1. A Model build withthe features given by first Boruta and then tried with combinations with features given by second Bauta.
##2. looks like only can add value "prod_code_num"
#--------------------------------------------------------------

