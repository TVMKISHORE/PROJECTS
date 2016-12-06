
#------------------------------------------------------------------------------------------
#Below is the preprocessing of train data
#------------------------------------------------------------------------------------------
train <- read.csv("C:/Users/Admin/Downloads/Big Mart Sales III/Train_UWu5bXk.csv", na.strings=c("","NA","NaN"))

train$Outlet_Size[is.na(train$Outlet_Size)]<-'Small'  # imputation
train$Item_Fat_Content=as.character(train$Item_Fat_Content)
train[c(row.names(train[ which(train$Item_Type=='Household' | train$Item_Type == 'Health and Hygiene'), ])),]$Item_Fat_Content="Non-Con"
train[c(which(train$Item_Fat_Content=='Low Fat')),]$Item_Fat_Content='LF'
train[c(which(train$Item_Fat_Content=='Regular')),]$Item_Fat_Content='Reg'
train$Item_Fat_Content=as.factor(train$Item_Fat_Content)
train['prod_code']=substr(train$Item_Identifier,start=1,stop=3)
train['outlet_age']=2016-train$Outlet_Establishment_Year
train_nona=train[complete.cases(train),]   #Remove NA data



#------------------------------------------------------------------------------------------
#Below is the preprocessing of test data
#------------------------------------------------------------------------------------------
test <- read.csv("C:/Users/Admin/Downloads/Big Mart Sales III/Test_u94Q5KV.csv", na.strings=c("","NA"))
test$Outlet_Size[is.na(test$Outlet_Size)]<-'Small'  # imputation
test['prod_code']=substr(test$Item_Identifier,start=1,stop=3)

test$Item_Fat_Content=as.character(test$Item_Fat_Content)
test[c(row.names(test[ which(test$Item_Type=='Household' | test$Item_Type == 'Health and Hygiene'), ])),]$Item_Fat_Content="Non-Con"
test[c(which(test$Item_Fat_Content=='Low Fat')),]$Item_Fat_Content='LF'
test[c(which(test$Item_Fat_Content=='Regular')),]$Item_Fat_Content='Reg'
test$Item_Fat_Content=as.factor(test$Item_Fat_Content)
test['prod_code']=substr(test$Item_Identifier,start=1,stop=3)
test['outlet_age']=2016-test$Outlet_Establishment_Year
test_nona=test[complete.cases(test),]   #Remove NA data



train_cols=c(  "Item_Identifier"           ,"Item_Weight"               ,"Item_Fat_Content"         
               ,"Item_Visibility"           ,"Item_Type"                 ,"Item_MRP"                 
               ,"Outlet_Identifier"         ,"Outlet_Establishment_Year" ,"Outlet_Size"              
#              ,"Outlet_Location_Type"      ,"Outlet_Type"               ,"Item_Outlet_Sales"        
               ,"Outlet_Location_Type"      ,"Outlet_Type"               ,"prod_code"         
               ,"outlet_age"      )



look_up=rbind(train_nona[train_cols],test_nona[train_cols])


#------------------------------------------------------------------------------------
# Get the missing Item weights from other stores for boths test  and train datasets
#------------------------------------------------------------------------------------
for (i in 1:nrow(train)) {
  if (is.na(train$Item_Weight[i])=='TRUE'){
    item<-as.character(train$Item_Identifier[i])
    temp_df=subset(look_up, Item_Identifier==item)
    train$Item_Weight[i]<-mean(temp_df$Item_Weight[1])
  }  
}


for (i in 1:nrow(test)) {
  if (is.na(test$Item_Weight[i])=='TRUE'){
    item<-as.character(test$Item_Identifier[i])
    temp_df=subset(look_up, Item_Identifier==item)
    test$Item_Weight[i]<-mean(temp_df$Item_Weight)
  }  
}

rm(test_nona,train_nona,temp_df,look_up) # Delete un necessary cols





#---------------------------------------------------------------------------------------------------
# The below code creates a data frame 'Outlets'.
# outlet_age-->2016-train$Outlet_Establishment_Year
# Creating a outlet demand feature --> (Outlet_Sales/Item_MRP)*outlet_age
#---------------------------------------------------------------------------------------------------
# T R A I N   data
#---------------------------------------------------------------------------------------------
train['outlet_age']=2016-train$Outlet_Establishment_Year
Outlet_Weight=(train$Item_Outlet_Sales/train$Item_MRP)*train$outlet_age
train['outlet_Weight']=as.data.frame(Outlet_Weight)
outlet_profile <- aggregate(train, by=list(train$Outlet_Identifier), FUN=mean)
train$outlet_Weight<-NULL
Outlets=outlet_profile[c("Group.1","outlet_Weight")]
names(Outlets)<- c("Outlet_Identifier","Outlet_Weight")
#----------------------------------------------------------------------------------------------
#lookup and add the feature to train data
#----------------------------------------------------------------------------------------------
what_tolook=c("Outlet_Identifier")
what_toget=c("Item_Weight" ,         "Item_Fat_Content",         
       "Item_Visibility",      "Item_Type" ,      "Item_MRP"  ,          
       "Outlet_Identifier",    "outlet_age",      "Outlet_Size"     ,            
       "Outlet_Location_Type", "Outlet_Type" ,    "Outlet_Weight" ,"Item_Outlet_Sales"     )

train1=lookup_table(train,Outlets,what_tolook,what_toget)

#---------------------------------------------------------------------------------------------
# T R A I N   data
#---------------------------------------------------------------------------------------------
test['outlet_age']=2016-test$Outlet_Establishment_Year
#Outlet_Weight=(test$Item_Outlet_Sales/test$Item_MRP)*test$outlet_age
#test['outlet_Weight']=as.data.frame(Outlet_Weight)
#outlet_profile <- aggregate(test, by=list(test$Outlet_Identifier), FUN=mean)
#test$outlet_Weight<-NULL
#Outlets=outlet_profile[c("Group.1","outlet_Weight")]
#names(Outlets)<- c("Outlet_Identifier","Outlet_Weight")
#----------------------------------------------------------------------------------------------
#lookup and add the feature to test data
#----------------------------------------------------------------------------------------------
what_tolook=c("Outlet_Identifier")
what_toget=c("Item_Weight" ,         "Item_Fat_Content",         
             "Item_Visibility",      "Item_Type" ,      "Item_MRP"  ,          
             "Outlet_Identifier",    "outlet_age",      "Outlet_Size"     ,            
             "Outlet_Location_Type", "Outlet_Type" ,    "Outlet_Weight"  )

test1=lookup_table(test,Outlets,what_tolook,what_toget)














#---------------------------------------------------------------------------------------------------
# Below is just a rough work.
#---------------------------------------------------------------------------------------------------

df2=train_na[c("Item_Identifier","Item_Weight")]

what_toget=c("Item_Identifier","Item_Weight.y","Item_Fat_Content","Item_Visibility","Item_Type","Item_MRP",                 
             "Outlet_Identifier","Outlet_Establishment_Year","Outlet_Size","Outlet_Location_Type","Outlet_Type","Item_Outlet_Sales")

what_tolook=c("Item_Identifier")

rename_cols=c("Item_Identifier","Item_Weight","Item_Fat_Content","Item_Visibility","Item_Type","Item_MRP",                 
                "Outlet_Identifier","Outlet_Establishment_Year","Outlet_Size","Outlet_Location_Type","Outlet_Type","Item_Outlet_Sales")
              

trainf=lookup_table(train_nona,df2,what_tolook,what_toget)
#Lookup function
lookup_table <- function(df1,df2,what_tolook,what_toget){
  temp_df = merge(df1,df2,by=what_tolook)
  ret_df  = temp_df[what_toget]
  return(ret_df)
}

rm(list=ls(all=TRUE))
