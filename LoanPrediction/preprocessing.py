import numpy as np
import pandas as pd
def preprocessing(loan_data):
    loan_data['Disbursed']=np.where(loan_data.Loan_Status=='Y', 1, 0)
    #target = 'Disbursed'
    #IDcol='Loan_ID'
    #converting features into numerics
    loan_data['gender']=np.where(loan_data.Gender=='Male', 1, 0)
    loan_data['married']=np.where(loan_data.Married=='Yes', 1, 0)
    loan_data['education']=np.where(loan_data.Education=='Graduate', 1, 0)
    loan_data['self_Employed']=np.where(loan_data.Self_Employed=='Yes', 1, 0)
    #how many unique values are there?
    #set(train.Dependents)
    Dep_dummies = pd.get_dummies(loan_data.Dependents, prefix='Dep_')
    loan_data = pd.concat([loan_data, Dep_dummies], axis=1)

    #set(train.Property_Area)    
    property_area = pd.get_dummies(loan_data.Property_Area, prefix='P_A')
    loan_data = pd.concat([loan_data, property_area], axis=1)

    #train=train.drop('Property_Area',axis=1)
    #train=train.drop('Dependents',axis=1)
    col_names=['Loan_ID', 'gender', 'married', 'Dep__1','Dep__2','Dep__3+', 'education','self_Employed', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
       'Loan_Amount_Term', 'Credit_History', 'P_A_Semiurban','P_A_Urban','P_A_Rural','Disbursed']
    #del(Dep_dummies)
    #del(property_area)
    train=loan_data[col_names] 
    return(train) 