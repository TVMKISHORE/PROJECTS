# module Tunes and Trains the Loan Dataset
#Import libraries:
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier  #GBM algorithm
from sklearn import cross_validation, metrics   #Additional scklearn functions
from sklearn.grid_search import GridSearchCV   #Perforing grid search
import matplotlib.pylab as plt
import math as ma
#matplotlib inline
from matplotlib.pylab import rcParams

rcParams['figure.figsize'] = 12, 4
path='C:\Users\Admin\Documents\Trainl.csv'
train = pd.read_csv(path)

target = 'Disbursed'
IDcol='Loan_ID'
train['Disbursed']=np.where(train.Loan_Status=='Y', 1, 0)
#**********************************************************
#Before proceeding further, lets define a function which will help us create GBM models and perform cross-validation.

def modelfit(alg, dtrain, predictors, performCV=True, printFeatureImportance=True, cv_folds=5):
    #Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain['Disbursed'])      
    #Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]
    
    #Perform cross-validation:
    if performCV:
       cv_score = cross_validation.cross_val_score(alg, dtrain[predictors], dtrain['Disbursed'], cv=cv_folds, scoring='roc_auc')

    #Print model report:
    print ("\nModel Report")
    print ("Accuracy : %.4g" % metrics.accuracy_score(dtrain['Disbursed'].values, dtrain_predictions))
    print ("AUC Score (Train): %f" % metrics.roc_auc_score(dtrain['Disbursed'], dtrain_predprob))
    
    if performCV:      
        print ("CV Score : Mean - %.7g" % np.mean(cv_score))
        print("Std - %.7g" % np.std(cv_score))
        print ("Min - %.7g" % np.min(cv_score))
        print ("Max - %.7g" % np.max(cv_score))
 
    #Print Feature Importance:
    if printFeatureImportance:
       feat_imp = pd.Series(alg.feature_importances_, predictors).sort_values(ascending=False)
       feat_imp.plot(kind='bar', title='Feature Importances')
       plt.ylabel('Feature Importance Score')
       
#without any tuning. Lets find out what it gives:
#Choose all predictors except target & IDcols
#predictors = [x for x in train.columns if x not in [target, IDcol,'gender','education']]
predictors = [x for x in train.columns if x not in [target,'Dependents1','Dependents2','Property_AreaRural','Property_AreaSemiurban','Loan_Status','Unnamed: 0',]]
#predictors = [x for x in train.columns if x not in [target,'Loan_Status','Unnamed: 0',]]
gbm0 = GradientBoostingClassifier(random_state=10)
#gbm0=GradientBoostingClassifier(learning_rate=0.1, n_estimators=80,max_depth=1, min_samples_split=3, min_samples_leaf=1, subsample=0.85, random_state=10)
modelfit(gbm0, train, predictors)        

#Feature Imp
Fea_imp1=sorted(zip(gbm0.feature_importances_,predictors))

#test file accuracy
predictors_test = [x for x in test.columns if x not in [IDcol,target,'Loan_Status','Unnamed: 0',]]
dtest_predprob=gbm0.predict(test[predictors_test])
dtest_predprob=pd.DataFrame(dtest_predprob)
dtest_predprob.columns=["Loan_Status"]
up_load_toAV=pd.concat([test["Loan_ID"],dtest_predprob["Loan_Status"]],axis=1)
up_load_toAV["Loan_Status"]=np.where(up_load_toAV.Loan_Status==1, 'Y', 'N')
up_load_toAV.to_csv('up_load_toAV.csv',index=False)
#Model Report
#Accuracy : 0.9042
#AUC Score (Train): 0.976097

gbm0.feature_importances_

'''
array([ 0.00487927,  0.01431704,  0.01050182,  0.00774186,  0.3144491 ,
        0.16125786,  0.21960323,  0.06023442,  0.13543204,  0.00622591,
        0.02832669,  0.00562533,  0.00208916,  0.01817703,  0.00701472,
        0.00412452])
'''

#Please note that all the above are just initial estimates and will be tuned later
#Choose all predictors except target & IDcols
predictors = [x for x in train.columns if x not in [target, IDcol]]
c=[i for i in range(10,100,10)]
param_test1 = [{'n_estimators':c}]
gsearch1 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1,min_samples_leaf=5,max_features='sqrt',subsample=0.8,random_state=10,min_samples_split=3, max_depth=1),
                        param_grid = param_test1, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch1.fit(train[predictors],train[target])
#concluded that  gsearch1.best_params_={'n_estimators': 80}

#tune for max_depth and min_sample_split
mx=[i for i in range(1,16,2)]
mn=[i for i in range(3,10,1)]
#param_test2 = [{'max_depth':mx},{'min_samples_split':mn}]
param_test2 = [{'min_samples_split':mn}]
param_test2 = [{'max_depth':mx}]
gsearch2 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=20, max_features='sqrt', subsample=0.8, random_state=10,max_depth=1), 
param_grid = param_test2, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch2.fit(train[predictors],train[target])
gsearch2.grid_scores_, gsearch2.best_params_, gsearch2.best_score_
# conclude that min_samples_split': 3, max_depth=1


#Tune Maximum number of features total features 15
fe=[i for i in range(1,16,2)]
param_test4 = {'max_features':fe}
gsearch4 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=80,max_depth=1, min_samples_split=3, min_samples_leaf=1, subsample=0.8, random_state=10),
param_grid = param_test4, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch4.fit(train[predictors],train[target])
gsearch4.grid_scores_, gsearch4.best_params_, gsearch4.best_score_
#Concluded that the maximum max_features:1


#Tuning subsample
sa=[i for i in range(1,16,2)]
param_test5 = {'subsample':[0.6,0.7,0.75,0.8,0.85,0.9]}
gsearch5 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.005, n_estimators=80,max_depth=1, min_samples_split=3, min_samples_leaf=1,max_features=1,subsample=0.8, random_state=10),
param_grid = param_test5, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch5.fit(train[predictors],train[target])
gsearch5.grid_scores_, gsearch5.best_params_, gsearch5.best_score_
#concluded that .85 is the best subsample






#---------------------------------------------------------------------
# Try with basic logistic regression
#---------------------------------------------------------------------
import pandas as pd
import numpy as np

target = 'Disbursed'
IDcol='Loan_ID'
train['Disbursed']=np.where(train.Loan_Status=='Y', 1, 0)


from sklearn.linear_model import LogisticRegression
predictors = [x for x in train.columns if x not in ['Loan_Amount_Term.x','Dependents1','Dependents2','Property_AreaRural','Property_AreaSemiurban',target,'Loan_Status','Unnamed: 0',]]
#predictors = [x for x in train.columns if x not in [target,'Loan_Status','Unnamed: 0',]]
X=train[predictors]
y=train['Disbursed']
#feature scaling
X["Loan_Amount_Term.x"]=(X["Loan_Amount_Term.x"]-X["Loan_Amount_Term.x"].mean())/10
X["CoapplicantIncome.x"]=((X["CoapplicantIncome.x"])-X["CoapplicantIncome.x"].mean())/1000
X["LoanAmount.x"]=((X["LoanAmount.x"])-X["LoanAmount.x"].mean())/100
X['ApplicantIncome.x']=((X['ApplicantIncome.x'])-X['ApplicantIncome.x'].mean())/1000

X["Loan_Amount_Term.x"]=ma.log(X["Loan_Amount_Term.x"])
X["CoapplicantIncome.x"]=ma.log(X["CoapplicantIncome.x"])
X["LoanAmount.x"]=ma.log((X["LoanAmount.x"]))
X['ApplicantIncome.x']=ma.log(X['ApplicantIncome.x'])

#add polynomial features
X['LAT_C']=X["Loan_Amount_Term.x"]*X["CoapplicantIncome.x"]
X['C_LA']=X["CoapplicantIncome.x"]*X["LoanAmount.x"]
X['LA_AI']=X["LoanAmount.x"]*X['ApplicantIncome.x']
X['AI_LAT']=X['ApplicantIncome.x']*X["Loan_Amount_Term.x"]

X['LAT_C_S']=X["Loan_Amount_Term.x"]*X["CoapplicantIncome.x"]**2
X['C_LA_S']=X["CoapplicantIncome.x"]*X["LoanAmount.x"]**2
X['LA_AI_S']=X["LoanAmount.x"]*X['ApplicantIncome.x']**2
X['AI_LAT_S']=X['ApplicantIncome.x']*X["Loan_Amount_Term.x"]**2

X['LAT_C_C']=(X["Loan_Amount_Term.x"]**2)*X["CoapplicantIncome.x"]
X['C_LA_C']=(X["CoapplicantIncome.x"]**2)*X["LoanAmount.x"]
X['LA_AI_C']=(X["LoanAmount.x"]**2)*X['ApplicantIncome.x']
X['AI_LAT_C']=(X['ApplicantIncome.x']**2)*X["Loan_Amount_Term.x"]

X['LAT_Cs']=X["Loan_Amount_Term.x"]*X["CoapplicantIncome.x"]*X["LoanAmount.x"]
X['C_LAs']=X["CoapplicantIncome.x"]*X["LoanAmount.x"]*X['ApplicantIncome.x']
X['LA_AIs']=X["LoanAmount.x"]*X['ApplicantIncome.x']*X["Loan_Amount_Term.x"]
X['LA_LATs']=X["CoapplicantIncome.x"]*X["LoanAmount.x"]*X["Loan_Amount_Term.x"]


#-----------------------------------------
#logistic regression
#-----------------------------------------

#Training with test set

logreg = LogisticRegression()
logGS=GridSearchCV(estimator=logreg, 
             param_grid = param_test1, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
logreg = LogisticRegression(C=1e9,fit_intercept=True,dual=False,penalty='l2',solver="liblinear", tol=0.000001, max_iter=200000, class_weight=None, n_jobs=4, verbose=0,intercept_scaling=1.0, multi_class='ovr', random_state=None)
c=np.arange(1000000000.0,10000000000.0,100000000)
param_test1 = [{'C':c}]
logGS.fit(X, y)
logreg.fit(X, y)

#vaidation on train set
y_pred = logreg.predict(X)
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
confusion_matrix = confusion_matrix(y, y_pred)
print('AUC Score: ',roc_auc_score(y, y_pred))
print ('Accuracy: ', accuracy_score(y, y_pred))

#predection on test set
predictors = [x for x in test.columns if x not in ['Loan_Amount_Term.y','Dependents1','Dependents2','Property_AreaRural','Property_AreaSemiurban',IDcol,target,'Loan_Status','Unnamed: 0',]]
predictors_test=['Credit_History', 'CoapplicantIncome', 'LoanAmount', 'ApplicantIncome']
X=test[predictors_test]
#X.columns=['Credit_History.x', 'CoapplicantIncome.x',  'LoanAmount.x','ApplicantIncome.x']
#X.columns=['Credit_History', 'CoapplicantIncome', 'LoanAmount','ApplicantIncome']

y_test = logreg.predict(X)

dtest_predprob=pd.DataFrame(y_test)
dtest_predprob.columns=["Loan_Status"]
up_load_toAV=pd.concat([test["Loan_ID"],dtest_predprob["Loan_Status"]],axis=1)
up_load_toAV["Loan_Status"]=np.where(up_load_toAV.Loan_Status==1, 'Y', 'N')
up_load_toAV.to_csv('up_load_toAV.csv',index=False)


#-----------------------------------------
#Support vector Machine application 
#-----------------------------------------

#train the model 

from sklearn import svm
#between .6909 and .691
#clf = svm.SVC(C=.6909067129)
clf = svm.SVC(C=.6909067140011100899)
Svmmodel=clf.fit(X, y) 
y_pred=Svmmodel.predict(X)

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
confusion_matrix = confusion_matrix(y, y_pred)
print('AUC Score: ',roc_auc_score(y, y_pred))
print ('Accuracy: ', accuracy_score(y, y_pred))

#predict the test set

predictors_test=['Credit_History', 'CoapplicantIncome', 'LoanAmount', 'ApplicantIncome']
X=test[predictors_test]

y_test = Svmmodel.predict(X)

dtest_predprob=pd.DataFrame(y_test)
dtest_predprob.columns=["Loan_Status"]
up_load_toAV=pd.concat([test["Loan_ID"],dtest_predprob["Loan_Status"]],axis=1)
up_load_toAV["Loan_Status"]=np.where(up_load_toAV.Loan_Status==1, 'Y', 'N')
up_load_toAV.to_csv('up_load_toAV.csv',index=False)

#AUC Score:  0.97635135135
#Accuracy:  0.985416666667

#Thgis model over fits and performed very mad on LB
#  Accuracy :0.65972

