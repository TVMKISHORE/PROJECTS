# module Tunes and Trains the Loan Dataset
#Import libraries:
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier  #GBM algorithm
from sklearn import cross_validation, metrics   #Additional scklearn functions
from sklearn.grid_search import GridSearchCV   #Perforing grid search
import matplotlib.pylab as plt
#matplotlib inline
from matplotlib.pylab import rcParams

rcParams['figure.figsize'] = 12, 4
path='C:\Users\Admin\Documents\Trainl.csv'
train = pd.read_csv(path)

target = 'Disbursed'
IDcol='Loan_ID'
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
        print ("CV Score : Mean - %.7g | Std - %.7g | Min - %.7g | Max - %.7g" % np.mean(cv_score),np.std(cv_score),np.min(cv_score),np.max(cv_score))
 
    #Print Feature Importance:
    if printFeatureImportance:
       feat_imp = pd.Series(alg.feature_importances_, predictors).sort_values(ascending=False)
       feat_imp.plot(kind='bar', title='Feature Importances')
       plt.ylabel('Feature Importance Score')
       
#without any tuning. Lets find out what it gives:
#Choose all predictors except target & IDcols
#predictors = [x for x in train.columns if x not in [target, IDcol,'gender','education']]
predictors = [x for x in train.columns if x not in [target, IDcol,'P_A_Urban','Dep__3+','gender','education']]
gbm0 = GradientBoostingClassifier(random_state=10)
#gbm0=GradientBoostingClassifier(learning_rate=0.1, n_estimators=80,max_depth=1, min_samples_split=3, min_samples_leaf=1, subsample=0.85, random_state=10)
modelfit(gbm0, train, predictors)        

#Feature Imp
Fea_imp1=sorted(zip(gbm0.feature_importances_,predictors))

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