import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression  
from sklearn.feature_selection import RFE
from sklearn import svm

count = 0
for infile in glob.glob('word_embedding_Phoebe/*.csv'):
    print ("file: ", infile)
    count += 1
    globals()['df'+str(count)] = pd.read_csv(infile)

dataframes = [df1, df2, df3, df4, df5]

def split_data(df):
    label = np.asarray(df.result)
    df_selected = df.drop(df.columns[:5],axis=1)

    # removed LSTM columns: df.drop('column_name', axis=1, inplace=True)
    # df_selected.drop('LSTM', axis=1, inplace=True)

    # keep only LSTM
    # features = df_selected.iloc[:,-1].reshape(-1, 1)

    features = df_selected.as_matrix()
    
    indices = np.arange(df_selected.shape[0])
    features_train, features_test, labels_train,labels_test, index_train, index_test= train_test_split( features, label, indices,test_size=0.20, random_state=0)
    return features_train, features_test, labels_train,labels_test, index_train, index_test

# L2 Logistic Regression 

def logistic_classification_L2(features_train, features_test, labels_train,labels_test,name):
    total_acc_test=0
    for time in range(1,31):
        # initialize 
        clf_l2_LR = LogisticRegression(C=0.01, penalty='l2', tol=0.01)
        # train the classifier using the training data 
        clf_l2_LR.fit(features_train, labels_train)
        # compute accuracy using test data 
        acc_test = clf_l2_LR.score(features_test, labels_test) 
        total_acc_test+=acc_test
    print (name+" L2 Test Average Accuracy:", total_acc_test/30) 

count = 0
for df in dataframes: 
    count += 1
    name = "set" + str(count)
    print "name: ", name
    features_train, features_test, labels_train,labels_test, index_train, index_test = split_data(df)
    logistic_classification_L2(features_train, features_test, labels_train,labels_test, name)

# L1 logistic regression 

def logistic_classification_L1(features_train, features_test, labels_train,labels_test,name):
    total_acc_test=0
    for time in range(1,31):
        # initialize 
        clf_l1_LR = LogisticRegression(C=0.01, penalty='l1', tol=0.01)
        # train the classifier using the training data 
        clf_l1_LR.fit(features_train, labels_train)
        # compute accuracy using test data 
        acc_test = clf_l1_LR.score(features_test, labels_test) 
        total_acc_test+=acc_test
    print (name+" L1 Test Average Accuracy:", total_acc_test/30) 

count = 0
for df in dataframes: 
    count += 1
    name = "set" + str(count)
    print "name: ", name
    features_train, features_test, labels_train,labels_test, index_train, index_test = split_data(df)
    logistic_classification_L1(features_train, features_test, labels_train,labels_test, name)
    

# find out which features are most important in L2 Regression using features set 1
features_train, features_test, labels_train,labels_test, index_train, index_test = split_data(df1)
model = LogisticRegression(C=0.01, penalty='l2', tol=0.01)
rfe = RFE(estimator=model, n_features_to_select=1, step=1)
rfe = rfe.fit(features_train, labels_train)
# summarize the selection of the attributes
print(rfe.support_)
print(rfe.ranking_)

# find out which features are most important in L1 Regression using features set 1
features_train, features_test, labels_train,labels_test, index_train, index_test = split_data(df1)
model = LogisticRegression(C=0.01, penalty='l1', tol=0.01)
rfe = RFE(estimator=model, n_features_to_select=1, step=1)
rfe = rfe.fit(features_train, labels_train)
# summarize the selection of the attributes
print(rfe.support_)
print(rfe.ranking_)
