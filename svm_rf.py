import pandas as pd 
import numpy as np 
from sklearn.feature_extraction import DictVectorizer 
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.grid_search import GridSearchCV  
from statistics import mode

df = pd.read_csv("feature_output.csv")
label = np.asarray(df.result)
df_selected = df.drop(df.columns[:5],axis=1)


def svm_classification(features_train, features_test, labels_train,labels_test,name):
	# initialize 
	clf =  LinearSVC() 
	# train the classifier using the training data 
	clf.fit(features_train, labels_train)
	# compute accuracy using test data 
	acc_test = clf.score(features_test, labels_test) 
	print (name+" SVM Test Accuracy:", acc_test) 


def rf_classification(features_train, features_test, labels_train,labels_test,name):
	total_acc_test=0
	for time in range(1,31):
		# initialize 
		clf = RandomForestClassifier(n_estimators=100)  
		# train the classifier using the training data 
		clf.fit(features_train, labels_train)
		# compute accuracy using test data 
		acc_test = clf.score(features_test, labels_test) 
		total_acc_test+=acc_test
	print (name+" RF Test Average Accuracy:", total_acc_test/30) 


features = df_selected.as_matrix()
indices = np.arange(df_selected.shape[0])
features_train, features_test, labels_train,labels_test, idex_train, index_test= train_test_split( features, label, indices,test_size=0.20, random_state=0)
# print labels_test.tolist()
svm_classification(features_train, features_test, labels_train,labels_test,'set1')
rf_classification(features_train, features_test, labels_train,labels_test,'set1')
