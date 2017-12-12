# Project Title

Quora Duplication Detection

## Getting Started

This ?readme? provides detailed instructions on how to run get the datasets, extract features, train the features, and evaluate the system.

### Prerequisites


In order to run the system, you will need the following packages:

# Python 3.6
# Tensorflow 1.4.0
# Nltk 3.2.4
# Pandas 0.20.3
# Numpy 1.13.1
# Sklearn 0.18

### Installing

You can install the tensorflow by:
```
$ pip install tensorflow
```
The rest of the packages can be installed using the same method. It is also strongly advised  to set up a virtual environment for this project.

## Run the system

### Getting the datasets

The first step is to download the raw dataset from Quora (https://data.quora.com/First-Quora-Dataset-Release-Question-Pairs) or use the file we downloaded in the root folder named ?quora_duplicate_questions.tsv?. 

The raw data needs preprocessing and cleaning. Locate to the project root folder and run quora_data_cleaning.py to get the cleaned data for feature extraction:
```
$ python quora_data_cleaning.py
```
This will generate a cleaned version of the dataset called ?quora_lstm.tsv?. This file will be used in later steps to generate all the features.

### Extracting features

Step 0: Get external libraries
Our traditional features involve external libraries, databases, and other data. So before run the extracting features file, we need to run the code to make the external resources usable.

First, go to the ?ppdb? folder and run the ppdb.py using:
```
$ python ppdb.py
```
This will generate a file called ?ppdb-output.json?.

Then go to the the ?glove_parse? folder and run ?glove_parse.py? using the same way. Finally, go to the ?tfidf_data? folder and run ?tfidf.py?.

The above steps are only required if you only have the raw file. In the system we submitted, we have deleted the raw files and uploaded the parsed files. So you do not need to run any of the above python code.

Step 1: get all the traditional features
Then run feature_extraction.py to create a feature list for each cross validation set
```
python feature_extraction.py [tf-idf file path] [result file path]
```
For example
```
# This will take a *long* time
$ python feature_extraction.py tfidf1_2_3_5.json feature_set4.csv
```
This command will return the feature list of cross validation set 4 based on our precalculated tf-idf file. This is a time consuming task, we use 4 different machines(only using CPU) and it took us approximately 2+ hours for each machine. You will need to run this step for five times in order to do a cross validation on the whole dataset (we have split the raw file into five subsets and stored them in the folder called ?train?.

We have stored the traditional features we extracted in the folder called ?features?. Please note that the files also contain the features we extracted from the LSTM features

Step 2: Get the LSTM features
First, reorganize the quora data to fit the LSTM model
Locate to folder ?./lstm? and run clean_quora.py
```
$ python clean_quora.py
```
The script will generate a new file named quora_lstm_new.tsv, which can be used to generate our lstm feature.
```
$ python train.py [options/defaults]
options:
--training_files TRAINING_FILES
                        Comma-separated list of training files (each file is
                        tab separated format) (default: None)
--num_epochs NUM_EPOCHS
                        Number of training epochs (default: 30, which was limited by our machine)
```
For example, in order to run one cross validation file:
```
# This will take a *long* time
$ python train.py --training_files ./train/train_1_2_3_4.tsv
```
These cross validation file are generated by clean.py in folder ?./lstm/train?, and they can be found in the same folder.

After the model is trained, you can see running graphs in a new folder ?./lstm/runs?
The eval.py code will use the model saved in this folder.
To run the eval.py code
```
$ python eval.py [options/default]
Options:
	--vocab_filepath VOCAB_FILEPATH
			   Load training time vocabulary (default: None)
--model MODEL_FILEPATH
		   Load trained model checkpoint (Default: None)
```
The correct vocab and model name can be found in ?./lstm/runs? file, the graph name is named according to time, so please make sure you change the file path when running eval after each train.
The Options can also be edited in file eval.py on line 8 and line 10
Especially thanks to github user dhwajraj for this part, see his/her repository: https://github.com/dhwajraj/deep-siamese-text-similarity

### Training the features

Add the LSTM feature as the last column in each feature_setn.csv (n:1-5). The final version features are stored under ??features? folder. The folder includes feature_set1.csv to feature_set5.csv. Each file includes all the features in each validation set. 
Experiment in SVM model, Random Forest Model and Logistic Regression:
```
$ python svm_rf.py
$ python logistic_regression.py
```
The program will read the feature files under ?features? folder. 
svm_rf.py will first print out the SVM model accuracy and the average accuracy of using Random Forest model under each cross validation. 

Logistic_regression.py will read print out first the L2 regularization model accuracy for each set (1 to 5) and then outputs the L1 regularization model accuracy for each set. 

### Evaluating the system

To get the random baseline and the majority baseline:
```
$ python baselien.py
```
This program will read quora_lstm.tsv and It will print out the random baseline and the majority baseline.
```
random baseline:
0.534137131442659
majority baseline:
0.6307820554681548
```
The accuracy of each model will be printed out when running svm_rf.py

## Feature Ranking
The ranking generated by Random Forest is generated by feature_rank.py file.
The ranking generated by Logistic Regression is generated by logistic_regression.py file.
After run the file, it will print out the feature ranking.
```
$ python feature_rank.py
$ python logistic_regression.py
```

## Built With
* [Tensorflow](https://www.tensorflow.org/) - The Siamese LSTM based
* [Tensorflow](http://scikit-learn.org/) - Used to evaluate our system


## Authors

* **Chong Li**
* **Anna Zheng**
* **Phoebe Liang**
* **Tianyi Liu**

## Acknowledgments

* dhwajraj?s Siamese LSTM repository: https://github.com/dhwajraj/deep-siamese-text-similarity