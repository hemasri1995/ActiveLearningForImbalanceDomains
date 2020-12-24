# ActiveLearningForImbalanceDomains
Using anomaly detection and resampling for Active Learning under imbalance domains

Project: Active Learning.
Author: Hema Sri Kambhampati
Organisation: University of Ottawa
Supervisor: Prof Paula Branco.

Description:
This Project is used to evaluate Active Learning with different imbalance datasets, sampling techniques and outlier detection.

Code Structure:
1: datasets
This folder has all datasets.

2: ALGeneric.py
This module has various functions.
    readData - function to loads the file and does basic imputations on data.
    feature_selection - Performs RFC feature selection.
    Lof - Gets top n outlier indices from the dataset using LOF algorithm.
    OneClassSVMOutlier - Gets top n outlier indices from the dataset using OneClassSVM algorithm.
    IForest -  Gets top n outlier indices from the dataset using IForest algorithm.
    plotGraph - plots Graph for provided metric data.
    Main - Contains main implementation.

3: main-adTracking.py, main-cisfraud.py, main-creditcard.py
These module are used to analyse adTracking, cisfraud and credit card Datasets.
functions in module:
    readData - function to loads the file and does basic imputations on data.
    plotHistogram - Plots Histograms for features
    plotHeatmap -  Plots Heatmaps using pearson, kendall, spearman correlation for features
    plotBoxPlot -  Analyse data using Boxplots for features
    plotScatterPlot -  Analyse data using Scatter plots for features

4: results
This folder has results for Active Learning Strategies, Data Sampling and outlier detection.


Prerequisites:

Python Version: 3
Modules required: matplotlib, sklearn, functools, modAL, numpy, pandas, imblearn
Operating System: Windows, Unix.



Example Usage:

Run the file ALGeneric.py:
$ python ALGeneric.py

Upon Running the Main file the program prompts for User inputs.

---------------------------------------
Select data set to perform active learning
1: Credit Card
2: adTracking
3: cis fraud
example: 1
---------------------------------------
Loading Dataset
---------------------------------------
Select Feature Selection option
1: Recursive Feature Selection
2: No Feature Selection
example: 1
---------------------------------------
Starting Feature Selection using Random Forest
Total Number of features 30 , Enter number of best features needed:-
example: 10

Shape of data set :(10000, 10)
Preparing Initial Training, Pool and Test sets for initial AL
training set size: 900
pool set size: 2100
test set size: 7000
---------------------------------------
Select Strategy
1: Ranked Batch using Uncertainity
2: Query By Committee Batch
example: 1
---------------------------------------
---------------------------------------
Enter Batch Size :(Default is 10)

example: 10
---------------------------------------
---------------------------------------
Enter Outlier Detection Method
1: LOF
2: One Class SVM
3: IForest
4: No Outlier

example: 1
---------------------------------------
---------------------------------------
Enter Data sampling Technique for Imbalanced Data
1: SMOTE
2: ADASYN
3: Random Oversampling
4: No Sampling

example: 1

After all inputs are provided by user, the programs starts executing.


Sample Output:

========================
Initial Training Results
========================
Accuracy after query 0: 0.9991
[[34890     6]
 [   25    79]]
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     34896
           1       0.93      0.76      0.84       104

    accuracy                           1.00     35000
   macro avg       0.96      0.88      0.92     35000
weighted avg       1.00      1.00      1.00     35000

AUC: 0.8797217225690402
========================
Learning phase with 10 queries
Total number of records per batch 10  divided into
Uncertain samples: 7
outlier samples: 3
========================
---------------------------------------
Learning phase Query: 1
Getting pool of data for learner using selected sampling method
getting Outliers from data pool
Data Sampling with selected option
teach with query using uncertainity and outlier data
========================
Accuracy after query 1: 0.9987
[[34871    25]
 [   22    82]]
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     34896
           1       0.77      0.79      0.78       104

    accuracy                           1.00     35000
   macro avg       0.88      0.89      0.89     35000
weighted avg       1.00      1.00      1.00     35000

AUC: 0.8938725619863859
---------------------------------------
Learning phase Query: 2
Getting pool of data for learner using selected sampling method
getting Outliers from data pool
Data Sampling with selected option
teach with query using uncertainity and outlier data
========================
Accuracy after query 2: 0.9987
[[34868    28]
 [   19    85]]
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     34896
           1       0.75      0.82      0.78       104

    accuracy                           1.00     35000
   macro avg       0.88      0.91      0.89     35000
weighted avg       1.00      1.00      1.00     35000

AUC: 0.9082526540401368
---------------------------------------
Learning phase Query: 3
Getting pool of data for learner using selected sampling method
getting Outliers from data pool
Data Sampling with selected option
teach with query using uncertainity and outlier data
========================
Accuracy after query 3: 0.9989
[[34872    24]
 [   14    90]]
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     34896
           1       0.79      0.87      0.83       104

    accuracy                           1.00     35000
   macro avg       0.89      0.93      0.91     35000
weighted avg       1.00      1.00      1.00     35000

AUC: 0.9323484287376997
---------------------------------------
Learning phase Query: 4
Getting pool of data for learner using selected sampling method
getting Outliers from data pool
Data Sampling with selected option
teach with query using uncertainity and outlier data
========================
Accuracy after query 4: 0.9989
[[34871    25]
 [   14    90]]
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     34896
           1       0.78      0.87      0.82       104

    accuracy                           1.00     35000
   macro avg       0.89      0.93      0.91     35000
weighted avg       1.00      1.00      1.00     35000

AUC: 0.9323341004479245
---------------------------------------
Learning phase Query: 5
Getting pool of data for learner using selected sampling method
getting Outliers from data pool
Data Sampling with selected option
teach with query using uncertainity and outlier data
========================
Accuracy after query 5: 0.9989
[[34872    24]
 [   16    88]]
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     34896
           1       0.79      0.85      0.81       104

    accuracy                           1.00     35000
   macro avg       0.89      0.92      0.91     35000
weighted avg       1.00      1.00      1.00     35000

AUC: 0.9227330441223152
---------------------------------------
Learning phase Query: 6
Getting pool of data for learner using selected sampling method
getting Outliers from data pool
Data Sampling with selected option
teach with query using uncertainity and outlier data
========================
Accuracy after query 6: 0.9985
[[34861    35]
 [   16    88]]
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     34896
           1       0.72      0.85      0.78       104

    accuracy                           1.00     35000
   macro avg       0.86      0.92      0.89     35000
weighted avg       1.00      1.00      1.00     35000

AUC: 0.9225754329347865
---------------------------------------
Learning phase Query: 7
Getting pool of data for learner using selected sampling method
getting Outliers from data pool
Data Sampling with selected option
teach with query using uncertainity and outlier data
========================
Accuracy after query 7: 0.9985
[[34861    35]
 [   19    85]]
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     34896
           1       0.71      0.82      0.76       104

    accuracy                           1.00     35000
   macro avg       0.85      0.91      0.88     35000
weighted avg       1.00      1.00      1.00     35000

AUC: 0.9081523560117095
---------------------------------------
Learning phase Query: 8
Getting pool of data for learner using selected sampling method
getting Outliers from data pool
Data Sampling with selected option
teach with query using uncertainity and outlier data
========================
Accuracy after query 8: 0.9985
[[34863    33]
 [   19    85]]
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     34896
           1       0.72      0.82      0.77       104

    accuracy                           1.00     35000
   macro avg       0.86      0.91      0.88     35000
weighted avg       1.00      1.00      1.00     35000

AUC: 0.9081810125912602
---------------------------------------
Learning phase Query: 9
Getting pool of data for learner using selected sampling method
getting Outliers from data pool
Data Sampling with selected option
teach with query using uncertainity and outlier data
========================
Accuracy after query 9: 0.9986
[[34863    33]
 [   15    89]]
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     34896
           1       0.73      0.86      0.79       104

    accuracy                           1.00     35000
   macro avg       0.86      0.93      0.89     35000
weighted avg       1.00      1.00      1.00     35000

AUC: 0.9274117818220293
---------------------------------------
Learning phase Query: 10
Getting pool of data for learner using selected sampling method
getting Outliers from data pool
Data Sampling with selected option
teach with query using uncertainity and outlier data
========================
Accuracy after query 10: 0.9986
[[34864    32]
 [   18    86]]
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     34896
           1       0.73      0.83      0.77       104

    accuracy                           1.00     35000
   macro avg       0.86      0.91      0.89     35000
weighted avg       1.00      1.00      1.00     35000

AUC: 0.9130030331887277


This is followed by graphs with all accuarcies, AUC's, F1 scores of minority, Geometric mean of recall of all iterations.

--------------------------------
