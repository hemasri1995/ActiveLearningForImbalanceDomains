# ActiveLearningForImbalanceDomains
Using anomaly detection and resampling for Active Learning under imbalance domains

Project: Active Learning

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



5: Prerequisites:

Python Version: 3

Modules required: matplotlib, sklearn, functools, modAL, numpy, pandas, imblearn

Operating System: Windows, Unix.



