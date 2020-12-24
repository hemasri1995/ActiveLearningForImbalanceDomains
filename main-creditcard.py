import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import math
import statistics as stat
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import matplotlib.gridspec as gridspec

def readData(filename):
    data = pd.read_csv(filename)
    return data

def plotHistogram(dataframe):

    hist_list = ['Time', 'V17', 'V14', 'V12', 'Amount']
    for i in hist_list:
        df1 = dataframe[dataframe.Class == 0]
        df2 = dataframe[dataframe.Class == 1]
        fig2 = plt.figure(constrained_layout=True)
        spec2 = gridspec.GridSpec(ncols=2, nrows=1, figure=fig2)
        f2_ax1 = fig2.add_subplot(spec2[0, 0])
        f2_ax2 = fig2.add_subplot(spec2[0, 1])
        sns.distplot(df1[i], kde=False, label='0', ax=f2_ax1)
        f2_ax1.set_title(str(i) + ' Feature distribution over Frequency for label 0')
        sns.distplot(df2[i], kde=False, label='1', ax=f2_ax2)
        f2_ax2.set_title(str(i) + ' Feature distribution over Frequency for label 1')
        plt.xlabel(str(i))
        plt.ylabel('Frequency')
        plt.show()



def plotHeatmap(dataframe):
    sns.set_theme()
    ax = sns.heatmap(dataframe.corr(method='pearson'))
    plt.title("Heatmap for features using pearson correlation ")
    plt.show()

    ax = sns.heatmap(dataframe.corr(method='kendall'))
    plt.title("Heatmap for features using kendall correlation ")
    plt.show()

    ax = sns.heatmap(dataframe.corr(method='spearman'))
    plt.title("Heatmap for features using spearman correlation ")
    plt.show()

def plotBoxPlot(dataframe):

    for i in data.columns:
        f, (ax_hist, ax_box) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.85, .15)})
        sns.distplot(dataframe[i], ax=ax_hist)
        sns.boxplot(dataframe[i], ax=ax_box)

        ax_box.set(xlabel=str(i)+' feature')
        plt.show()



def plotScatterPlot(dataframe):
    sns.scatterplot(data=dataframe, x="V2", y="Amount", hue="Class")
    plt.show()

    sns.scatterplot(data=dataframe, x="V5", y="Amount", hue="Class")
    plt.show()

    sns.scatterplot(data=dataframe, x="V7", y="Amount", hue="Class")
    plt.show()

    sns.scatterplot(data=dataframe, x="V20", y="Amount", hue="Class")
    plt.show()


if __name__ == '__main__':
    print("Analysing Data with Visualization")
    data = readData("./datasets/creditcard.csv")
    print(type(data))
    #display information of the dataset
    print(data.info())
    print("======================================================================")
    print("mean for each feature")
    print(data.mean(axis=0))
    print("======================================================================")
    print("max for each feature")
    print(data.max(axis=0))
    print("======================================================================")
    print("min for each feature")
    print(data.min(axis=0))
    print("======================================================================")
    print("standard deveation for each feature")
    for column in data:
        print(column)
        print(np.std(data[column]))
    print(np.std(data))
    cat_list= data.select_dtypes(include=['object', 'category']).columns.tolist()
    if not cat_list:
        print("no catagorical data in the features of data set")
    else:
        print(str(cat_list)[1:-1])


    #print the number of 0 and 1 in class label
    print(pd.value_counts(data.Class))

    print("=================================================================")
    print(data[data.columns[1:]].corr()['Class'][:])

    plotHeatmap(data)
    plotScatterPlot(data)
    plotHistogram(data)
    plotBoxPlot(data)




















