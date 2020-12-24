# Importing all Required modules
import warnings
warnings.filterwarnings("ignore")
from sklearn.decomposition import PCA
import matplotlib as mpl
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from functools import partial
from modAL.batch import uncertainty_batch_sampling, ranked_batch
from modAL.models import ActiveLearner,Committee
from sklearn.metrics import classification_report
import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from numpy import quantile, where, random
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import OneClassSVM
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFE
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
import math
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import IsolationForest



def readData(filename):
    """

    :param filename: String format of file name with the complete path eg: C:/Desktop/creditcard.csv
    :return: returns Data after basic cleaning in the form of Data Frame Object.
    """
    data = pd.read_csv(filename)
    non_floats = []
    for col in data:
        if data[col].dtypes != "float64" and data[col].dtypes != "int64":
            non_floats.append(col)
            data[col] = data[col].astype('category')
    data[non_floats] = data[non_floats].apply(lambda x: x.cat.codes)
    data = data.dropna(thresh=len(data) // 2, axis=1)
    data.fillna(data.mean())
    return data

def feature_selection(dataset,targets):
    """

    :param dataset: dataframe object containing all records of features
    :param targets: dataframe object of target values corresponding to dataset
    :return: returns new dataset with mentioned number of features
    """

    number_of_features = int(input("Total Number of features "+str(dataset.shape[1])+" , Enter number of best features needed:- "))
    selector = RFE(estimator=DecisionTreeClassifier(), n_features_to_select=number_of_features, step=1)
    selector.fit(dataset, targets)
    new_dataset = selector.transform(dataset)

    return pd.DataFrame(new_dataset)




def Lof(data,number_of_top_outliers):
    """

    :param data: features values
    :param number_of_top_outliers: number of top ranked outliers
    :return: all top n outliers indices of data.
    """
    clf = LocalOutlierFactor(n_neighbors=2)
    y_pred = clf.fit_predict(data)
    score = clf.negative_outlier_factor_
    # very small number --> more the outlier
    # eg: -40000 --> outlier
    # -1.2  --> Not Outlier
    top_outlier_indices = score.argsort()[:number_of_top_outliers]
    return top_outlier_indices


def OneClassSVMOutlier(data,number_of_top_outliers):
    """

    :param data: features values
    :param number_of_top_outliers: number of top ranked outliers
    :return: all top n outliers indices of data.
    """

    clf = OneClassSVM(gamma='auto')
    y_pred = clf.fit_predict(data)
    score = clf.score_samples(data)
    # very small number --> more the outlier
    # eg: -40000 --> outlier
    # -1.2  --> Not Outlier
    top_outlier_indices = score.argsort()[:number_of_top_outliers]
    return top_outlier_indices

def IForest(data,number_of_top_outliers):
    """

    :param data: features values
    :param number_of_top_outliers: number of top ranked outliers
    :return: all top n outliers indices of data.
    """
    clf = IsolationForest(random_state=0)
    y_pred = clf.fit_predict(data)
    score = clf.score_samples(data)

    top_outlier_indices = score.argsort()[:number_of_top_outliers]
    return top_outlier_indices



def plotGraph(performance,type = 'Accuracy'):
    """

    :param performance: list of all performance metric values eg: list of accuracies/AUC/F1 etc
    :param type: type of performance metric
    :return: None
    """
    # Plot our performance over time.
    fig, ax = plt.subplots(figsize=(8.5, 6), dpi=130)

    ax.plot(performance)
    ax.scatter(range(len(performance)), performance, s=13)

    ax.set_title('Incremental classification '+str(type))
    ax.set_xlabel('Query iteration')
    ax.set_ylabel('Classification '+str(type))

    plt.show()




# Actice Learning Main Module.

if __name__ == '__main__':



    BATCH_SIZE = 10
    N_RAW_SAMPLES = 100


    # Prompting User to select a dataset from the list of available options.
    # Upon selecting a valid option, it loads the dataset in the form of dataset.
    print("---------------------------------------")
    print("Select data set to perform active learning")
    print("1: Credit Card\n2: adTracking\n3: cis fraud")
    data_option = int(input())
    print("---------------------------------------")
    print("Loading Dataset")
    if(data_option == 1):
        data = readData('./datasets/creditcard.csv')
        y_raw = data['Class']
        X_raw = data.drop('Class', axis=1)
    elif(data_option == 2):
        data = readData('./datasets/adTracking.csv')
        y_raw = data['is_attributed']
        X_raw = data.drop('is_attributed', axis=1)
    elif (data_option == 3):
        data = readData('./datasets/cisfraud.csv')
        y_raw = data['isFraud']
        X_raw = data.drop('isFraud',axis=1)
    else:
        print("enter right option for loading dataset")
        exit(1)

    # Prompting User to select a feature selection option.
    # Upon selecting a valid option, Best features are selected.
    print("---------------------------------------")
    print("Select Feature Selection option\n1: Recursive Feature Selection\n2: No Feature Selection")
    fs_option = int(input())
    print("---------------------------------------")
    if(fs_option == 1):
        print("Starting Feature Selection using Random Forest")
        X_raw = feature_selection(X_raw, y_raw)


    print("Shape of data set :" + str(X_raw.shape))
    # converting data frommdataframe to ndarray
    y_raw = y_raw.values
    X_raw = X_raw.values


    # batch_data_continous_X,y stores data and adds to it after every iteration of batch
    batch_data_continous_X = np.array([]).reshape(0,X_raw.shape[1])
    batch_data_continous_y = np.array([]).reshape(0,1)

    # Spliting dataset into Train, Pool(for batch) and Test.
    print("Preparing Initial Training, Pool and Test sets for initial AL")
    X_train, X_test, y_train, y_test = train_test_split(X_raw, y_raw, test_size=0.7, stratify=y_raw)
    X_train, X_pool, y_train, y_pool = train_test_split(X_train, y_train, test_size=0.7, stratify=y_train)

    print("training set size: "+str(len(X_train)))
    print("pool set size: " + str(len(X_pool)))
    print("test set size: " + str(len(X_test)))


    # lists to store all metric values after each iteration
    accuracy_history = []
    auc_history = []
    f1_minority_history = []
    recal_GM_history = []
    dt = DecisionTreeClassifier()


    # Prompting User to Active Learning Strategy from the list of available options.
    # Upon selecting a valid option, it creates batches using Active Learning Strategy provided.
    print("---------------------------------------")
    print("Select Strategy\n1: Ranked Batch using Uncertainity\n2: Query By Committee Batch")
    strategy_option = int(input())
    print("---------------------------------------")

    # Prompting User to provide batch size eg: 10.
    print("---------------------------------------")
    print("Enter Batch Size :(Default is 10)\n")
    BATCH_SIZE = int(input())
    N_RAW_SAMPLES = 10 * BATCH_SIZE
    print("---------------------------------------")

    # Prompting User to select an outlier detection method from the list of available options.
    # Upon selecting a valid option, it gets best outliers for every batch.
    print("---------------------------------------")
    print("Enter Outlier Detection Method")
    outlier_option = int(input(("1: LOF\n2: One Class SVM\n3: IForest\n4: No Outlier")))
    print("---------------------------------------")

    # Prompting User to select Data Sampling method from the list of available options.
    # Upon selecting a valid option, it samples batch_data_continous_X,y data for every batch.
    print("---------------------------------------")
    print("Enter Data sampling Technique for Imbalanced Data")
    sampling_option = int(input(("1: SMOTE\n2: ADASYN\n3: Random Oversampling\n4: No Sampling\n")))
    print("---------------------------------------")

    if(strategy_option == 1):
        if(outlier_option == 4):
            preset_batch = partial(uncertainty_batch_sampling, n_instances=int(BATCH_SIZE))
        else:
            preset_batch = partial(uncertainty_batch_sampling, n_instances=int(BATCH_SIZE*0.7))
        # Specify our active learning model.
        learner = ActiveLearner(
            estimator=dt,
            X_training=X_train,
            y_training=y_train,
            query_strategy=preset_batch
        )
    elif(strategy_option == 2):
        n_members = 2
        n_initial = len(X_train)
        learner_list = list()
        for member_idx in range(n_members):

            train_idx = np.random.choice(range(X_train.shape[0]), size=int(n_initial/n_members), replace=False)
            X_train_temp = X_train[train_idx]
            y_train_temp = y_train[train_idx]

            # creating a reduced copy of the data with the known instances removed
            X_train = np.delete(X_train, train_idx, axis=0)
            y_train = np.delete(y_train, train_idx)

            # initializing learner
            learnerAL = ActiveLearner(
                estimator=RandomForestClassifier(),
                X_training=X_train_temp, y_training=y_train_temp
            )
            learner_list.append(learnerAL)

        # assembling the committee
        learner = Committee(learner_list=learner_list)

    batch_data_continous_X = np.vstack([batch_data_continous_X, X_train])
    batch_data_continous_y = np.append(batch_data_continous_y, y_train)



    # Initial training with trainset
    print("========================")
    print("Initial Training Results")
    # Isolate the data we'll need for plotting.
    predictions = learner.predict(X_test)
    is_correct = (predictions == y_test)

    model_accuracy = learner.score(X_test, y_test)
    predictions = learner.predict(X_test)
    print("========================")
    print('Accuracy after query {n}: {acc:0.4f}'.format(n=0, acc=model_accuracy))
    print(confusion_matrix(y_test, predictions))
    print(classification_report(y_test, predictions))
    print("AUC: "+str(roc_auc_score(y_test, predictions)))
    accuracy_history.append(model_accuracy)
    auc_history.append(roc_auc_score(y_test, predictions))
    f1_minority_history.append(f1_score(y_test, predictions, average=None)[-1])
    recal_values = recall_score(y_test, predictions, average=None)
    recal_GM_history.append(math.sqrt(recal_values[0]*recal_values[-1]))



    N_QUERIES = int(N_RAW_SAMPLES // BATCH_SIZE)
    print("========================")
    print("Learning phase with "+ str(N_QUERIES)+" queries")
    print("Total number of records per batch "+ str(BATCH_SIZE)+"  divided into")
    if(outlier_option == 4):
        print("Uncertain samples: " + str(int(BATCH_SIZE)))
        print("outlier samples: 0")
    else:
        print("Uncertain samples: " + str(int(BATCH_SIZE * 0.7)))
        print("outlier samples: " + str(int(BATCH_SIZE * 0.3)))

    print("========================")
    for index in range(N_QUERIES):

        print("---------------------------------------")
        print("Learning phase Query: " + str(index + 1))

        print("Getting pool of data for learner using selected sampling method")

        if(strategy_option == 1):
            query_index, query_instance = learner.query(X_pool)
            # Teach our ActiveLearner model the record it has requested.
            X_batch, y_batch = X_pool[query_index], y_pool[query_index]

            # Remove the queried instance from the unlabeled pool.
            X_pool = np.delete(X_pool, query_index, axis=0)
            y_pool = np.delete(y_pool, query_index)

        elif (strategy_option == 2):

            if(outlier_option == 4):
                strategy_batch_size = int(BATCH_SIZE)
            else:
                strategy_batch_size = int(BATCH_SIZE*0.7)

            X_batch = np.array([]).reshape(0,X_pool.shape[1])
            y_batch = np.array([]).reshape(0,1)
            for strategy_batch_size_i in range(strategy_batch_size):
                query_index, query_instance = learner.query(X_pool)

                # Teach our ActiveLearner model the record it has requested.
                X_batch = np.vstack([X_batch, X_pool[query_index]])
                y_batch = np.append(y_batch, y_pool[query_index])

                # Remove the queried instance from the unlabeled pool.
                X_pool = np.delete(X_pool, query_index, axis=0)
                y_pool = np.delete(y_pool, query_index)


        print("getting Outliers from data pool")
        if(outlier_option == 1):
            outlier_index = Lof(X_pool, int(BATCH_SIZE*0.3))
        elif(outlier_option == 2):
            outlier_index = OneClassSVMOutlier(X_pool, int(BATCH_SIZE*0.3))
        elif(outlier_option == 3):
            outlier_index = IForest(X_pool, int(BATCH_SIZE * 0.3))
        else:
            outlier_index = []


        # Teach our ActiveLearner model the outlier record it has requested.
        X_outlier, y_outlier = X_pool[outlier_index], y_pool[outlier_index]

        # Remove the outlier instance from the unlabeled pool.
        X_pool = np.delete(X_pool, outlier_index, axis=0)
        y_pool = np.delete(y_pool, outlier_index)


        X_batch_l = X_batch.tolist()
        y_batch_l = y_batch.tolist()

        X_outlier_l =  X_outlier.tolist()
        y_outlier_l = y_outlier.tolist()

        X_batch_l.extend(X_outlier_l)
        y_batch_l.extend(y_outlier_l)

        X = np.array(X_batch_l)
        y = np.array(y_batch_l)

        batch_data_continous_X = np.vstack([batch_data_continous_X, X])
        batch_data_continous_y = np.append(batch_data_continous_y, y)


        teach_X = []
        teach_y = []
        print("Data Sampling with selected option")
        if(sampling_option == 1):
            oversample = SMOTE(k_neighbors=2)
            teach_X, teach_y = oversample.fit_resample(batch_data_continous_X, batch_data_continous_y)
        elif(sampling_option == 2):
            oversample = ADASYN()
            teach_X, teach_y = oversample.fit_resample(batch_data_continous_X, batch_data_continous_y)
        elif(sampling_option == 3):
            oversample = RandomOverSampler(sampling_strategy='minority')
            teach_X, teach_y = oversample.fit_resample(batch_data_continous_X, batch_data_continous_y)
        else:
            teach_X = batch_data_continous_X
            teach_y = batch_data_continous_y

        print("teach with query using uncertainity and outlier data")
        learner.teach(X=teach_X, y=teach_y)

        # Calculate and report our model's accuracy.
        model_accuracy = learner.score(X_test, y_test)
        predictions = learner.predict(X_test)
        print("========================")
        print('Accuracy after query {n}: {acc:0.4f}'.format(n=index + 1, acc=model_accuracy))

        print(confusion_matrix(y_test, predictions))
        print(classification_report(y_test, predictions))
        print("AUC: " + str(roc_auc_score(y_test, predictions)))

        accuracy_history.append(model_accuracy)
        auc_history.append(roc_auc_score(y_test, predictions))
        f1_minority_history.append(f1_score(y_test, predictions, average=None)[-1])
        recal_values = recall_score(y_test, predictions, average=None)
        recal_GM_history.append(math.sqrt(recal_values[0] * recal_values[-1]))


    # Plot metrics
    plotGraph(accuracy_history,'Accuracy')
    plotGraph(auc_history, 'AUC')
    plotGraph(f1_minority_history, 'F1 scores for Minority class')
    plotGraph(recal_GM_history, 'Geometric means of recal values')


