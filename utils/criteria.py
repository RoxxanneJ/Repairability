import numpy as np
import utils.evaluation as evl
import utils.error_generation as eg


def criterion(X_train, X_test, y_train, y_test, crt_name, model, dataset_name):
    """
    Compute the accuracy and f1 score on the model for X_train, X_test, y_train, y_test when the criterion crt_name is
    applied to training data.

    :param X_train: (pandas dataframe) training set X
    :param X_test: (pandas dataframe) testing set X
    :param y_train: (pandas series) training y
    :param y_test: (pandas series) testing y
    :param crt_name: (string) name of the criterion to apply
    :param model: (string) name of the classification model to evaluate (cf classification.py)
    :param dataset_name: (string) name of the dataset, used to match the dataset and the model with the right
    hyper-parameters if they have been added in classification.py
    :return: (tuple of 2 float) accuracy and f1 score of the model when criteria crt_name have been applied to X_train
    """
    try:
        if crt_name == 'missing':
            accuracies, f1s = crt_missing(X_train, X_test, y_train, y_test, model, dataset_name)
        elif crt_name == 'fuzzing':
            accuracies, f1s = crt_fuzzing(X_train, X_test, y_train, y_test, model, dataset_name)
        elif crt_name == 'outlier':
            accuracies, f1s = crt_outlier(X_train, X_test, y_train, y_test, model, dataset_name)
    except ValueError as e:
        print("criterion ", crt_name, " failed\n", e)
    return accuracies, f1s


def crt_missing(X_train, X_test, y_train, y_test, model, dataset_name):
    """
    Compute the accuracy and f1 score on the model for X_train, X_test, y_train, y_test when 5% of missing values
    have been injected in X_train.

    :param X_train: (pandas dataframe) training set X
    :param X_test: (pandas dataframe) testing set X
    :param y_train: (pandas series) training y
    :param y_test: (pandas series) testing y
    :param model: (string) name of the classification model to train and evaluate
    :param dataset_name: (string) name of the dataset, used to match the dataset and the model with the right
    hyper-parameters if they have been added in classification.py
    :return: (tuple of 2 float) accuracy and f1 score of the model when
    5% of missing values have been injected in X_train
    """
    try:
        accuracy = np.nan
        f1 = np.nan
        crt_data = X_train.copy()
        crt_data = eg.missing_value_generation(crt_data, 5)
        crt_data['class'] = y_train.copy()
        crt_data.dropna(inplace=True)
        crt_y_train = crt_data['class'].copy()
        crt_data.drop(columns=['class'], inplace=True)
        accuracy, f1 = evl.ml(model, crt_data, X_test, crt_y_train, y_test, dataset_name)
    except ValueError as e:
        print("criterion missing failed", e)
    return accuracy, f1


def crt_outlier(X_train, X_test, y_train, y_test, model, dataset_name):
    """
    Compute the accuracy and f1 score on the model for X_train, X_test, y_train, y_test when 5% of outliers
    have been injected in X_train.

    :param X_train: (pandas dataframe) training set X
    :param X_test: (pandas dataframe) testing set X
    :param y_train: (pandas series) training y
    :param y_test: (pandas series) testing y
    :param model: (string) name of the classification model to train and evaluate
    :param dataset_name: (string) name of the dataset, used to match the dataset and the model with the right
    hyper-parameters if they have been added in classification.py
    :return: (tuple of 2 float) accuracy and f1 score of the model when
    5% of outliers have been injected in X_train
    """
    try:
        accuracy = np.nan
        f1 = np.nan
        crt_X_train = X_train.copy()
        crt_y_train = y_train.copy()
        crt_X_train = eg.outlier_generation(crt_X_train, 5)
        accuracy, f1 = evl.ml(model, crt_X_train, X_test, crt_y_train, y_test, dataset_name)
    except ValueError as e:
        print("criterion outlier failed", e)
    return accuracy, f1


def crt_fuzzing(X_train, X_test, y_train, y_test, model, dataset_name):
    """
    Compute the accuracy and f1 score on the model for X_train, X_test, y_train, y_test when 5% of fuzzing
    have been injected in X_train.

    :param X_train: (pandas dataframe) training set X
    :param X_test: (pandas dataframe) testing set X
    :param y_train: (pandas series) training y
    :param y_test: (pandas series) testing y
    :param model: (string) name of the classification model to train and evaluate
    :param dataset_name: (string) name of the dataset, used to match the dataset and the model with the right
    hyper-parameters if they have been added in classification.py
    :return: (tuple of 2 float) accuracy and f1 score of the model when
    5% of fuzzing have been injected in X_train
    """
    try:
        accuracy = np.nan
        f1 = np.nan
        crt_X_train = X_train.copy()
        crt_X_train['class'] = y_train.copy()
        crt_X_train = eg.fuzzing_generation(crt_X_train, 5)
        crt_y_train = crt_X_train['class'].copy()
        crt_X_train.drop(columns=['class'], inplace=True)
        accuracy, f1 = evl.ml(model, crt_X_train, X_test, crt_y_train, y_test, dataset_name)
    except ValueError as e:
        print("criterion fuzzing failed", e)
    return accuracy, f1
