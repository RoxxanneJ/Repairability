import numpy as np
import pandas as pd
import utils.evaluation as ev
import utils.split as sp
import time as t
import random as rd
from multiprocessing import cpu_count
from itertools import repeat
from multiprocessing import get_context


def comp_q1(base_accs, base_f1s, nb_classes):
    try:
        # we compute qa1
        lim = 1 / nb_classes  # value for which the accuracy and f1 score are the same as a random guess
        base_accs = np.nan_to_num(base_accs)  # if the classification model didn't converge accuracy is 0 by default
        base_f1s = np.nan_to_num(base_f1s)  # if the classification model didn't converge f1 score is 0 by default
        if len(base_accs.shape) == 2:
            base_accs = np.nanmean(base_accs, axis=0)
        if len(base_f1s.shape) == 2:
            base_f1s = np.nanmean(base_f1s, axis=0)
        base_acc = np.nanmean(base_accs, axis=0)
        base_f1 = np.nanmean(base_f1s, axis=0)
        qa1 = 1 - ((base_acc - lim) * (base_acc > lim) / (1 - lim))
        qf1 = 1 - ((base_f1 - lim) * (base_f1 > lim) / (1 - lim))
    except ValueError as e:
        print("computation of comp_q1 failed: ", e)
    return qa1, qf1


def base_scores(X_train, X_test, y_train, y_test, models, data_name):
    """
    Compute accuracies and f1 scores with training and test for a list of classification models.
    :param X_train: (pandas dataframe) training set X
    :param X_test: (pandas dataframe) testing set X
    :param y_train: (pandas series) training y
    :param y_test: (pandas series) testing y
    :param models: (list of string) list of classification models to use (see classification.py)
    :param data_name: (string) dataset name, this is used to match it with the right hyper-parameters for each
    classification model if they are known (other wise default are used see classification.py)
    :return: (tuple of np.array of float) of size (m, m) with m the number of classification models the arrays in the
    tuple respectively contain accuracies and f1 scores for the m models
    """
    try:
        acc = np.zeros(len(models))
        f1 = np.zeros(len(models))
        for model in range(len(models)):
            # evaluation of each model on data
            acc[model], f1[model] = ev.ml(models[model], X_train, X_test, y_train, y_test, data_name)
    except ValueError as e:
        print("base score failed", e)
    return acc, f1


def q1(df, models, data_name, nb_iter=30):
    """
    Non-parallel implementation of computing qa1 and qf1 when there is no dedicated test data
    (we use 30 resamplings of train test).
    :param df: (pandas dataframe) data to evaluate the target for classification must be named 'class'
    :param models: (list of string) list of the names of the models we want to evaluate data on
    :param data_name: (string) name of dataset to evaluate this is used both to name the saved file and match with the
    right hyper-parameters in classification.py if they are defined
    :param nb_iter: (int) default is 30 (cf paper) number of iterations for resampling
    :return: (float, float) qa1 and qf1
    """
    try:
        nb_classes = df['class'].nunique()
        base_accs = np.zeros((nb_iter, len(models)))
        base_f1s = np.zeros((nb_iter, len(models)))
        for k in range(nb_iter):
            df.dropna(inplace=True)
            rd.seed(t.time())
            X_train, X_test, y_train, y_test = sp.sampling(df, 0.2)
            base_accs[k], base_f1s[k] = base_scores(X_train, X_test, y_train, y_test, models, data_name)
        qa1, qf1 = comp_q1(np.copy(base_accs), np.copy(base_f1s), nb_classes)
    except ValueError as e:
        print("computation of qa1 and qf1 failed: ", e)
    return qa1, qf1


def q1_one_iter(k, df, models, data_name):
    try:
        rd.seed(t.time())
        base_acc = np.full(len(models), np.nan)
        base_f1 = np.full(len(models), np.nan)
        X_train, X_test, y_train, y_test = sp.sampling(df, 0.2)
        base_acc, base_f1 = base_scores(X_train, X_test, y_train, y_test, models, data_name)
    except ValueError as e:
        print("computation of q1_one_iter failed: ", e)
    return base_acc, base_f1


def q1_para(nb_iter, df, models, data_name):
    """
    Parallel implementation of computing qa1 and qf1 when there is no dedicated test data.
    The parallelization is done on nb_iter.
    :param nb_iter: (int) number of resamplings used to minimize the influence of the choice of training and test on
    accuracies and f1 scores (cf paper) this is also what is paralleled
    :param df: (pandas dataframe) data to evaluate the target for classification must be named 'class'
    :param models: (list of string) list of the names of the models we want to evaluate data on
    :param data_name: (string) name of dataset to evaluate this is used both to name the saved file and match with the
    right hyper-parameters in classification.py if they are defined
    :return: (float, float) qa1 and qf1
    """
    try:
        nb_classes = df['class'].nunique()
        cpus = cpu_count()  # we use all cpus
        with get_context("spawn").Pool(cpus) as pool:
            base_accs, base_f1s = zip(*pool.starmap(q1_one_iter, zip(list(range(nb_iter)), repeat(df), repeat(models),
                                                                     repeat(data_name))))
        qa1, qf1 = comp_q1(np.copy(base_accs), np.copy(base_f1s), nb_classes)
    except ValueError as e:
        print("computation of qa1 and qf1 failed: ", e)
    return qa1, qf1


def q1_test(X_train, X_test, y_train, y_test, models, data_name):
    """
    Non-parallel implementation of computing qa1 and qf1 when there is dedicated test data.
    :param X_train: (pandas dataframe) training set X
    :param X_test: (pandas dataframe) testing set X
    :param y_train: (pandas series) training y
    :param y_test: (pandas series) testing y
    :param models: (list of string) list of the names of the models we want to evaluate data on
    :param data_name: (string) name of dataset to evaluate this is used both to name the saved file and match with the
    right hyper-parameters in classification.py if they are defined
    :return: (float, float) qa1 and qf1
    """
    try:
        X_train['class'] = y_train.copy()
        X_test['class'] = y_test.copy()
        df = pd.concat([X_train, X_test])
        nb_classes = df['class'].nunique()

        X_train.dropna(inplace=True)
        y_train = X_train['class'].copy()
        X_train.drop(columns=['class'], inplace=True)
        X_test.dropna(inplace=True)
        y_test = X_test['class'].copy()
        X_test.drop(columns=['class'], inplace=True)

        base_accs, base_f1s = base_scores(X_train, X_test, y_train, y_test, models, data_name)

        qa1, qf1 = comp_q1(np.copy(base_accs), np.copy(base_f1s), nb_classes)
    except ValueError as e:
        print("computation of qa1 and qf1 failed: ", e)
    return qa1, qf1


def q1_test_para(X_train, X_test, y_train, y_test, models, data_name):
    """
    Parallel implementation (on the models) of computing qa1 and qf1 when there is dedicated test data.

    :param X_train: (pandas dataframe) training set X
    :param X_test: (pandas dataframe) testing set X
    :param y_train: (pandas series) training y
    :param y_test: (pandas series) testing y
    :param models: (list of string) list of the names of the models we want to evaluate data on
    :param data_name: (string) name of dataset to evaluate this is used both to name the saved file and match with the
    right hyper-parameters in classification.py if they are defined
    :return: (float, float) qa1 and qf1
    """
    try:
        X_train['class'] = y_train.copy()
        X_test['class'] = y_test.copy()
        df = pd.concat([X_train, X_test])
        nb_classes = df['class'].nunique()

        X_train.dropna(inplace=True)
        y_train = X_train['class'].copy()
        X_train.drop(columns=['class'], inplace=True)
        X_test.dropna(inplace=True)
        y_test = X_test['class'].copy()
        X_test.drop(columns=['class'], inplace=True)

        cpus = cpu_count()  # we use all cpus
        with get_context("spawn").Pool(cpus) as pool:
            base_accs, base_f1s = zip(*pool.starmap(ev.ml, zip(models, repeat(X_train), repeat(X_test), repeat(y_train),
                                                               repeat(y_test), repeat(data_name))))
        qa1, qf1 = comp_q1(np.copy(base_accs), np.copy(base_f1s), nb_classes)
        pool.close()
        pool.join()
    except ValueError as e:
        print("computation of qa1 and qf1 failed: ", e)
    return qa1, qf1
