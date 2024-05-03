import numpy as np
import pandas as pd
import utils.evaluation as ev
import utils.apply_criteria as ac
import utils.split as sp
import time as t
import random as rd
from multiprocessing import cpu_count
from itertools import repeat
from multiprocessing import get_context


def comp_qa(base_accs, var_accs, nb_classes):
    """
    Compute the parameters qa1 and qa2 of the metric. See conference paper for definition.
    :param base_accs: (np.array of floats) of shape (n, m) or (m) accuracies computed for n resamplings over m models of
    classification
    :param var_accs: (np.array of floats) of shape (n, m, e) or (m, e) accuracy when 5% of error is introduced in
    training data for n resamplings over m models of classification and e errors
    :param nb_classes: (int) number of classes in data
    :return: (float, float) qa1 and qa2
    """
    try:
        # we compute qa1
        lim = 1 / nb_classes  # value for which the accuracy is the same as a random guess
        base_accs = np.nan_to_num(base_accs)  # if the classification model didn't converge accuracy is 0 by default
        if len(base_accs.shape) == 2:
            base_accs = np.nanmean(base_accs, axis=0)
        base_acc = np.nanmean(base_accs, axis=0)
        qa1 = 1 - ((base_acc - lim) * (base_acc > lim) / (1 - lim))

        # we compute qa2
        var_accs = np.nan_to_num(var_accs)
        if len(var_accs.shape) == 3:
            var_accs = np.nanmean(var_accs, axis=0)
        var_acc = np.nanmean(var_accs, axis=0)
        qa2 = np.nanmean(np.array([min(abs(base_acc - v) / 0.1, 1) for v in var_acc]))
    except ValueError as e:
        print("computation of qa failed: ", e)
    return qa1, qa2


def comp_qf(base_f1s, var_f1s, nb_classes):
    """
    Compute the parameters qf1 and qf2 of the metric. See conference paper for definition.
    :param base_f1s: (np.array of floats) of shape (n, m) or (m) f1 scores computed for n resamplings over m models of
    classification
    :param var_f1s: (np.array of floats) of shape (n, m, e) or (m, e) f1 score when 5% of error is introduced in
    training data for n resamplings over m models of classification and e errors
    :param nb_classes: (int) number of classes in data
    :return: (float, float) qf1 and qf2
    """
    try:
        # we compute qf1
        lim = 1 / nb_classes  # value for which the f1 score is the same as a best f1 score of a random guess
        base_f1s = np.nan_to_num(base_f1s)  # if the classification model didn't converge f1 score is 0 by default
        if len(base_f1s.shape) == 2:
            base_f1s = np.nanmean(base_f1s, axis=0)
        base_f1 = np.nanmean(base_f1s, axis=0)
        qf1 = 1 - ((base_f1 - lim) * (base_f1 > lim) / (1 - lim))

        # we compute qf2
        var_f1s = np.nan_to_num(var_f1s)
        if len(var_f1s.shape) == 3:
            var_f1s = np.nanmean(var_f1s, axis=0)
        var_f1 = np.nanmean(var_f1s, axis=0)
        qf2 = np.nanmean(np.array([min(abs(base_f1 - v) / 0.1, 1) for v in var_f1]))
    except ValueError as e:
        print("computation of y failed: ", e)
    return qf1, qf2


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


def dq_metric(df, crt_names, models, data_name, save_name, nb_iter=30):
    """
    Non-parallel implementation of computing the data quality metric when there is no dedicated test data
    (we use 30 resamplings of train test).
    results are saved in:
        -output/variations  -> a file with the accuracies when 5% of errors is injected in data over 30 resamplings,
                                all classification models in models, and over all errors in crt_names
                            -> a file with the f1 scores when 5% of errors is injected in data over 30 resamplings,
                                all classification models in models, and over all errors in crt_names
        -output/scores      -> a file with the tuple (x,qa,qa1,qa2,qf,qf1,qf2,time) with time being the execution time
                                and the parameters for the metric as defined in the paper
        -output/base_scores -> a file with the accuracies evaluated over 30 resamplings and for all classification
                                models in models
                            -> a file with the f1 scores evaluated over 30 resamplings and for all classification
                                models in models
    :param df: (pandas dataframe) data to evaluate the target for classification must be named 'class'
    :param crt_names: (list of string) list of the names of the errors we want to evaluate accuracy and f1 score
    variations for (cf paper) (implemented in criteria.py)
    :param models: (list of string) list of the names of the models we want to evaluate data on
    :param data_name: (string) name of dataset to evaluate this is used both to name the saved file and match with the
    right hyper-parameters in classification.py if they are defined
    :param save_name: (string) name of the dataset used to save the results file
    :param nb_iter: (int) default is 30 (cf paper) number of iterations for resampling
    :return: (int) 0 if execution terminate properly
    """
    try:
        start = t.time()
        nb_classes = df['class'].nunique()
        var_accs = np.zeros((nb_iter, len(models), len(crt_names)))
        var_f1s = np.zeros((nb_iter, len(models), len(crt_names)))
        base_accs = np.zeros((nb_iter, len(models)))
        base_f1s = np.zeros((nb_iter, len(models)))
        for k in range(nb_iter):
            df.dropna(inplace=True)
            rd.seed(t.time())
            X_train, X_test, y_train, y_test = sp.sampling(df, 0.2)

            base_accs[k], base_f1s[k] = base_scores(X_train, X_test, y_train, y_test, models, data_name)

            var_accs[k], var_f1s[k] = ac.apply_crt(X_train, X_test, y_train, y_test, crt_names, models, data_name)

        qa1, qa2 = comp_qa(np.copy(base_accs), np.copy(var_accs), nb_classes)
        qf1, qf2 = comp_qf(np.copy(base_f1s), np.copy(var_f1s), nb_classes)
        qa = max(qa1, qa2)
        qf = max(qf1, qf2)
        stop = t.time()

        np.save("output/variations/" + save_name + "_var_accs.npy", var_accs)
        np.save("output/variations/" + save_name + "_var_f1s.npy", var_f1s)
        np.save("output/base_scores/" + save_name + "_base_accs.npy", base_accs)
        np.save("output/base_scores/" + save_name + "_base_f1s.npy", base_f1s)
        np.save("output/scores/" + save_name + "_(qa,qf,time).npy",
                ((qa, qa1, qa2), (qf, qf1, qf2), stop-start))

        print("\n**********\n", save_name, "\nqa=", qa, "with qa1=", qa1, " and qa2=", qa2,
              "\nqf=", qf, "with qf1=", qf1, " and qf2=", qf2)
    except ValueError as e:
        print("computation of the data quality metric failed: ", e)
    return 0


def dq_metric_one_iter(k, df, crt_names, models, data_name):
    """
    Implementation of computing the accuracies, f1 scores for data, and for data with 5% of errors on all models
    for all errors in crt_names when there is no dedicated test data. This function is meant to be used when
    computing the data quality metric parallelized for each resampling (cf the next function: dq_metric_para)
    :param k: (int) used to call the function in parallel for all the resamplings in dq_metric_para
    :param df: (pandas dataframe) data to evaluate the target for classification must be named 'class'
    :param crt_names: (list of string) list of the names of the errors we want to evaluate accuracy and f1 score
    variations for (cf paper) (implemented in criteria.py)
    :param models: (list of string) list of the names of the models we want to evaluate data on
    :param data_name: (string) name of dataset to evaluate this is used both to name the saved file and match with the
    right hyper-parameters in classification.py if they are defined
    :return: (tuple of 4 np.array of float) of shape ((m, e), (m, e), m, m) with m the number of models and e the number
    of errors.
    """
    try:
        rd.seed(t.time())
        var_acc = np.full((len(models), len(crt_names)), np.nan)
        var_f1 = np.full((len(models), len(crt_names)), np.nan)
        base_acc = np.full(len(models), np.nan)
        base_f1 = np.full(len(models), np.nan)
        X_train, X_test, y_train, y_test = sp.sampling(df, 0.2)
        base_acc, base_f1 = base_scores(X_train, X_test, y_train, y_test, models, data_name)
        var_acc, var_f1 = ac.apply_crt(X_train, X_test, y_train, y_test, crt_names, models, data_name)
    except ValueError as e:
        print("computation of the data quality metric failed: ", e)
    return var_acc, var_f1, base_acc, base_f1


def dq_metric_para(nb_iter, df, crt_names, models, data_name, save_name):
    """
    Parallel implementation of computing the data quality metric when there is no dedicated test data.
    The parallelization is done on nb_iter.
    results are saved in:
        -output/variations  -> a file with the accuracies when 5% of errors is injected in data over:
                                nb_iter resamplings, all classification models in models, and all errors in crt_names
                            -> a file with the f1 scores when 5% of errors is injected in data over:
                                nb_iter resamplings, all classification models in models, and all errors in crt_names
        -output/scores      -> a file with the tuple (x,qa,qa1,qa2,qf,qf1,qf2,time) with time being the execution time
                                and the parameters for the metric as defined in the paper
        -output/base_scores -> a file with the accuracies evaluated over nb_iter resamplings and for all classification
                                models in models
                            -> a file with the f1 scores evaluated over nb_iter resamplings and for all classification
                                models in models
    :param nb_iter: (int) number of resamplings used to minimize the influence of the choice of training and test on
    accuracies and f1 scores (cf paper) this is also what is paralleled
    :param df: (pandas dataframe) data to evaluate the target for classification must be named 'class'
    :param crt_names: (list of string) list of the names of the errors we want to evaluate accuracy and f1 score
    variations for (cf paper) (implemented in criteria.py)
    :param models: (list of string) list of the names of the models we want to evaluate data on
    :param data_name: (string) name of dataset to evaluate this is used both to name the saved file and match with the
    right hyper-parameters in classification.py if they are defined
    :param save_name: (string) name of the dataset used to save the results file
    :return: (int) 0 if execution terminate properly
    """
    try:
        start = t.time()
        nb_classes = df['class'].nunique()
        cpus = cpu_count()  # we use all cpus
        with get_context("spawn").Pool(cpus) as pool:
            var_accs, var_f1s, base_accs, base_f1s = \
                zip(*pool.starmap(dq_metric_one_iter, zip(list(range(nb_iter)), repeat(df), repeat(crt_names),
                                                          repeat(models), repeat(data_name))))
        qa1, qa2 = comp_qa(np.copy(base_accs), np.copy(var_accs), nb_classes)
        qf1, qf2 = comp_qf(np.copy(base_f1s), np.copy(var_f1s), nb_classes)
        qa = max(qa1, qa2)
        qf = max(qf1, qf2)
        stop = t.time()

        np.save("output/variations/" + save_name + "_noTest_var_accs.npy", var_accs)
        np.save("output/variations/" + save_name + "_noTest_var_f1s.npy", var_f1s)
        np.save("output/base_scores/" + save_name + "_noTest_base_accs.npy", base_accs)
        np.save("output/base_scores/" + save_name + "_noTest_base_f1s.npy", base_f1s)
        np.save("output/scores/" + save_name + "_noTest_(qa,qf,time).npy",
                ((qa, qa1, qa2), (qf, qf1, qf2), stop - start))
        print("\n**********\n", save_name, "\nqa=", qa, "with qa1=", qa1, " and qa2=", qa2,
              "\nqf=", qf, "with qf1=", qf1, " and qf2=", qf2, "\ntime=", stop - start)
    except ValueError as e:
        print("computation of the data quality metric failed: ", e)
    return 0


def dq_metric_test(X_train, X_test, y_train, y_test, crt_names, models, data_name, save_name):
    """
    Non-parallel implementation of computing the data quality metric when there is dedicated test data.
    results are saved in:
        -output/variations  -> a file with the accuracies when 5% of errors is injected in data over all classification
                               models in models, and over all errors in crt_names
                            -> a file with the f1 scores when 5% of errors is injected in data over all classification
                               models in models, and over all errors in crt_names
        -output/scores      -> a file with the tuple (x,qa,qa1,qa2,qf,qf1,qf2,time) with time being the execution time
        and the parameters for the metric as defined in the paper
        -output/base_scores -> a file with the accuracies evaluated over all classification models in models
                            -> a file with the f1 scores evaluated over all classification models in models
    :param X_train: (pandas dataframe) training set X
    :param X_test: (pandas dataframe) testing set X
    :param y_train: (pandas series) training y
    :param y_test: (pandas series) testing y
    :param crt_names: (list of string) list of the names of the errors we want to evaluate accuracy and f1 score
    variations for (cf paper) (implemented in criteria.py)
    :param models: (list of string) list of the names of the models we want to evaluate data on
    :param data_name: (string) name of dataset to evaluate this is used both to name the saved file and match with the
    right hyper-parameters in classification.py if they are defined
    :param save_name: (string) name of the dataset used to save the results file
    :return: (int) 0 if execution terminate properly
    """
    try:
        start = t.time()
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
        var_accs, var_f1s = ac.apply_crt(X_train, X_test, y_train, y_test, crt_names, models, data_name)

        qa1, qa2 = comp_qa(np.copy(base_accs), np.copy(var_accs), nb_classes)
        qf1, qf2 = comp_qf(np.copy(base_f1s), np.copy(var_f1s), nb_classes)
        qa = max(qa1, qa2)
        qf = max(qf1, qf2)
        stop = t.time()

        np.save("output/variations/" + save_name + "_var_accs.npy", var_accs)
        np.save("output/variations/" + save_name + "_var_f1s.npy", var_f1s)
        np.save("output/base_scores/" + save_name + "_base_accs.npy", base_accs)
        np.save("output/base_scores/" + save_name + "_base_f1s.npy", base_f1s)
        np.save("output/scores/" + save_name + "_(qa,qf,time).npy",
                ((qa, qa1, qa2), (qf, qf1, qf2), stop-start))

        print("\n**********\n", save_name, "\nqa=", qa, "with qa1=", qa1, " and qa2=", qa2,
              "\nqf=", qf, "with qf1=", qf1, " and qf2=", qf2)
    except ValueError as e:
        print("computation of the data quality metric failed: ", e)
    return 0


def dq_metric_test_para(X_train, X_test, y_train, y_test, crt_names, models, data_name, save_name):
    """
    Parallel implementation (on the models) of computing the data quality metric when there is dedicated test data.
    results are saved in:
        -output/variations  -> a file with the accuracies when 5% of errors is injected in data over all classification
                               models in models, and over all errors in crt_names
                            -> a file with the f1 scores when 5% of errors is injected in data over all classification
                               models in models, and over all errors in crt_names
        -output/scores      -> a file with the tuple (x,qa,qa1,qa2,qf,qf1,qf2,time) with time being the execution time
        and the parameters for the metric as defined in the paper
        -output/base_scores -> a file with the accuracies evaluated over all classification models in models
                            -> a file with the f1 scores evaluated over all classification models in models
    :param X_train: (pandas dataframe) training set X
    :param X_test: (pandas dataframe) testing set X
    :param y_train: (pandas series) training y
    :param y_test: (pandas series) testing y
    :param crt_names: (list of string) list of the names of the errors we want to evaluate accuracy and f1 score
    variations for (cf paper) (implemented in criteria.py)
    :param models: (list of string) list of the names of the models we want to evaluate data on
    :param data_name: (string) name of dataset to evaluate this is used both to name the saved file and match with the
    right hyper-parameters in classification.py if they are defined
    :param save_name: (string) name of the dataset used to save the results file
    :return: (int) 0 if execution terminate properly
    """
    try:
        start = t.time()
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
            var_accs, var_f1s = zip(*pool.starmap(ac.apply_crt_one_model, zip(models, repeat(X_train), repeat(X_test),
                                                                              repeat(y_train), repeat(y_test),
                                                                              repeat(crt_names), repeat(data_name))))
        qa1, qa2 = comp_qa(np.copy(base_accs), np.copy(var_accs), nb_classes)
        qf1, qf2 = comp_qf(np.copy(base_f1s), np.copy(var_f1s), nb_classes)
        qa = max(qa1, qa2)
        qf = max(qf1, qf2)
        stop = t.time()
        pool.close()
        pool.join()

        np.save("output/variations/" + save_name + "_var_accs.npy", var_accs)
        np.save("output/variations/" + save_name + "_var_f1s.npy", var_f1s)
        np.save("output/base_scores/" + save_name + "_base_accs.npy", base_accs)
        np.save("output/base_scores/" + save_name + "_base_f1s.npy", base_f1s)
        np.save("output/scores/" + save_name + "_(qa,qf,time).npy",
                ((qa, qa1, qa2), (qf, qf1, qf2), stop-start))

        print("\n**********\n", save_name, "\nqa=", qa, "with qa1=", qa1, " and qa2=", qa2,
              "\nqf=", qf, "with qf1=", qf1, " and qf2=", qf2, "\ntime=", stop-start)
    except ValueError as e:
        print("computation of the data quality metric failed: ", e)
    return 0
