import numpy as np
import utils.criteria as crt
import utils.classification as cl


def apply_crt(X_train, X_test, y_train, y_test, crt_names, models, dataset_name):
    """
    Apply the criterion in crt_names, train and evaluate accuracies and f1 scores on the models  for X_train, X_test,
    y_train, y_test. Return a tuple with 2 np.array with the corresponding accuracies and f1 scores.

    :param X_train: (pandas dataframe) training set X
    :param X_test: (pandas dataframe) testing set X
    :param y_train: (pandas series) training y
    :param y_test: (pandas series) testing y
    :param crt_names: (list of string) names of the criterion to apply (cf criteria.py)
    :param models: (string) or (list of strings) if it's set to the string 'classification' it will be evaluated on all
    classification models (cf classification.py). Otherwise list of the names of the classification models to
    evaluate on.
    :param dataset_name: (string) name of the dataset (match with the corresponding hyper-parameters for the models if
    they exist)
    :return: (tuple of 2 np.array of float) of shape ((m, e), (m, e)) with m the number of models and e the number of
    criterion.
    """
    try:
        if models == "classification":  # apply to the list of all classification models defined in classification.py
            models = cl.Classification
        var_accuracies = np.zeros((len(models), len(crt_names)))
        var_f1scores = np.zeros((len(models), len(crt_names)))

        # for each model and criterion, we apply the criterion, train and evaluate the model
        for model in range(len(models)):
            for crt_name in range(len(crt_names)):
                var_accuracies[model][crt_name], var_f1scores[model][crt_name] = crt.criterion(
                    X_train, X_test, y_train, y_test, crt_names[crt_name], models[model], dataset_name)

    except ValueError as e:
        print("apply_crt failed\n", e)
    return var_accuracies, var_f1scores


def apply_crt_one_model(model, X_train, X_test, y_train, y_test, crt_names, dataset_name):
    """
    Apply the criterion in crt_names, train and evaluate accuracies and f1 scores on the model  for X_train, X_test,
    y_train, y_test. Return a tuple with 2 np.array with the corresponding accuracies and f1 scores.

    :param X_train: (pandas dataframe) training set X
    :param X_test: (pandas dataframe) testing set X
    :param y_train: (pandas series) training y
    :param y_test: (pandas series) testing y
    :param crt_names: (list of string) names of the criterion to apply (cf criteria.py)
    :param model: (string) name of the classification model to evaluate on.
    :param dataset_name: (string) name of the dataset (match with the corresponding hyper-parameters for the models if
    they exist)
    :return: (tuple of 2 np.array of float) of size e with e the number of criterion.
    """
    try:
        var_accuracies = np.zeros(len(crt_names))
        var_f1scores = np.zeros(len(crt_names))

        for crt_name in range(len(crt_names)):
            var_accuracies[crt_name], var_f1scores[crt_name] = crt.criterion(X_train, X_test, y_train, y_test,
                                                                             crt_names[crt_name], model, dataset_name)
    except ValueError as e:
        print("apply_crt failed\n", e)
    return var_accuracies, var_f1scores
