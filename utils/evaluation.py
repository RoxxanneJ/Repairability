import utils.classification as cl


def ml(model, X_train, X_test, y_train, y_test, dataset_name):
    """
    Compute the accuracy and f1 score of the classification model on X_train, X_test, y_train, y_test.

    :param model: (string) name of the classification model to be used
    :param X_train: (pandas dataframe) training set X
    :param X_test: (pandas dataframe) testing set X
    :param y_train: (pandas series) training y
    :param y_test: (pandas series) testing y
    :param dataset_name: (string) name of the dataset used to match the dataset and the model with the right
    hyper-parameters if they have been implemented
    :return: (tuple of 2 float) accuracy and f1 score
    """

    try:
        if model in cl.Classification:
            acc, f1 = cl.classification(model, X_train, X_test, y_train, y_test, dataset_name)
    except ValueError as e:
        print("ml ", model, " failed\n", e)
    return acc, f1
