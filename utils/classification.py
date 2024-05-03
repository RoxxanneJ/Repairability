import os
import sys
import warnings
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.metrics import f1_score, accuracy_score
from sklearn.gaussian_process.kernels import RBF, RationalQuadratic


# List of all the classification models (used when we want to evaluate on all models)
Classification = ['logistic regression', 'knn', 'decision tree', 'random forest', 'ada boost', 'naive bayes', 'xgboost',
                  'svc', 'gaussian process', 'mlp', 'sgd', 'gradient boosting']

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = 'ignore::UserWarning, ignore::RuntimeWarning, ignore::FutureWarning'


def LogisticReg(X_train, X_test, y_train, y_test, dataset_name):
    """
    Train the logistic regression classifier model from the scikit learn library with the hyper-parameters corresponding
    to dataset_name if they are implemented or with params = {'solver': 'liblinear'} otherwise.
    Compute and return the accuracy and f1 score.

    :param X_train: (pandas dataframe) training set X
    :param X_test: (pandas dataframe) testing set X
    :param y_train: (pandas series) training y
    :param y_test: (pandas series) testing y
    :param dataset_name: (string) name of the dataset used to match the dataset and the model with the right
    hyper-parameters if they have been implemented
    :return: (tuple of 2 float) accuracy and f1 score
    """

    params = {}
    # match dataset with the right hyper-parameters
    if dataset_name == 'iris':
        params = {'penalty': 'l1', 'solver': 'saga'}
    elif dataset_name == 'adult':
        params = {'solver': 'liblinear'}
    elif dataset_name == 'wine':
        params = {'C': 1000, 'solver': 'newton-cg'}
    elif dataset_name == 'cancer':
        params = {'C': 100, 'penalty': 'none', 'solver': 'saga'}
    elif dataset_name == 'mnist':
        params = {'C': 0.1, 'penalty': 'l1', 'solver': 'saga'}
    elif dataset_name in ['fashion_mnist', 'water']:
        params = {'C': 0.01, 'solver': 'newton-cg'}
    elif dataset_name in ['spambase', 'abalone']:
        params = {'penalty': 'l1', 'solver': 'liblinear'}
    elif dataset_name in ['heart', 'statlog']:
        params = {'C': 0.1, 'solver': 'newton-cg'}
    elif dataset_name == 'bean':
        params = {'C': 10.0, 'solver': 'newton-cg'}

    clf = LogisticRegression().set_params(**params)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    f1 = f1_score(y_test, y_pred, average='macro')
    accuracy = accuracy_score(y_test, y_pred, normalize=True)
    return accuracy, f1


def KNN(X_train, X_test, y_train, y_test, dataset_name):
    """
    Train the k-nearest neighbors classifier model from the scikit learn library with the hyper-parameters corresponding
    to dataset_name if they are implemented or with the default parameters from the scikit learn library otherwise.
    Compute and return the accuracy and f1 score.

    :param X_train: (pandas dataframe) training set X
    :param X_test: (pandas dataframe) testing set X
    :param y_train: (pandas series) training y
    :param y_test: (pandas series) testing y
    :param dataset_name: (string) name of the dataset used to match the dataset and the model with the right
    hyper-parameters if they have been implemented
    :return: (tuple of 2 float) accuracy and f1 score
    """

    params = {}
    # match dataset with the right hyper-parameters
    if dataset_name in ['wine', 'bean']:
        params = {'metric': 'manhattan', 'weights': 'distance'}
    elif dataset_name == 'fashion_mnist':
        params = {'metric': 'manhattan', 'n_neighbors': 1}
    elif dataset_name == 'spambase':
        params = {'metric': 'manhattan', 'n_neighbors': 15, 'weights': 'distance'}
    elif dataset_name == 'heart':
        params = {'metric': 'manhattan', 'n_neighbors': 55}
    elif dataset_name == 'abalone':
        params = {'n_neighbors': 25}
    elif dataset_name == 'statlog':
        params = {'metric': 'manhattan', 'n_neighbors': 35, 'weights': 'distance'}
    elif dataset_name == 'water':
        params = {'metric': 'manhattan', 'n_neighbors': 66, 'weights': 'distance'}
    neigh = KNeighborsClassifier().set_params(**params)
    neigh.fit(X_train, y_train)
    y_pred = neigh.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred, normalize=True)
    f1 = f1_score(y_test, y_pred, average='macro')
    return accuracy, f1


def DecisionTree(X_train, X_test, y_train, y_test, dataset_name):
    """
    Train the decision tree classifier model from the scikit learn library with the hyper-parameters corresponding
    to dataset_name if they are implemented or with the parameters {'criterion': 'gini', 'max_depth': 4} otherwise.
    Compute and return the accuracy and f1 score.

    :param X_train: (pandas dataframe) training set X
    :param X_test: (pandas dataframe) testing set X
    :param y_train: (pandas series) training y
    :param y_test: (pandas series) testing y
    :param dataset_name: (string) name of the dataset used to match the dataset and the model with the right
    hyper-parameters if they have been implemented
    :return: (tuple of 2 float) accuracy and f1 score
    """

    params = {'max_depth': 4}
    # match dataset with the right hyper-parameters
    if dataset_name == 'cancer':
        params = {'criterion': 'entropy', 'max_depth': 4}
    elif dataset_name == 'mnist':
        params = {'criterion': 'entropy', 'max_depth': 34}
    elif dataset_name == 'fashion_mnist':
        params = {'criterion': 'entropy', 'max_depth': 42}
    elif dataset_name == 'spambase':
        params = {'criterion': 'entropy', 'max_depth': 45}
    elif dataset_name == 'heart':
        params = {'criterion': 'entropy', 'max_depth': 85}
    elif dataset_name in ['abalone', 'statlog']:
        params = {'criterion': 'entropy', 'max_depth': 5}
    elif dataset_name == 'bean':
        params = {'max_depth': 15}
    elif dataset_name == 'water':
        params = {'max_depth': 5}
    clf_tree = tree.DecisionTreeClassifier().set_params(**params)
    clf_tree.fit(X_train, y_train)
    y_pred = clf_tree.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred, normalize=True)
    f1 = f1_score(y_test, y_pred, average='macro')
    return accuracy, f1


def RandomForest(X_train, X_test, y_train, y_test, dataset_name):
    """
    Train the random forest classifier model from the scikit learn library with the hyper-parameters corresponding
    to dataset_name if they are implemented or with params = {'max_depth': 2} otherwise.
    Compute and return the accuracy and f1 score.

    :param X_train: (pandas dataframe) training set X
    :param X_test: (pandas dataframe) testing set X
    :param y_train: (pandas series) training y
    :param y_test: (pandas series) testing y
    :param dataset_name: (string) name of the dataset used to match the dataset and the model with the right
    hyper-parameters if they have been implemented
    :return: (tuple of 2 float) accuracy and f1 score
    """

    params = {}
    # match dataset with the right hyper-parameters
    if dataset_name == 'iris':
        params = {'max_depth': 3, 'n_estimators': 600, 'max_features': 'auto'}
    elif dataset_name == 'wine':
        params = {'max_depth': 3, 'n_estimators': 400, 'max_features': 'log2'}
    elif dataset_name == 'cancer':
        params = {'max_depth': 6, 'n_estimators': 200, 'criterion': 'entropy'}
    elif dataset_name == 'mnist':
        params = {'max_depth': 9, 'n_estimators': 800, 'max_features': 'log2', 'criterion': 'entropy'}
    elif dataset_name == 'fashion_mnist':
        params = {'max_depth': 20, 'n_estimators': 200, 'max_features': 'log2'}
    elif dataset_name == 'cov':
        params = {'criterion': 'entropy', 'max_depth': 90, 'n_estimators': 700}
    elif dataset_name == 'heart':
        params = {'criterion': 'entropy', 'max_depth': 90}
    elif dataset_name in ['spambase', 'bean']:
        params = {'max_depth': 30, 'n_estimators': 300}
    elif dataset_name == "abalone":
        params = {'max_depth': 10, 'n_estimators': 900}
    elif dataset_name == ['statlog', 'water']:
        params = {'max_depth': 10, 'max_features': 'log2', 'n_estimators': 700}

    clf_forest = RandomForestClassifier().set_params(**params)
    clf_forest.fit(X_train, y_train)
    y_pred = clf_forest.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred, normalize=True)
    f1 = f1_score(y_test, y_pred, average='macro')
    return accuracy, f1


def AdaBoost(X_train, X_test, y_train, y_test, dataset_name):
    """
    Train the ada boost classifier model from the scikit learn library with the hyper-parameters corresponding
    to dataset_name if they are implemented or with params = {'n_estimators': 100} otherwise.
    Compute and return the accuracy and f1 score.

    :param X_train: (pandas dataframe) training set X
    :param X_test: (pandas dataframe) testing set X
    :param y_train: (pandas series) training y
    :param y_test: (pandas series) testing y
    :param dataset_name: (string) name of the dataset used to match the dataset and the model with the right
    hyper-parameters if they have been implemented
    :return: (tuple of 2 float) accuracy and f1 score
    """

    params = {}
    svc = svm.SVC(probability=True, kernel='linear')
    # match dataset with the right hyper-parameters
    if dataset_name in ['iris', 'bean']:
        params = {'n_estimators': 100}
    elif dataset_name == 'wine':
        params = {'learning_rate': 0.01, 'n_estimators': 200, 'base_estimator': svc}
    elif dataset_name == 'cancer':
        params = {'learning_rate': 0.1, 'n_estimators': 600}
    elif dataset_name == 'mnist':
        params = {'learning_rate': 0.1, 'n_estimators': 400, 'base_estimator': svc}
    elif dataset_name == 'fashion_mnist':
        params = {'learning_rate': 0.1, 'n_estimators': 850}
    elif dataset_name == 'spambase':
        params = {'n_estimators': 700, 'learning_rate': 0.1}
    elif dataset_name == 'heart':
        params = {'learning_rate': 0.01, 'n_estimators': 700}
    elif dataset_name == 'abalone':
        params = {'learning_rate': 0.1, 'n_estimators': 700}
    elif dataset_name == 'statlog':
        params = {'learning_rate': 0.1, 'n_estimators': 500}
    elif dataset_name == 'water':
        params = {'n_estimators': 600}
    clf_ada = AdaBoostClassifier().set_params(**params)
    clf_ada.fit(X_train, y_train)
    y_pred = clf_ada.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred, normalize=True)
    f1 = f1_score(y_test, y_pred, average='macro')
    return accuracy, f1


def NaiveBayes(X_train, X_test, y_train, y_test, dataset_name):
    """
    Train the naive bayes classifier model from the scikit learn library with the hyper-parameters corresponding
    to dataset_name if they are implemented or with params = {'var_smoothing': 1e-11} otherwise.
    Compute and return the accuracy and f1 score.

    :param X_train: (pandas dataframe) training set X
    :param X_test: (pandas dataframe) testing set X
    :param y_train: (pandas series) training y
    :param y_test: (pandas series) testing y
    :param dataset_name: (string) name of the dataset used to match the dataset and the model with the right
    hyper-parameters if they have been implemented
    :return: (tuple of 2 float) accuracy and f1 score
    """

    params = {}
    # match dataset with the right hyper-parameters
    if dataset_name == 'wine':
        params = {'var_smoothing': 1e-5}
    elif dataset_name in ['iris', 'adult']:
        params = {'var_smoothing': 1e-11}
    elif dataset_name == 'cancer':
        params = {'var_smoothing': 1e-9}
    elif dataset_name == 'fashion_mnist':
        params = {'var_smoothing': 0.02848035868435802}
    elif dataset_name in ['spambase', 'heart', 'bean']:
        params = {'var_smoothing': 1e-10}
    elif dataset_name in ['abalone', 'statlog', 'mnist']:
        params = {'var_smoothing': 0.1}
    elif dataset_name == 'water':
        params = {'var_smoothing': 1e-6}
    gnb = GaussianNB().set_params(**params)
    gnb.fit(X_train, y_train)
    y_pred = gnb.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred, normalize=True)
    f1 = f1_score(y_test, y_pred, average='macro')
    return accuracy, f1


def XGBoost(X_train, X_test, y_train, y_test):
    """
    Train the xgboost classifier model from the scikit learn library.
    Compute and return the accuracy and f1 score.

    :param X_train: (pandas dataframe) training set X
    :param X_test: (pandas dataframe) testing set X
    :param y_train: (pandas series) training y
    :param y_test: (pandas series) testing y
    :return: (tuple of 2 float) accuracy and f1 score
    """
    clf_xgb = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss').fit(X_train, y_train)
    y_pred = clf_xgb.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred, normalize=True)
    f1 = f1_score(y_test, y_pred, average='macro')
    return accuracy, f1


def SVC_base(X_train, X_test, y_train, y_test, dataset_name):
    """
    Train the svm.svc classifier model from the scikit learn library with the hyper-parameters corresponding
    to dataset_name if they are implemented or with the default parameters from the scikit learn library otherwise.
    Compute and return the accuracy and f1 score.

    :param X_train: (pandas dataframe) training set X
    :param X_test: (pandas dataframe) testing set X
    :param y_train: (pandas series) training y
    :param y_test: (pandas series) testing y
    :param dataset_name: (string) name of the dataset used to match the dataset and the model with the right
    hyper-parameters if they have been implemented
    :return: (tuple of 2 float) accuracy and f1 score
    """

    params = {}
    # match dataset with the right hyper-parameters
    if dataset_name == 'wine':
        params = {'C': 0.1, 'gamma': 1, 'kernel': 'poly'}
    elif dataset_name in ['spambase', 'heart']:
        params = {'C': 0.1, 'gamma': 0.1, 'kernel': 'linear'}
    elif dataset_name == 'abalone':
        params = {'C': 100, 'gamma': 1, 'kernel': 'poly'}
    elif dataset_name == 'bean':
        params = {'C': 0.1, 'gamma': 1, 'kernel': 'linear'}
    elif dataset_name == 'statlog':
        params = {'C': 10, 'gamma': 1, 'kernel': 'linear'}
    elif dataset_name == 'water':
        params = {'gamma': 0.01}
    clf_svc = svm.SVC().set_params(**params)
    clf_svc.fit(X_train, y_train)
    y_pred = clf_svc.predict(X_test)
    f1 = f1_score(y_test, y_pred, average='macro')
    accuracy = accuracy_score(y_test, y_pred, normalize=True)
    return accuracy, f1


def GaussianProcess(X_train, X_test, y_train, y_test, dataset_name):
    """
    Train the gaussian process classifier model from the scikit learn library with the hyper-parameters corresponding
    to dataset_name if they are implemented or with kernel = 1.0 * RBF(1.0) otherwise.
    Compute and return the accuracy and f1 score.

    :param X_train: (pandas dataframe) training set X
    :param X_test: (pandas dataframe) testing set X
    :param y_train: (pandas series) training y
    :param y_test: (pandas series) testing y
    :param dataset_name: (string) name of the dataset used to match the dataset and the model with the right
    hyper-parameters if they have been implemented
    :return: (tuple of 2 float) accuracy and f1 score
    """
    kernel = None
    # match dataset with the right hyper-parameters
    if dataset_name in ['iris', 'cancer', 'wine', 'abalone', 'bean', 'adult', 'spambase']:
        kernel = 1.0 * RBF(1.0)
    elif dataset_name == 'mnist':
        kernel = 1 * RationalQuadratic()
    elif dataset_name in ['fashion_mnist', 'heart', 'statlog', 'batiment']:
        kernel = 1 ** 2 * RationalQuadratic(alpha=1, length_scale=1)
    gpc = GaussianProcessClassifier(kernel=kernel).fit(X_train, y_train)
    y_pred = gpc.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred, normalize=True)
    f1 = f1_score(y_test, y_pred, average='macro')
    return accuracy, f1


def MLP(X_train, X_test, y_train, y_test, dataset_name):
    """
    Train the multi-layer perceptron classifier model from the scikit learn library with the hyper-parameters
    corresponding to dataset_name if they are implemented or with the default parameters from the scikit learn
    library otherwise.
    Compute and return the accuracy and f1 score.

    :param X_train: (pandas dataframe) training set X
    :param X_test: (pandas dataframe) testing set X
    :param y_train: (pandas series) training y
    :param y_test: (pandas series) testing y
    :param dataset_name: (string) name of the dataset used to match the dataset and the model with the right
    hyper-parameters if they have been implemented
    :return: (tuple of 2 float) accuracy and f1 score
    """

    params = {}
    # match dataset with the right hyper-parameters
    if dataset_name == 'wine':
        params = {'activation': 'tanh', 'alpha': 0.05, 'hidden_layer_sizes': (50, 50, 50), 'learning_rate': 'adaptive'}
    elif dataset_name == 'fashion_mnist':
        params = {'activation': 'tanh', 'learning_rate': 'adaptive'}
    elif dataset_name == 'spambase':
        params = {'activation': 'tanh'}
    elif dataset_name == 'heart':
        params = {'learning_rate': 'adaptive'}
    elif dataset_name in ['abalone', 'bean']:
        params = {'alpha': 0.05, 'hidden_layer_sizes': (50, 50, 50)}
    elif dataset_name == 'statlog':
        params = {'activation': 'tanh', 'alpha': 0.01}
    elif dataset_name == 'water':
        params = {'activation': 'tanh', 'hidden_layer_sizes': (10, 30, 10)}
    mlp = MLPClassifier().set_params(**params)
    mlp.fit(X_train, y_train)
    y_pred = mlp.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred, normalize=True)
    f1 = f1_score(y_test, y_pred, average='macro')
    return accuracy, f1


def SGD(X_train, X_test, y_train, y_test, dataset_name):
    """
    Train the sgd classifier model from the scikit learn library with the hyper-parameters corresponding
    to dataset_name if they are implemented or params = {'alpha': 0.1, 'epsilon': 1e-05, 'eta0': 0.0,
    'fit_intercept': True, 'l1_ratio': 0.15, 'learning_rate': 'optimal', 'loss': 'log', 'max_iter': 1000,
    'n_iter_no_change': 5, 'n_jobs': -1, 'penalty': 'l1', 'power_t': 0.5, 'shuffle': True, 'tol': 1e-05,
    'validation_fraction': 0.99} otherwise.
    Compute and return the accuracy and f1 score.

    :param X_train: (pandas dataframe) training set X
    :param X_test: (pandas dataframe) testing set X
    :param y_train: (pandas series) training y
    :param y_test: (pandas series) testing y
    :param dataset_name: (string) name of the dataset used to match the dataset and the model with the right
    hyper-parameters if they have been implemented
    :return: (tuple of 2 float) accuracy and f1 score
    """

    params = {}

    # match dataset with the right hyper-parameters
    if dataset_name in ['iris', 'adult']:
        params = {'alpha': 0.1, 'epsilon': 1e-05, 'loss': 'log', 'penalty': 'l1', 'tol': 1e-05}
    elif dataset_name == 'wine':
        params = {'alpha': 10, 'epsilon': 1e-05, 'loss': 'log'}
    elif dataset_name == 'cancer':
        params = {'alpha': 100, 'epsilon': 0.01, 'loss': 'modified_huber'}
    elif dataset_name == 'mnist':
        params = {'loss': 'modified_huber', 'penalty': 'l1'}
    elif dataset_name == 'fashion_mnist':
        params = {'penalty': 'l1', 'tol': 100000}
    elif dataset_name == 'spambase':
        params = {'alpha': 0.01, 'epsilon': 0.01, 'loss': 'modified_huber', 'penalty': 'l1', 'tol': 1e-05}
    elif dataset_name == 'heart':
        params = {'alpha': 10, 'epsilon': 0.01, 'loss': 'squared_hinge', 'penalty': 'elasticnet', 'tol': 1000}
    elif dataset_name == 'abalone':
        params = {'alpha': 0.01, 'epsilon': 0.01, 'loss': 'modified_huber', 'tol': 1000}
    elif dataset_name == 'bean':
        params = {'alpha': 0.01, 'epsilon': 0.001, 'penalty': 'l1'}
    elif dataset_name == 'statlog':
        params = {'alpha': 0.1, 'epsilon': 0.01}
    elif dataset_name == 'water':
        params = {'alpha': 0.1, 'epsilon': 1e-05, 'loss': 'log', 'penalty': 'elasticnet', 'tol': 0.01}
    sgd = SGDClassifier().set_params(**params)
    sgd.fit(X_train, y_train)
    y_pred = sgd.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred, normalize=True)
    f1 = f1_score(y_test, y_pred, average='macro')
    return accuracy, f1


def GradientBoosting(X_train, X_test, y_train, y_test):
    """
    Train the gradient boosting classifier model from the scikit learn library.
    Compute and return the accuracy and f1 score.

    :param X_train: (pandas dataframe) training set X
    :param X_test: (pandas dataframe) testing set X
    :param y_train: (pandas series) training y
    :param y_test: (pandas series) testing y
    :return: (tuple of 2 float) accuracy and f1 score
    """
    gb = GradientBoostingClassifier().fit(X_train, y_train)
    y_pred = gb.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred, normalize=True)
    f1 = f1_score(y_test, y_pred, average='macro')
    return accuracy, f1


def classification(ml_model, X_train, X_test, y_train, y_test, dataset_name):
    """
    Call the function corresponding to the right classification model on data and returns the accuracy and f1 score.

    :param ml_model: (string) name of the classification model to be used
    :param X_train: (pandas dataframe) training set X
    :param X_test: (pandas dataframe) testing set X
    :param y_train: (pandas series) training y
    :param y_test: (pandas series) testing y
    :param dataset_name: (string) name of the dataset used to match the dataset and the model with the right
    hyper-parameters if they have been implemented
    :return: (tuple of 2 float) accuracy and f1 score
    """

    try:
        acc = np.nan  # if ml cannot converge the accuracy is 0
        f1 = np.nan  # if ml cannot converge the F1 score is 0
        if ml_model == 'logistic regression':
            acc, f1 = LogisticReg(X_train, X_test, y_train, y_test, dataset_name)
        elif ml_model == 'knn':
            acc, f1 = KNN(X_train, X_test, y_train, y_test, dataset_name)
        elif ml_model == 'decision tree':
            acc, f1 = DecisionTree(X_train, X_test, y_train, y_test, dataset_name)
        elif ml_model == 'random forest':
            acc, f1 = RandomForest(X_train, X_test, y_train, y_test, dataset_name)
        elif ml_model == 'ada boost':
            acc, f1 = AdaBoost(X_train, X_test, y_train, y_test, dataset_name)
        elif ml_model == 'naive bayes':
            acc, f1 = NaiveBayes(X_train, X_test, y_train, y_test, dataset_name)
        elif ml_model == 'xgboost':
            acc, f1 = XGBoost(X_train, X_test, y_train, y_test)
        elif ml_model == 'svc':
            acc, f1 = SVC_base(X_train, X_test, y_train, y_test, dataset_name)
        elif ml_model == 'gaussian process':
            acc, f1 = GaussianProcess(X_train, X_test, y_train, y_test, dataset_name)
        elif ml_model == 'mlp':
            acc, f1 = MLP(X_train, X_test, y_train, y_test, dataset_name)
        elif ml_model == 'sgd':
            acc, f1 = SGD(X_train, X_test, y_train, y_test, dataset_name)
        elif ml_model == 'gradient boosting':
            acc, f1 = GradientBoosting(X_train, X_test, y_train, y_test)
    except ValueError as e:
        print(e)
    return acc, f1
