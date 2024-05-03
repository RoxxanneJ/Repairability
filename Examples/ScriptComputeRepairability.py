import numpy as np
import pandas as pd
import utils.Computeqa1Andqf1 as q1
import repairability as rp
from multiprocessing import freeze_support

models = ['logistic regression', 'knn', 'decision tree', 'random forest', 'ada boost', 'naive bayes', 'xgboost',
          'svc', 'gaussian process', 'mlp', 'sgd', 'gradient boosting']

if __name__ == '__main__':
    freeze_support()
    # We load the trusted test data (no errors were injected)
    X_test = pd.read_csv("../Datasets/iris/iris_test.csv")
    y_test = X_test['class'].copy()
    X_test.drop(columns=['class'], inplace=True)

    # We load the artificially deteriorated training data
    X_train = pd.read_csv("../Datasets/iris/missing_outlier/iris_train_missing_outlier_10.csv")
    X_train.dropna(inplace=True)
    y_train = X_train['class'].copy()
    X_train.drop(columns=['class'], inplace=True)

    # We load the repaired dataset that was obtained as shown in the example script
    # "ScriptRepairMissingValuesAndOutliers.py"
    X_train_repaired = pd.read_csv("../Examples/RepairedDataset/iris_missing_outlier_10_repaired.csv")
    y_train_repaired = X_train_repaired['class'].copy()
    X_train_repaired.drop(columns=['class'], inplace=True)

    # We compute 1-qa1 and 1-qf1 for both train data
    unrepaired_qa1, unrepaired_qf1 = q1.q1_test_para(X_train, X_test, y_train, y_test, models, "iris")
    unrepaired_qa1, unrepaired_qf1 = 1 - unrepaired_qa1, 1 - unrepaired_qf1
    repaired_qa1, repaired_qf1 = q1.q1_test_para(X_train_repaired, X_test, y_train_repaired, y_test, models, 'iris')
    repaired_qa1, repaired_qf1 = 1 - repaired_qa1, 1 - repaired_qf1

    # We compute the degree of repairability
    degree_rep_qa1 = rp.repairability_degree(unrepaired_qa1, repaired_qa1)
    degree_rep_qf1 = rp.repairability_degree(unrepaired_qf1, repaired_qf1)
    print(degree_rep_qa1, degree_rep_qf1)

    # We save the degree of repairability in Examples/DegreeOfRepairability
    np.save("DegreeOfRepairability/iris_missing_outlier_5_degree_repairability.npy", (degree_rep_qa1, degree_rep_qf1))
