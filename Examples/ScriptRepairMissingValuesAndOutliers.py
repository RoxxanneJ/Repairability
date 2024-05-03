import numpy as np
import pandas as pd
from multiprocessing import freeze_support
from multiprocessing import cpu_count
from pyod.models.iforest import IForest

if __name__ == '__main__':
    freeze_support()
    cpus = cpu_count()
    for p in [30, 35]:
        file = "../Datasets/bean/missing_outlier/bean_train_missing_outlier_" + str(p) + "_cleaned.csv"
        X_test = pd.read_csv("../Datasets/bean/bean_test.csv")
        y_test = X_test['class'].copy()
        X_test.drop(columns=['class'], inplace=True)
        X_train = pd.read_csv(file)
        y_train = X_train['class'].copy()
        X_train.drop(columns=['class'], inplace=True)

        # clf_name = 'IForest'
        clf = IForest(n_jobs=-1)
        clf.fit(X_train)
        # get the prediction labels and outlier scores of the training data
        y_train_pred = clf.labels_  # binary labels (0: inliers, 1: outliers)
        y_train_scores = clf.decision_scores_  # raw outlier scores
        X_train['outlier'] = y_train_pred.copy()
        X_train['class'] = y_train.copy()
        X_train.drop(X_train[X_train['outlier'] == 1].index, inplace=True)
        X_train.drop(columns=['outlier'], inplace=True)
        X_train.to_csv("../Datasets/bean/missing_outlier/repaired/bean_train_missing_outlier_"+str(p)+"_repaired.csv", index=False)
