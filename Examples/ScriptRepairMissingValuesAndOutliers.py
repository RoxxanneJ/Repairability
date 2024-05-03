import numpy as np
import pandas as pd
from multiprocessing import freeze_support
from pyod.models.iforest import IForest

datasets = ["iris", "cancer", "adult", "heart", "abalone", "statlog", "spambase", "bean"]
errors = ["missing", "outlier", "missing_outlier"]

if __name__ == '__main__':
    freeze_support()
    for data in datasets:
        for error in errors:
            for p in range(5, 55, 5):
                file = "../Datasets/" + data + "/" + error + "/" + data + "_train_" + error + "_" + str(p) + ".csv"
                df = pd.read_csv(file)

                # We first repair missing values by replacing them with the mean of the column
                df.fillna(value=df.mean(), inplace=True)

                # We then repair outliers by detecting them with an isolation forest and removing them
                clf = IForest(n_jobs=-1)
                clf.fit(df)
                # get the prediction labels and outlier scores of the training data
                y_train_pred = clf.labels_  # binary labels (0: inliers, 1: outliers)
                y_train_scores = clf.decision_scores_  # raw outlier scores
                df['outlier'] = y_train_pred.copy()
                df.drop(df[df['outlier'] == 1].index, inplace=True)
                df.drop(columns=['outlier'], inplace=True)
                df.to_csv("../Examples/RepairedDataset/"+data+"_"+error+"_"+str(p)+"_repaired.csv", index=False)
