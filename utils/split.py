from sklearn.model_selection import train_test_split
import numpy as np


def sampling(df, test_size):
    """
    Split the dataframe into test and train with the ratio test_size.

    :param df: (pandas dataframe) dataframe to split, the target for classification must be named 'class'
    :param test_size: (float) ratio of the dataframe that will be the test data (between 0 and 1)
    :return: (tuple of 2 pandas dataframes and 2 pandas series) X_train, X_test, y_train, y_test
    """
    X_train, X_test, y_train, y_test = np.nan, np.nan, np.nan, np.nan
    try:
        # separate into data X and target y
        X = df.loc[:, df.columns != 'class'].copy()
        y = df['class'].copy()
        # split the data and target into train and test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    except ValueError as e:
        print("failed split", e)
    return X_train, X_test, y_train, y_test
