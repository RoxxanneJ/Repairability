import random as rd
import time as t
import numpy as np
import math as m
import pandas as pd


def degrade(error_name, dataset, percentage, number=False):
    """
    Apply the function injecting the error error_name in the dataset at a percentage.
    If number is passed as an argument, it overrides percentage and define the number of errors to be injected.

    :param error_name: (string) name of the error to be injected
    :param dataset: (pandas dataframe) dataset the errors are to be injected in
    :param percentage: (int) percentage of errors to inject in data (for missing and outliers it's a percentage of cells
    in the dataset while for partial duplicates it's a percentage of samples of the dataset)
    :param number: (int) when passed it overrides the parameter percentaeg an is the number of cells to inject errors
    in (for missing and outliers) or rows (for partial duplicates)
    :return: (pandas dataframe) the dataset containing errors
    """
    if error_name == "missing":
        dataset = missing_value_generation(dataset, percentage, number)
    elif error_name == "outlier":
        dataset = outlier_generation(dataset, percentage, number)
    elif error_name == "fuzzing":
        dataset = fuzzing_generation(dataset, percentage, number)
    return dataset


def split_n(dataframe, n):
    """
    Split the dataframe in a list of n disjoint sub dataframes of equal sizes which rows have been randomly selected
    from the original dataframe.
    (This function is not used for the metric I use it to generate deteriorated datasets with more than on type of error
    for my experiments)

    :param dataframe: (pandas dataframe) dataframe to split
    :param n: (int) number of sub dataframes wanted
    :return: (list of dataframes)
    """
    sub_dataframes = []
    rows = dataframe.index.values.tolist()
    sub_size = len(rows)//n
    for sub_data in range(n):
        selected_rows = np.random.choice(rows, sub_size, replace=False, p=[1 / len(rows)] * len(rows))
        for el in selected_rows:
            rows.remove(el)
        sub_dataframes.append(dataframe.loc[selected_rows].copy())
    return sub_dataframes


def gen_dataset(data, errors, i):
    """
    Inject errors in the list errors in disjointed subsections of the data.
    The global percentage of errors injected is i and each error in errors is present in the same quantity.

    :param data: (pandas dataframe) dataset we inject the errors in,
    the target for classification needs to be named 'class'
    :param errors: (list of string) list of the names of the errors we want to inject in data
    :param i: (int) total percentage of errors to inject
    :return: (pandas dataframe) data with the errors injected
    """
    np.random.seed(seed=int(t.time()))
    d = data.copy()
    y = data['class'].copy()  # we make a copy of the target because we don't inject errors in it
    nb_errors = len(errors)
    sub_dataset = split_n(d, nb_errors)  # we split the data in n disjointed parts to inject the different errors in
    for k in range(nb_errors):
        if errors[k] != "fuzzing":
            # if the errors are missing values or outliers we drop the classification target in order to avoid
            # degrading it, it will be reattached later
            sub_dataset[k].drop(columns=['class'], inplace=True)
        if errors[k] == "fuzzing":
            # compute the number of row to inject with errors of from the global percentage of error to inject
            nb_to_modify = round(d.shape[0] * i / (nb_errors * 100))
        else:
            # compute the number of cells to inject with errors of from the global percentage of error to inject
            nb_to_modify = round(d.shape[0] * (d.shape[1] - 1) * i / (nb_errors * 100))

        # inject the errors in the sub dataset (number always overrides percentage here)
        sub_dataset[k] = degrade(error_name=errors[k], dataset=sub_dataset[k], percentage=i, number=nb_to_modify)

        if errors[k] != "fuzzing":  # reattach the classification target
            sub_dataset[k]['class'] = y

    d = pd.concat(sub_dataset)  # concatenate the sub datasets to recreate a full dataset
    return d


def missing_value_generation(data, percentage, number_of_missing=False):
    """
    Inject missing values by replacing values with None in data for a percentage of its cells or for a number of its cells if number_of_missing is
    passed (number_of_missing overrides percentage when passed).

    :param data: (dataframe) data to inject the missing values in
    :param percentage: (int) percentage of the cells of the data to replace values with None
    :param number_of_missing: (int) number of the cells of the data to replace values with None
    (False by default but overrides percentage when passed)
    :return: (dataframe) data that has been injected with missing values
    """
    try:
        np.random.seed(seed=int(t.time()))
        if not number_of_missing:
            number_of_missing = m.ceil(percentage * data.shape[0] * data.shape[1] / 100)

        cells = []
        rows = data.index.values.tolist()
        cols = list(data.columns)
        for k in range(data.shape[1]):
            cells.append(rows.copy())
        for cell in range(number_of_missing):
            if cols:
                col = np.random.choice(cols, 1, p=[1 / len(cols)] * len(cols))[0]
                index_col = data.columns.get_loc(col)
                row = np.random.choice(cells[index_col], 1, p=[1 / len(cells[index_col])] * len(cells[index_col]))[0]
                cells[index_col].remove(row)
                if not cells[index_col]:
                    cols.remove(col)  # If all the cells in the column have been selected before, the column is removed
                data.at[row, col] = None
    except ValueError as e:
        print("gen_missing_values() failed", e)
    return data


def gen_fuzz(data, row, nb_to_fuzz):
    """
    Generate a fuzzy row from a real dataframe row. The attributes to be fuzzed are chosen randomly and their values in
    the fuzzy row are randomly modified (uniform distribution) to be in the interval [v_0 - std*0.01 ; v_0 + std*0.01[
    with v_0 the original value for this attribute in the row and std the standard deviation on this attribute in the
    dataframe.
    :param data: (pandas dataframe)
    :param row: (pandas series) row to be fuzzed from the pandas dataframe
    :param nb_to_fuzz: (int) number of attributes to fuzz in the row entered
    :return: (pandas series) fuzzy row
    """
    np.random.seed(seed=int(t.time()))
    stds = data.std()
    columns = data.columns.values.tolist()
    if 'class' in columns:
        columns.remove('class')
    to_fuzz = np.random.choice(columns, nb_to_fuzz, p=[1/len(columns)]*len(columns))
    for col in to_fuzz:
        std = stds[col]
        row[col] = rd.uniform(row[col] - std*0.01, row[col] + std*0.01)
    return row


def fuzzing_generation(data, percentage, number_of_duplicates=False):
    """
    Inject partial duplicates by duplicating and fuzzing rows in data for a percentage of its rows or for a number of
    its rows if number_of_duplicates is passed (number_of_duplicates overrides percentage when passed).

    :param data: (dataframe) data to inject the partial duplicates in
    :param percentage: (int) percentage of the rows of the data to duplicate and fuzz
    :param number_of_duplicates: (int) number of the rows of the data to duplicate and fuzz
    (False by default but overrides percentage when passed)
    :return: (dataframe) data that has been injected with partial duplicates
    """
    try:
        np.random.seed(seed=int(t.time()))
        if not number_of_duplicates:
            number_of_duplicates = m.ceil(data.shape[0] * percentage / 100)

        data.reset_index(drop=True, inplace=True)
        indexes = data.index.values.tolist()

        for k in range(number_of_duplicates):
            rd.seed(t.time())
            row_index = np.random.choice(indexes, 1)[0]
            row = data.iloc[row_index].copy()
            nb_to_fuzz = rd.randint(1, int(data.shape[1]/10) + 1)
            row = gen_fuzz(data, row, nb_to_fuzz)  # fuzz the duplicated row before injecting it in the dataset
            data = data.append(row, ignore_index=True)
    except ValueError as e:
        print("gen_duplicates_partial() failed", e)
    return data


def outlier_generation(data, percentage, number_of_outliers=False):
    """
    Inject outliers by replacing values with outliers a percentage of the dataset cells or for a number of its cells if
     number_of_outliers is passed (number_of_outliers overrides percentage when passed).
     An outlier in a cell in the column c is a randomly generated integer or float (depending on the type of the column)
     in the interval [q_0.003(c) - 2std(c), q_0.003(c)] U [q_0.997(c), 2std(c) + q_0.997(c)] for integers,
     and [q_0.003(c) - 2std(c), q_0.003(c)[ U [q_0.997(c), 2std(c) + q_0.997(c)[ for float
     with q_x(c) the x quantile of column c and std(c) the standard deviation of column c.

    :param data: (pandas dataframe) dataset to inject the outliers in
    :param percentage: (int) percentage of cells to replace by outliers
    :param number_of_outliers: (int) number of cells to replace by outliers if passed (overrides percentage)
    set to False otherwise
    :return: (pandas dataframe) dataset that has been injected with outliers
    """
    try:
        np.random.seed(seed=int(t.time()))
        if not number_of_outliers:
            number_of_outliers = m.ceil(percentage * data.shape[0] * data.shape[1] / 100)

        rd.seed(t.time())
        cells = []
        rows = data.index.values.tolist()
        cols = list(data.columns)
        types = data.dtypes
        quantiles_low = data.quantile(0.003)  # high and low quantiles for outliers
        quantiles_high = data.quantile(0.997)
        for k in range(data.shape[1]):
            cells.append(rows.copy())
        for cell in range(number_of_outliers):
            if cols:
                col = np.random.choice(cols, 1, p=[1 / len(cols)] * len(cols))[0]
                index_col = data.columns.get_loc(col)
                row = np.random.choice(cells[index_col], 1, p=[1 / len(cells[index_col])] * len(cells[index_col]))[0]
                cells[index_col].remove(row)
                if not cells[index_col]:
                    cols.remove(col)

                # we generate outliers that are at most distant from the high and low quantiles by 2 standard deviations
                radius = data[col].std() * 2
                # we randomly decide if the value will be an outlier from the high (foo=0) or low (foo=1) quantile
                foo = rd.randint(0, 1)
                if foo:  # low outlier
                    quantile = quantiles_low[col]
                    if isinstance(data.dtypes[col], int):  # the column type is int
                        value = rd.randint(int(quantile - radius), quantile)
                    else:  # the column type is float
                        value = rd.uniform(quantile - radius, quantile)
                else:  # high outlier
                    quantile = quantiles_high[col]
                    if isinstance(data.dtypes[col], int):  # the column type is int
                        value = rd.randint(quantile, int(quantile + radius))
                    else:  # the column type is float
                        value = rd.uniform(quantile, quantile + radius)
                if types[col] == 'int64':
                    value = int(value)
                data.at[row, col] = value  # replace cell value by outlier
    except ValueError as e:
        print("gen_outliers_mixed() failed", percentage, e)
    return data
