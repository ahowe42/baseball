''' This module holds objective functions for use by function trees & the GP. '''
import numpy as np
import pandas as pd
import ipdb
import sys

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

sys.path.append('../')
from util.Utils import *



def RegressionRMSE(data, tree):
    '''
    For a given dataset and tree, evalute the tree on the
    dataset, and assess the linear relationship between the
    results and the target in the data.
    :param data: dataframe of data; columns should include
        'target', and 'X0', 'X1', ...
    :param tree: evaluateable string function of a tree
    :return RMSE: single-element tuple holding RMSE from a
        linear fit between the tree results and the target data column
    :return preds: array-like of predictions using linear regression model
    :return linReg: fit linear regression estimator
    '''
    # evaluate the tree function
    treeRes = eval(tree.replace('X', 'data.X'))
    if type(treeRes) is pd.Series:
        # if the tree just encodes a series, have to get values
        treeRes = treeRes.values
    try:
        treeRes = treeRes.reshape(-1, 1)
    except AttributeError:
        # if the tree just encodes a constant value, make an array
        treeRes = np.array([treeRes]*len(data)).reshape(-1, 1)
    
    # regression between target and the tree results
    try:
        linReg = LinearRegression(fit_intercept=False)
        linReg.fit(X=treeRes, y=data['target'].values)
        preds = linReg.predict(X=treeRes)
        RMSE = mean_squared_error(y_true=data['target'].values, y_pred=preds, squared=False)
    except ValueError:
        # nans or infs in the tree res, so just pass out np.inf for RMSE
        preds = [np.nan]*len(data)
        RMSE = np.inf
    
    return (RMSE, preds, linReg)


def TreeRMSE(data, tree):
    '''
    For a given dataset and tree, evalute the tree on the
    dataset, and compute the RMSE between the target and the
    tree results data.
    :param data: dataframe of data; columns should include
        'target', and 'X0', 'X1', ...
    :param tree: evaluateable string function of a tree
    :return RMSE: single-element tuple holding RMSE between
        the tree results and the target data column
    :return treeRes: array-like of tree results
    '''
    # evaluate the tree function
    treeRes = eval(tree.replace('X', 'data.X'))
    if type(treeRes) is pd.Series:
        # if the tree just encodes a series, have to get values
        treeRes = treeRes.values
    try:
        treeRes = treeRes.reshape(-1, 1)
    except AttributeError:
        # if the tree just encodes a constant value, make an array
        treeRes = np.array([treeRes]*len(data)).reshape(-1, 1)
    
    # compute RMSE
    try:
        RMSE = mean_squared_error(y_true=data['target'].values, y_pred=treeRes, squared=False)
    except ValueError:
        # nans or infs in the tree res, so just pass out np.inf for RMSE
        RMSE = np.inf
    
    return (RMSE, treeRes)