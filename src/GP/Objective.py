''' This module holds objective functions for use by function trees & the GP. '''
import numpy as np
import pandas as pd
import ipdb
import sys

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
from sklearn.metrics import accuracy_score, f1_score

sys.path.append('../')
from util.Utils import *



def TreeRegressionMetric(data, tree, feats, metric='RMSE', optimGoal=-1):
    '''
    For a given dataset and tree, evalute the tree on the
    dataset, and assess the linear relationship between the
    results and the target in the data, returning the specified
    metric. Metric choices include 'RMSE', 'MSE', 'MAPE', 'R^2',
    or you can pass in a callable with 'y_true' and 'y_pred' arguments.
    :param data: dataframe of data; columns should include
        'target', and 'X0', 'X1', ...
    :param tree: evaluateable string function of a tree
    :param feats: list of feature column names
    :param metric: optional (default='RMSE') metric to compute
    :param optimGoal: flag indicating what to do with the metric
        (1 = maximize, -1 = minimize); this is only used to put a sign
        on a returned np.inf, in the case of an error
    :return metricVal: single-element tuple holding the value of the
        specified metric from a linear fit between the tree results
        and the target data column; if an error occurs np.inf is returned
    :return preds: array-like of predictions using linear regression model
    :return linReg: fit linear regression estimator
    '''
    
    # append the dataframe to the column name
    for feat in feats:
        tree = tree.replace(feat, 'data.'+feat)
        
    # evaluate the tree function
    treeRes = eval(tree)
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
        # compute the metric
        if metric == 'RMSE':
            metricVal = mean_squared_error(y_true=data['target'].values, y_pred=preds, squared=False)
        elif metric == 'MSE':
            metricVal = mean_squared_error(y_true=data['target'].values, y_pred=preds, squared=True)
        elif metric == 'MAPE':
            metricVal = mean_absolute_percentage_error(y_true=data['target'].values, y_pred=preds)
        elif metric == 'R^2':
            metricVal = r2_score(y_true=data['target'].values, y_pred=preds)
        elif not isinstance(metric, str):
            metricVal = metric(y_true=data['target'].values, y_pred=preds)
    except ValueError:
        # nans or infs in the tree res, so just pass out np.inf
        preds = [np.nan]*len(data)
        metricVal = np.inf*optimGoal*-1
    except Exception as err:
        # don't know, but perhaps an error with the metric function
        print('Unkonwn error: %s'%err)
        preds = [np.nan]*len(data)
        metricVal = np.inf*optimGoal*-1
    
    return (metricVal, preds, linReg)


def TreeClassificationMetric(data, tree, feats, metric='accuracy', optimGoal=-1):
    '''
    For a given dataset and tree, evalute the tree on the
    dataset, and assess the classification relationship between
    the results and the target in the data, returning the specified
    metric. Metric choices include 'accuracy', 'F1', 'wF1', 
    or you can pass in a callable with 'y_true' and 'y_pred' arguments.
    :param data: dataframe of data; columns should include
        'target', and 'X0', 'X1', ...
    :param tree: evaluateable string function of a tree
    :param feats: list of feature column names
    :param metric: optional (default='accuracy') metric to compute
    :param optimGoal: flag indicating what to do with the metric
        (1 = maximize, -1 = minimize); this is only used to put a sign
        on a returned np.inf, in the case of an error
    :return metricVal: single-element tuple holding the value of the
        specified metric from a linear fit between the tree results
        and the target data column; if an error occurs np.inf is returned
    :return preds: array-like of predictions using linear regression model
    :return linReg: fit decision tree estimator
    '''
    
    # append the dataframe to the column name
    for feat in feats:
        tree = tree.replace(feat, 'data.'+feat)
        
    # evaluate the tree function
    treeRes = eval(tree)
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
        decTree = DecisionTreeClassifier()
        decTree.fit(X=treeRes, y=data['target'].values)
        preds = decTree.predict(X=treeRes)
        # compute the metric
        if metric == 'accuracy':
            metricVal = accuracy_score(y_true=data['target'].values, y_pred=preds)
        elif metric == 'F1':
            metricVal = f1_score(y_true=data['target'].values, y_pred=preds)
        elif metric == 'wF1':
            metricVal = f1_score(y_true=data['target'].values, y_pred=preds, average='weighted')
        elif not isinstance(metric, str):
            metricVal = metric(y_true=data['target'].values, y_pred=preds)
    except ValueError:
        # nans or infs in the tree res, so just pass out np.inf
        preds = [np.nan]*len(data)
        metricVal = np.inf*optimGoal*-1
    except Exception as err:
        # don't know, but perhaps an error with the metric function
        print('Unkonwn error: %s'%err)
        preds = [np.nan]*len(data)
        metricVal = np.inf*optimGoal*-1
    
    return (metricVal, preds, decTree)


def TreeMetric(data, tree, feats, metric='RMSE', optimGoal=-1):
    '''
    For a given dataset and tree, evalute the tree on the
    dataset, and compute a specified metric between the target
    and the tree results data. Metric choices include 'RMSE',
    'MSE', 'MAPE', 'R^2', 'accurary', 'F1', 'wF1' or you can
    pass in a callable with 'y_true' and 'y_pred' arguments.
    :param data: dataframe of data; columns should include
        'target', and 'X0', 'X1', ...
    :param tree: evaluateable string function of a tree
    :param feats: list of feature column names
    :param metric: optional (default='RMSE') metric to compute
    :param optimGoal: flag indicating what to do with the metric
        (1 = maximize, -1 = minimize); this is only used to put a sign
        on a returned np.inf, in the case of an error
    :return metricVaL: single-element tuple holding the value
        of the specified metric between the tree results and
        the target data column; if an error occurs np.inf is returned
    :return treeRes: array-like of tree results
    '''
    
    # append the dataframe to the column name
    for feat in feats:
        tree = tree.replace(feat, 'data.'+feat)
        
    # evaluate the tree function
    treeRes = eval(tree)
    if type(treeRes) is pd.Series:
        # if the tree just encodes a series, have to get values
        treeRes = treeRes.values
    try:
        treeRes = treeRes.reshape(-1, 1)
    except AttributeError:
        # if the tree just encodes a constant value, make an array
        treeRes = np.array([treeRes]*len(data)).reshape(-1, 1)
    
    # compute the metric
    try:
        if metric == 'RMSE':
            metricVal = mean_squared_error(y_true=data['target'].values, y_pred=preds, squared=False)
        elif metric == 'MSE':
            metricVal = mean_squared_error(y_true=data['target'].values, y_pred=preds, squared=True)
        elif metric == 'MAPE':
            metricVal = mean_absolute_percentage_error(y_true=data['target'].values, y_pred=preds)
        elif metric == 'R^2':
            metricVal = r2_score(y_true=data['target'].values, y_pred=preds)
        elif metric == 'accuracy':
            metricVal = accuracy_score(y_true=data['target'].values, y_pred=preds)
        elif metric == 'F1':
            metricVal = f1_score(y_true=data['target'].values, y_pred=preds)
        elif metric == 'wF1':
            metricVal = f1_score(y_true=data['target'].values, y_pred=preds, average='weighted')
        elif not isinstance(metric, str):
            metricVal = metric(y_true=data['target'].values, y_pred=preds)
    except ValueError:
        # nans or infs in the tree res, so just pass out np.inf
        metricVaL = np.inf*optimGoal*-1
    except Exception as err:
        # don't know, but perhaps an error with the metric function
        print('Unkonwn error: %s'%err)
        preds = [np.nan]*len(data)
        metricVal = np.inf*optimGoal*-1
    
    return (metricVaL, treeRes)