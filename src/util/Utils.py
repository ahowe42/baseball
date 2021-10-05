''' This module holds generic utility functions, plus functions for binary artihmetic operations. '''
import numpy as np
import pandas as pd
import datetime as dt
import ipdb
from collections import OrderedDict
import time
from itertools import chain
import copy
import re

import chart_studio.plotly as ply
import chart_studio.tools as plytool
import plotly.figure_factory as ff
import plotly.graph_objs as go
import plotly.offline as plyoff
import plotly.subplots as plysub

from sklearn.linear_model import LinearRegression



''' generic function to perform weighted random selection '''
def RandomWeightedSelect(keys, wats, randSeed=None):
    '''
    Randomly select an item from a list, according to a set of
    specified weights.
    :param keys: array-like of items from which to select
    :param wats: array-like of weights associated with the input
        keys; must be sorted in descending weight
    :param randSeed: optional random seed for np.random; if no
        randomization is desired, pass 0
    :return selection: selected item
    :return randSeed: random seed used
    '''
    
    # ranodmize, perhaps
    if randSeed != 0:
        if randSeed is None:
            randSeed = int(str(time.time()).split('.')[1])
        np.random.seed(randSeed)
    
    # ensure weights sum to 1
    totWats = sum(wats)
    if totWats != 1:
        wats = [v/totWats for v in wats]
    
    # get the cumulative weights
    cumWats = np.cumsum(wats)
    # get the indices of where the random [0,1] is < the cum weight
    rnd = np.random.rand()
    seld = rnd < cumWats
    
    return [k for (k,s) in zip(keys, seld) if s][0], randSeed


''' binary arithmetic operations that can be called as functions '''
# addition
def ad(a, b):
    return a+b


# subtraction
def sb(a, b):
    return a-b


# multiplication
def ml(a, b):
    return np.nan_to_num(a * b, posinf=np.nan, neginf=np.nan)


# division
def dv(a, b):
    # check for longest dimensional match
    try:
        lna = len(a)
    except TypeError:
        lna = 1
    try:
        lnb = len(b)
    except TypeError:
        lnb = 1
    # compute
    if (lna == lnb) & (lna == 1):
        # both scalars
        if b != 0:
            res = a/b
        else:
            res = np.nan
    else:
        # at least 1 iterable
        if lna < lnb:
            # a is scalar, b is not
            a = [a]*lnb
        elif lnb < lna:
            # b is scalar, a is not
            b = [b]*lna
        res = np.nan_to_num(a / b, posinf=np.nan, neginf=np.nan)
    return res


# power
def pw(a, b):
    # determine if either a or b is an iterable; if so, need to ensure
    # inputs are floats
    try:
        lna = len(a)
    except TypeError:
        # not an iterable
        lna = 0
        a = float(a)
    else:
        # an iterable, so cast
        a = a.astype('float')
    try:
        lnb = len(b)
    except TypeError:
        # not an iterable
        lnb = 0
        b = float(b)
    else:
        # an iterable, so cast
        b = b.astype('float')
        
    # if either is an iterable, the rest of this function is simpler if both are (using xor)
    if (lna > 0) ^ (lnb > 0):
        if lna == 0:
            a = np.ones(b.shape)*a
            lna = lnb
        elif lnb == 0:
            b = np.ones(a.shape)*b
            lnb = lna
    
    # need to specially handle 0 ^ (negative)
    if (lna == 0) & (lnb == 0):
        # both scalars
        if (a == 0) & (b < 0):
            res = np.nan
        else:
            try:
                res = np.nan_to_num(a**b, posinf=np.nan, neginf=np.nan)
            except OverflowError:
                # unsure why this is needed, as it should just to to np.inf?
                res = np.nan
    elif (lna > 0) & (lnb > 0):
        # both iterables
        bads = (a==0) & (b<0)
        if any(bads):
            # initialize results
            res = np.zeros(a.shape)
            # fill partly with nans
            res[bads] = np.nan
            # compute for the goods
            goods = np.logical_not(bads)
            try:
                res[goods] = np.nan_to_num(a[goods]**b[goods], posinf=np.nan, neginf=np.nan)
            except OverflowError:
                # not sure how to find what caused this, so just set all to nan
                res[goods] = np.nan
        else:
            res = np.nan_to_num(a**b, posinf=np.nan, neginf=np.nan)
        
    return res


# minimum
def mn(a, b):
    # check for longest dimensional match
    try:
        lna = len(a)
    except TypeError:
        lna = 1
    try:
        lnb = len(b)
    except TypeError:
        lnb = 1
    # compute
    if (lna == lnb) & (lna == 1):
        # both scalars
        res = min(a, b)
    elif lna == lnb:
        # both iterables
        res = np.where(a < b, a, b)
    elif lna < lnb:
        # a is scalar, b is not
        tmp = [a]*lnb
        res = np.where(tmp < b, tmp, b)
    elif lnb < lna:
        # b is scalar, a is not
        tmp = [b]*lna
        res = np.where(a < tmp, a, tmp)
    return res


# maximum
def mx(a, b):
    # check for longest dimensional match
    try:
        lna = len(a)
    except TypeError:
        lna = 1
    try:
        lnb = len(b)
    except TypeError:
        lnb = 1
    # compute
    if (lna == lnb) & (lna == 1):
        # both scalars
        res = max(a, b)
    elif lna == lnb:
        # both iterables
        res = np.where(a > b, a, b)
    elif lna < lnb:
        # a is scalar, b is not
        tmp = [a]*lnb
        res = np.where(tmp > b, tmp, b)
    elif lnb < lna:
        # b is scalar, a is not
        tmp = [b]*lna
        res = np.where(a > tmp, a, tmp)
    return res


def ResultsPlots(data, sequenceCol, responseCol, predCol, resdCol, colorCol, overall_title, plot_colors=('red',)*4):
    '''
    This creates and returns a quad-plot of results from a model. The four plots are: a) prediction vs response,
    b) histogram of residuals, c) residuals by sequence, d) residuals by response. They are arranged as
    [[a,b],[c,d]].
    :param data: dataframe holding the sequence, response, prediction, and residual columns
    :param sequenceCol: column in the dataframe holding the time or sequence counter; if None, a counter will be used
    :param responseCol: column in the dataframe holding the response variable
    :param predCol: column in the dataframe holding the model predictions
    :param resdCol: column in the dataframe holding the model residuals
    :param colorCol: optional column in the dataframe holding the color for each observation for the plots
    :param overall_title: title to go on top of the quad plot
    :param plot_colors (default=(red, red, red, red)): optional tuple of colors for each plot; only used if colorCol
        not passed
    :return fig: plotly plot figure
    '''
    
    # copy the input dataframe
    data = data.copy(deep=True)
    
    if colorCol is None:
        colorCol = 'NOCOLOR'
    
    # setup the subplot
    figRes = plysub.make_subplots(rows=2, cols=2, subplot_titles=['Predictions vs Responses', 'Residuals Distribution','Residuals by Sequence', 'Residuals vs Responses'])
    
    # for the actual vs preds plot, build the fit line
    actual = data[responseCol].values.reshape(-1,1)
    lindat = np.linspace(actual.min(), actual.max(), 10).reshape(-1, 1)
    fitlin = LinearRegression(n_jobs=-1)
    fitlin.fit(X=actual, y=data[predCol])
    actpred = fitlin.predict(X=lindat)
    r2 = np.corrcoef(actual.squeeze(), data[predCol].values)[0][1]
    # create the trace and annotation
    r2trc = go.Scatter(x=lindat.squeeze(), y=actpred, mode='lines', name='fit', line={'color':'black','width':1}, showlegend=False)
    r2ann = dict(x=lindat[5][0], y=actpred[5], xref='x1', yref='y1', text='$\\rho=%0.3f$'%(r2), showarrow=False, bgcolor='#ffffff')
    
    # actuals vs resids plot
    if colorCol == 'NOCOLOR':
        data[colorCol] = plot_colors[0]
    figRes.add_trace(go.Scatter(x=data[responseCol], y=data[predCol], mode='markers', marker={'color':data[colorCol]}, showlegend=False), 1,1)
    figRes.add_trace(r2trc, 1, 1)
    figRes['layout']['xaxis1'].update(title=responseCol)
    figRes['layout']['yaxis1'].update(title=predCol)
    
    # residuals histogram
    if colorCol == 'NOCOLOR':
        data[colorCol] = plot_colors[1]
    figRes.add_trace(go.Histogram(x=data[resdCol], histnorm='', marker={'color':data[colorCol]}, showlegend=False), 1,2)
    figRes['layout']['xaxis2'].update(title=resdCol)
    figRes['layout']['yaxis2'].update(title='count')
    
    # get the time variable
    if sequenceCol is None:
        seq = list(range(len(data)))
        seqNam = 'sequence'
    else:
        seq = data[sequenceCol]
        seqNam = sequenceCol
    
    # residuals by time plot
    if colorCol == 'NOCOLOR':
        data[colorCol] = plot_colors[2]
    figRes.add_trace(go.Scatter(x=seq, y=data[resdCol], mode='markers', marker={'color':data[colorCol]}, showlegend=False), 2, 1)
    figRes['layout']['xaxis3'].update(title=seqNam)
    figRes['layout']['yaxis3'].update(title=resdCol)
    
    # residuals by response plot
    if colorCol == 'NOCOLOR':
        data[colorCol] = plot_colors[3]
    figRes.add_trace(go.Scatter(x=data[responseCol], y=data[resdCol], mode='markers', marker={'color':data[colorCol]}, showlegend=False), 2,2)
    figRes['layout']['xaxis4'].update(title=responseCol)
    figRes['layout']['yaxis4'].update(title=resdCol)
    
    # update layout
    figRes['layout'].update(title=overall_title, height=1000)
    anns = list(figRes['layout']['annotations'])
    anns.append(r2ann)
    figRes['layout']['annotations'] = anns
    
    return figRes