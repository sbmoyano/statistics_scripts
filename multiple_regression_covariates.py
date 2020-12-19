# -*- coding: utf-8 -*-
"""
Created on Sat Dic 19 15:21:00 2020

@author: Sebastián Moyano

PhD Candidate at the Developmental Cognitive Neuroscience Lab (labNCd)
Center for Mind, Brain and Behaviour (CIMCYC)
University of Granada (UGR)
Granada, Spain

Description:
Function to perform multiple regression analysis including covariates.
Neccesary normality_test functions available in my GitHub. Otherwise,
remove it.
"""

import pandas as pd
import scipy.stats as stats
from scipy.stats import norm
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns


def multiple_reg(df, predictors, dv, cov=None):
    """ Perform multiple linear regression with covariates and plot
    residuals.

    Input:
        df: DataFrame.
        predictors (list): list of predictor variables.
        dv (str): dependent variable as string.
        cov (str): covariate to control for. If none then None.

    Output:
        Multiple regression coefficients controlling for covariates,
        and plot residuals.
    """

    #  predictor
    if cov != None:
        # new predictor with covariate
        pred = [cov] + predictors
        # to filter the df
        merge_pred_cov_dv = [cov] + predictors
        merge_pred_cov_dv.append(dv)
        # drop NaN values
        df = df[merge_pred_cov_dv].dropna()
    else:
        # new predictor
        pred = predictors
        pred.append(dv)
        df = df[pred].dropna()

    X = df[pred]
    y = df[dv]
    X = sm.add_constant(X, has_constant='add')
    model = sm.OLS(y, X, missing='drop')
    results = model.fit()

    # show regression statistics
    print('\n Simple Linear Regression statistics: \n')
    display(results.summary())

    # normality tests for residuals
    residuals = pd.DataFrame(results.resid, columns=['values'])
    normality_residuals = normality_tests(residuals, ['values'])

    # plot residuals
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    print('Distribution of the residuals: \n')
    stats.probplot(results.resid, plot=ax[0])
    sns.distplot(results.resid, fit=norm, color='indianred', ax=ax[1])
    plt.show()

    print('Normality tests for residuals: \n')
    display(normality_residuals)

    return results, normality_tests
