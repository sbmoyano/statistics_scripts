# -*- coding: utf-8 -*-
"""
Created on Sun Nov 01 18:37:00 2020

@author: Sebastián Moyano

PhD Candidate at the Developmental Cognitive Neuroscience Lab (labNCd)
Center for Mind, Brain and Behaviour (CIMCYC)
University of Granada (UGR)
Granada, Spain

Description:
Function to compute different normality tests (Kolmogorov-Smirnov,
Shapiro-Wilks and D'Agostino-Pearson)
"""

# =============================================================================
# IMPORT LIBRARIES
# =============================================================================

import pandas as pd
import scipy.stats as stats

# =============================================================================
# NORMALITY TESTS
# =============================================================================


def normality_tests(df, list_vars):

    """
    Perform normality tests computing Kolmogorov-Smirnov, Shapiro-Wilks and
    D'Agostino-Pearson.

    Input:
        df: DataFrame
        list_vars: list of variable names (DataFrame columns) to check for
                   normality.

    Output:
        df_normality_tests: DataFrame with statistic and p-value for each variable.
    """

    # dictionary with tests
    dict_tests = {'K-S': stats.kstest, 'S-W': stats.shapiro, 'D-P': stats.normaltest}
    normality_dfs = []

    for normal_name, normal_module in dict_tests.items():
        if normal_name == 'K-S':
            df_normality_tests = pd.DataFrame(dict([(dv, normal_module(df[dv].dropna(), 'norm')) for dv
                                                   in list_vars])).rename({0: normal_name, 1: 'p-value'}).T
            normality_dfs.append(df_normality_tests)
        else:
            df_normality_tests = pd.DataFrame(dict([(dv, normal_module(df[dv].dropna()))
                                                   for dv in list_vars])).rename({0: normal_name, 1: 'p-value'}).T
            normality_dfs.append(df_normality_tests)

    df_normality_tests = pd.concat(normality_dfs, axis=1, sort=False).round(4)

    return df_normality_tests