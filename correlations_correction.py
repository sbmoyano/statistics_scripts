# -*- coding: utf-8 -*-
"""
Created on Sun Nov 01 19:24:00 2020

@author: Sebastián Moyano

PhD Candidate at the Developmental Cognitive Neuroscience Lab (labNCd)
Center for Mind, Brain and Behaviour (CIMCYC)
University of Granada (UGR)
Granada, Spain

Description:
Function to compute correlation coefficients for multiple variables of the
same DataFrame, applying corrections for multiple comparisons and controlling
by covariates.
"""

# =============================================================================
# IMPORT LIBRARIES
# =============================================================================

import pandas as pd
import statsmodels.stats as sm
import pingouin as pg

# =============================================================================
# CORRELATIONS
# =============================================================================


def correlations(df, var_list, tail, correction, method, cov_list, *args):
    """
    Compute correlations por each group of variables dropping variables of the same group (i.e. if
    we have Depression_mother, Depression_father, Depression_daughter, correlations between these
    variables that are part of the same set of variables are excluded, only if we specified a common
    name of these variables ('Depression') in *args. Otherwise they are included.
    Apply a correction for multiple comparisons for all uncorrected p-values of each group of variables,
    not to all the correlations in the DataFrame. Joins everything in an unique DataFrame.

    Input:
        df: DataFame with data.
        var_list: list of lists with variables for correlations.
        tail (str): 'one-sided' or 'two-sided'.
        correction (str): check statsmodels.multitest.multipletest for correction options.
        method (str): 'pearson', 'spearman', etc. (check methods for pg.pairwise_corr).
        cov_list: list of covariates.
        *args (str): strings to exclude intra group correlations from DataFrame and apply
                     correction without these correlations. The string should be a name that
                     is common to all the names of the variables that are part of the group.

                     Specify one string for the X column of the DataFrame:
                     - 'Depression' if is the common string - Depression_mother, Depression_father, etc.
                        and we just have that group for the X column
                     - 'Depression|Anger' if we have two groups - Depression_mother, Depression_father, etc. +
                        Anger_mother, Anger_father, etc.

                     Specify another string with the same logic for the Y column of the DataFrame.
    Output:
        df: DataFrame with correlation values dropping correlations between variables of the same group.
    """

    list_of_dfs = []

    # list_to_exclude_both_rows = [excludeintra_X, excludeintra_Y, excludeintra_Z]
    df_corr = pd.DataFrame(pg.pairwise_corr(df, columns=var_list, covar=cov_list, tail=tail, method=method,
                                            nan_policy='pairwise')).round(5)

    # drop intra task correlations. Reset index because of dropped rows
    # to ease concat
    for row_excl in args:
        # drop rows that contains the same string in both columns X and Y
        df_corr = df_corr[~((df_corr['X'].str.contains(row_excl)) & (df_corr['Y'].str.contains(row_excl)))].reset_index(
            drop=True)

    # apply correction to p-values of no intratask correlations
    # transpose df as it returns two rows and multiple columns
    # and rename columns
    FDR_corr = list(sm.multitest.multipletests(df_corr['p-unc'], alpha=0.05, method=correction, is_sorted=False,
                                               returnsorted=False))
    # extract in a dict alpha corrected by Sidak and Bonferroni to ease concat
    alpha_SidakBonf = dict(alphacSidak=FDR_corr[2], alphacBonf=FDR_corr[3])
    # select two first elements, transpose and rename columns
    FDR_corr = pd.DataFrame(FDR_corr[0:2]).transpose().rename(columns={0: 'FDR', 1: 'pvals_corrected'})
    # add columns of p-values with FDR correction to df
    df_corr_FDR = pd.concat([df_corr, FDR_corr], axis=1, join='outer', ignore_index=False)
    # add cloumns with corrected p-values
    df_corr_FDR['alphacSidak'], df_corr_FDR['alphacBonf'] = alpha_SidakBonf['alphacSidak'], alpha_SidakBonf[
        'alphacBonf']
    # list of dfs, each df contains correlations for each frequency band
    list_of_dfs.append(df_corr_FDR)

    # concat list of dfs into 1 df
    # reset index. After concat remains the index of each df on its own rows
    df = pd.concat(list_of_dfs).reset_index(drop=True)

    return df
