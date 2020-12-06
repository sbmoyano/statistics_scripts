# -*- coding: utf-8 -*-
"""
Created on Fri Dic 06 20:05:00 2020

@author: Sebastián Moyano

PhD Candidate at the Developmental Cognitive Neuroscience Lab (labNCd)
Center for Mind, Brain and Behaviour (CIMCYC)
University of Granada (UGR)
Granada, Spain

Description:
Function to plot bars for main effects and interactions. A maximum of
two levels on the dependent variable are included. The function could
be modified to include more.
"""

# =============================================================================
# IMPORT LIBRARIES
# =============================================================================

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# =============================================================================
# PLOT DISTRIBUTION
# =============================================================================


def plot_main_effects_compute_errors(df, col_groupby, dv, xlabel, ylabel, labels_groups):
    """
    Plot main effects on a variable. Error bars are compute
    dividing SD by sample size of each group.

    Input:
        df: DataFrame with data.
        col_groupby(str): name of the column with group labels
        dv(str): name of column with the dependent variable
        xlabel(str): x-axis label name
        ylabel(str): y-axis label name
        labels_groups(list): list of labels for each group in the col_groupby variable

    Output:
        Plot. You can also return the DataFrame with stats info (optional)
    """

    # compute errors based on sample size
    df_stats = df.groupby([col_groupby])[dv].describe()
    # activate in case you do df.groupby.describe()
    # df_stats.columns = df_stats.columns = ['_'.join(col) for col in df_stats.columns.values]
    df_stats['error'] = df_stats['std'] / np.sqrt(df_stats['count'])
    df_stats.reset_index(drop=False, inplace=True)

    # plot
    sns.set_style('white')
    sns.set_context('paper')
    fig, ax = plt.subplots(figsize=(7, 5))
    df_stats.plot(kind='bar', x=col_groupby, y='mean', yerr='error', capsize=5,
                  error_kw={'errwidth': 1}, color='Grey', legend=None, ax=ax)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(ticks=(0, 1, 2, 3, 4), labels=labels_groups, rotation=0)
    sns.despine()


def plot_interaction_effect_compute_errors(df, col_groupby, dv1, dv2, xlabel, ylabel, dv1_name, dv2_name,
                                           labels_groups):
    """
    Plot interaction of a dependent variable with two levels. Error bars
    are compute dividing SD by sample size of each group.

    Input:
        df: DataFrame with data
        col_groupby(str): name of the column with group labels
        dv1(str): name of the column with the first dependent variable
        dv2(str): name of the column with the second dependent variable
        xlabel(str): x-axis label name
        ylabel(str): y-axis label name
        dv1_name(str): alternative label for the first dependent variable
        dv2_name(str): alternative label for the second dependent variable
        labels_groups(list): list of labels for each group in the col_groupby variable
    """

    df_stats = df[[col_groupby, dv1, dv2]].set_index(col_groupby)
    df_stats = df_stats[[dv1, dv2]].stack().reset_index(drop=False).rename(
        columns={'level_1': 'exp_condition', 0: 'measure'})

    # compute errors based on sample size
    df_stats = df_stats.groupby([col_groupby, 'exp_condition'])['measure'].describe()
    # activate in case you do df.groupby.describe()
    # df_stats.columns = df_stats.columns = ['_'.join(col) for col in df_stats.columns.values]
    df_stats['error'] = df_stats['std'] / np.sqrt(df_stats['count'])
    df_stats.reset_index(drop=False, inplace=True)

    # list with experimental conditions
    list_conditions = [dv1, dv2]

    # plot
    ind = np.arange(len(list(df[col_groupby].unique())))  # the x locations for the groups
    width = 0.35  # the width of the bars
    sns.set_style('white')
    sns.set_context('paper')
    fig, ax = plt.subplots(figsize=(7, 5))
    df_plot_one = df_stats[df_stats['exp_condition'] == list_conditions[0]]
    df_plot_two = df_stats[df_stats['exp_condition'] == list_conditions[1]]
    rect1 = ax.bar(ind - width / 2, df_plot_one['mean'], width, label=dv1_name, yerr=df_plot_one['error'],
                   error_kw={'errwidth': 1, 'capsize': 5}, color='dimgray')
    rect2 = ax.bar(ind + width / 2, df_plot_two['mean'], width, label=dv2_name, yerr=df_plot_two['error'],
                   error_kw={'errwidth': 1, 'capsize': 5}, color='silver')

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(ticks=(0, 1, 2, 3, 4), labels=labels_groups, rotation=0)
    plt.legend(frameon=False)
    sns.despine()
