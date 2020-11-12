"""
Created on Thur Nov 12 21:47:00 2020

@author: Sebastián Moyano

PhD Candidate at the Developmental Cognitive Neuroscience Lab (labNCd)
Center for Mind, Brain and Behaviour (CIMCYC)
University of Granada (UGR)
Granada, Spain

Description:
Function plot the distribution of a dataset. It plots a histogram fitting
the normal curve distribution and a Q-Q plot below it.
"""

# =============================================================================
# IMPORT LIBRARIES
# =============================================================================

import pandas as pd
import scipy.stats as stats
from scipy.stats import norm
import seaborn as sns
import matplotlib.pyplot as plt

# =============================================================================
# PLOT DISTRIBUTION
# =============================================================================


def plot_distribution(df, vars_list, number_columns_per_row, transf=None):
    """
    Plot variable distribution using Seaborn distplot (histogram) and ScipyStats probplot
    (Q-Q plot). Plots histograms in the first row with normal curve and distribution curve
    and Q-Q plots in the second row.
    Drop NaN values before plotting Q-Q plots as ScipyStats does not work
    with NaN values.

    Input:
        df: DataFrame with data.
        vars_list: list of dependent variables to plot.
        number_columns_per_row (int): number of colums to be plotted.
        transf: transformation to apply to the data, using Numpy. Default set to None.

    Output:
        Histograms with normal curve and Q-Q plots.

    """

    def plot(df, vars_list, transf=transf):

        """
        Nested function that plots using Seaborn displot and ScipyStats probplot.
        Takes arguments from the main function.

        Input:
            df: df input for the main function.
            vars_list: vars_list for the main function.
            transf: transformation specified in the main funciton.

        Output:
            Plots
        """

        sns.set_context('paper', font_scale=1.5)

        # we just need the vds to be plotted
        df_plot = df[vars_list]

        if transf != None:
            df_plot = df_plot.apply(transf)
        else:
            df_plot = df_plot

        if len(vars_list) == 1:
            fig1, ax = plt.subplots(2, len(vars_list), figsize=(10, 10))
            plt.subplots_adjust(hspace=0.5)
            print('Plotting info for: ' + str(vars_list[0]))
            # drop NaN values for scipy.stats, seaborn ignore NaNs
            sns.distplot(df_plot[vars_list].dropna(), fit=norm, color='indianred', ax=ax[0])
            stats.probplot(df_plot[vars_list].dropna(), plot=ax[1])
            # after calling the plot
            sns.despine()

        elif len(vars_list) > 1:
            fig2, ax = plt.subplots(2, len(vars_list), figsize=(20, 10))
            plt.subplots_adjust(hspace=0.5)
            for i in range(len(vars_list)):
                print('Plotting info for: ' + str(vars_list[i]))
                # drop NaN values for scipy.stats, seaborn ignore NaNs
                sns.distplot(df_plot[vars_list[i]], fit=norm, color='indianred', ax=ax[0, i])
                stats.probplot(df_plot[vars_list[i]].dropna(), plot=ax[1, i])
                # after calling the plot
                sns.despine()

    onset = 0
    for i in range(onset, len(vars_list) + 1):

        modulus = len(vars_list[onset:onset + number_columns_per_row]) % number_columns_per_row

        # slice list
        sublist = vars_list[onset:onset + number_columns_per_row]

        # if the plots could be distributed in the number of columns per plot as specified
        # (i.e.) 18 variables and in groups of 3 (3 columns) = 18/3 = 0, just call nested function
        if modulus == 0:
            # call nested function
            plot(df, sublist, transf=None)
            onset += number_columns_per_row

        # if we have 19 variables to split in groups of three there will be one variables plotted alone
        # in this case...
        else:

            # while the length of the sublist is equal to the number of columns per row, just plot
            # the sublist
            while len(sublist) % number_columns_per_row == 0:
                # call nested function
                plot(df, sublist, transf=None)
                onset += number_columns_per_row

            # if the length of the sublist is not equal to the number of columns per row
            # create another list that would include the last dependent variables of the
            # main list and plot it without following the specified number of columns
            else:
                # call nested function
                plots = len(vars_list) % number_columns_per_row
                last_list = vars_list[-plots::]
                plot(df, last_list, transf=None)
                onset += plots