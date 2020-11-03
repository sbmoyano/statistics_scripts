# -*- coding: utf-8 -*-
"""
Created on Sun Nov 01 18:08:00 2020

@author: Sebastián Moyano

PhD Candidate at the Developmental Cognitive Neuroscience Lab (labNCd)
Center for Mind, Brain and Behaviour (CIMCYC)
University of Granada (UGR)
Granada, Spain

Description:
Function to compute permutation tests, being able to define the number
of iterations and confidence intervals.

In this example we can use pandas Series as Scipy.Stats spearmanr function
allows to work with this structure type, and also allows to choose a nan
policy. If we want to work with Numpy array will it be necessary to drop nan
values from both array so they have the same length.

Important:
Code taken and modified from Statistical Thinking in Python (Part 2) - Datacamp
course - by Justin Bois (Lecturer at the California Institute of Technology).
"""

# =============================================================================
# IMPORT LIBRARIES
# =============================================================================

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# =============================================================================
# PERFORM PERMUTATION
# =============================================================================


def draw_perm_reps_spearman(data_1, data_2, func, iterations=1000, ci=95):
    
    """
    Generate multiple permutation replicates.

    Important:
    Code taken and modified from Statistical Thinking in Python (Part 2) - Datacamp
    course - by Justin Bois (Lecturer at the California Institute of Technology).

    Input:
        data_1: array or Series of data (data set 1).
        data_2: array or Series of data (data set 2).
        func: funtion to apply to the permutes samples.
        iterations (int): number of iterations. Default = 1000.
        ci (int): percentage of confidence intervals. Default 95%.

    Output:
        p_value: statistical p-value.
        confidence_intervals: confidence intervals.

    Notes:
        To compute the p-value, compare every permutation value with the empirical value
        (the percentage of simulations where the simulated statistic was more extreme,
        towards the alternative hypothesis) than the observed empirical value.
        Mark as a boolean if the value of the permuted sample is higher or equal to the
        empirical value. We sum the boolean True.
    """

    def permutation_sample(data_1, data_2):

        """
        Generate a permutation sample from two data sets.
        Permutation: random reordering of entries in an array.
        It is the heart of simulating a null hypothesis where
        we assume two quantiles are identically distributed.

        Concatenate data using Numpy concatenate.

        Input:
            data_1: array or Series of data (data set 1).
            data_2: array or Series of data (data set 2).

        Output:
            perm_sample_1: sample permuted with the length of data_1.
            perm_sample_2: sample permuted with the length of the rest
                           of the permuted_data.
        """

        # Concatenate the data sets: data
        data = np.concatenate((data_1, data_2))

        # Permute the concatenated array: permuted_data
        permuted_data = np.random.permutation(data)

        # Split the permuted array into two: perm_sample_1, perm_sample_2
        perm_sample_1 = permuted_data[:len(data_1)]
        perm_sample_2 = permuted_data[len(data_1):]

        return perm_sample_1, perm_sample_2

    empirical_r, empirical_p = func(data_1, data_2)

    # Initialize array of replicates: perm_replicates
    perm_replicates = np.empty(iterations)

    for i in range(iterations):
        # Generate permutation sample (nested function)
        perm_sample_1, perm_sample_2 = permutation_sample(data_1, data_2)

        # Compute the test statistic. Change if needed.
        spearman_r_empirical, p_value_empirical = func(perm_sample_1, perm_sample_2)

        # Add to permutes results
        perm_replicates[i] = spearman_r_empirical

    # Chose percentiles based on the percentage of confidence intervals
    if ci == 95:
        percentiles = [2.5, 97.5]
    elif ci == 99:
        percentiles = [0.5, 99.5]

    # Compute confidence intervals
    confidence_intervals = np.percentile(perm_replicates, percentiles)
    # Compute p-value
    p_value = np.sum(perm_replicates >= empirical_r) / len(perm_replicates)

    return perm_replicates, p_value, confidence_intervals


# =============================================================================
# PLOT PERMUTATION REPLICATES
# =============================================================================


def plot_permutation_replicates(empirical_test_statistic, perm_replicates, bins):
    
    """
    Plot normed histogram of permutation replicates.

    Input:
        empirical_test_statistic: empirical value of the test statistic computed with data
                                  of the original dataset.
        perm_replicates: array with all the permutation replicates of the test statistic.
        bins (int): number of bins for the histogram.

    Output:
        Normed histogram with vertical line set at the value of the empirical test
        statistic. Legend also added with the value of the empirical test statistic.
    """

    # plot permutation replicates
    plt.hist(perm_replicates, density=True, bins=bins)
    # set vertical line to empirical test statistic value
    plt.axvline(x=empirical_corr, label='line at x = {}'.format(empirical_test_statistic), linestyle='--',
                color='black')
    plt.legend()
    plt.show()


# =============================================================================
# DEFINE FUNCTION TO PERFORM TEST IF NEEDED
# =============================================================================


def ecdf(data):
    
    """
    Compute ECDF for a one-dimensional array of measurements.
    """

    # Number of data points: n
    n = len(data)

    # x-data for the ECDF: x
    x = np.sort(data)

    # y-data for the ECDF: y
    y = np.arange(1, len(data) + 1) / n

    return x, y


def spearman_r(data_1, data_2):
    
    """
    Compute Spearman rank correlation coefficient and p-value between
    two arrays.

    Input:
        data_1: array or Series of data (data set 1).
        data_2: array or Series of data (data set 2).

    Output:
        corr_coef: correlation coefficient.
        p_value: statistical p-value.
    """

    corr_coef, p_value = stats.spearmanr(data_1, data_2, nan_policy='omit')

    return corr_coef, p_value


# =============================================================================
# EXAMPLE
# =============================================================================


# perform permutation
perm_replicates, p_value, confidence_intervals = draw_perm_reps_spearman(DataFrame['column_name_1'], 
                                                                         DataFrame['column_name_2'], 
                                                                         spearman_r, 10000, 95)
# compute empirical test statistic
empirical_test_statistic, empirical_p_value = spearman_r(DataFrame['column_1'], DataFrame['column_2'])
# plot permutation replicates
plot_permutation_replicates(empirical_test_statistic, perm_replicates, bins)
