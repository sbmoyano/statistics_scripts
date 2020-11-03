# -*- coding: utf-8 -*-
"""
Created on Tus Nov 03 09:08:00 2020

@author: Sebastián Moyano

PhD Candidate at the Developmental Cognitive Neuroscience Lab (labNCd)
Center for Mind, Brain and Behaviour (CIMCYC)
University of Granada (UGR)
Granada, Spain

Description:
Function to perform bootstrapping in one dataset or two. These functions
can be adapted to use it with multiple datasets. It is though to use it
with a pandas DataFrame.

I recommend to create an independent function to call when applying a test
statistic and handle NaN values in that function.

Important:
Code taken and modified from Statistical Thinking in Python (Part 2) - Datacamp
course - by Justin Bois (Lecturer at the California Institute of Technology).
"""

# =============================================================================
# IMPORT LIBRARIES
# =============================================================================

import pandas as pd
import scipy.stats as stats
import numpy as np
import pingouin as pg
import matplotlib.pyplot as plt

# =============================================================================
# BOOTSTRAP (SINGLE DATA SET)
# =============================================================================


def bootstrap(data, func, iterations=1, ci=95):
    """
    Perform pairs bootstrap for a single statistic to one set of data.
    Generates an array of indices with the same length of x. Iterate and
    in each iteration select a set of indices randomly from bs_inds,
    select the values for each indices from the original data, apply the
    function to the bootstrapped sample and store it in bs_replicates.

    I recommend to create an independent function to perform test statistic
    (considering how to handle nan values).

    Important:
    Code taken and modified from Statistical Thinking in Python (Part 2) - Datacamp
    course - by Justin Bois (Lecturer at the California Institute of Technology).

    Input:
        data: array of data.
        func: function to apply.
        iterations (int): number of iterations. Default set to 1.
        ci (int): percentage of confidence intervals. Default set to 95%.

    Output:
        bs_replicates: array of results after applying the test statistic
                       to a different bootstrapped sample in each iteration.
        confidence_intervals: confidence intervals for bootstrapped sample.

    Notes:
        Bootstrap: resampled data to perform statistical inference.
        Bootstrap sample: resampled array of data.
        Bootstrap replicate: value of the test statistic computed from the
                             bootstrapped sample.
    """

    # Chose percentiles based on the percentage of confidence intervals
    if ci == 95:
        percentiles = [2.5, 97.5]
    elif ci == 99:
        percentiles = [0.5, 99.5]

    # array to store the bootstrap sample
    bs_replicates = np.empty(iterations)

    # generate the bootstrap sample
    for i in range(iterations):
        # resample from the array of data randomly and apply function
        bs_replicates[i] = func(np.random.choice(data, size=len(data)))

    # plot the histogram of the replicates
    plt.hist(bs_replicates, bins=50)
    plt.xlabel('x_label')
    plt.ylabel('y_label')
    plt.show()

    # get confidence intervals of the bootstrapped statistic
    confidence_intervals = np.percentile(bs_replicates, percentiles)

    return bs_replicates, confidence_intervals


# =============================================================================
# BOOTSTRAP PAIRS (TWO DATA SETS)
# =============================================================================


def bootstrap_pairs(data_1, data_2, func, iterations=1, ci=95):

    """
    Perform pairs bootstrap for a single statistic to two sets of data.
    Generates an array of indices with the same length of x. Iterate and
    in each iteration select a set of indices randomly from bs_inds,
    select the values for each indices from the original data, apply the
    function to the bootstrapped sample and store it in bs_replicates.

    I recommend to create an independent function to perform test statistic
    (considering how to handle nan values).

    Important:
    Code taken and modified from Statistical Thinking in Python (Part 2) - Datacamp
    course - by Justin Bois (Lecturer at the California Institute of Technology).

    Input:
        data_1: array of data.
        data_2: array of data.
        func: function to apply.
        iterations (int): number of iterations. Default set to 1.
        ci (int): percentage of confidence intervals. Default set to 95%.

    Output:
        bs_replicates: array of results after applying the test statistic
                       to a different bootstrapped sample in each iteration.
        confidence_intervals: confidence intervals for bootstrapped sample.
    """

    # Set up array of indices to sample from: inds
    inds = np.arange(len(data_1))

    # Initialize replicates: bs_replicates
    bs_replicates = np.empty(iterations)

    # Chose percentiles based on the percentage of confidence intervals
    if ci == 95:
        percentiles = [2.5, 97.5]
    elif ci == 99:
        percentiles = [0.5, 99.5]

    # Generate replicates
    for i in range(iterations):
        # Chose randomly as many indices as the length of indices
        bs_inds = np.random.choice(inds, size=len(inds))
        # Select from x and y the chosen indices (from DataFrame)
        bs_x, bs_y = data_1.iloc[bs_inds], data_2.iloc[bs_inds]
        # Compute the test statistic. Change if needed.
        spearman_r_empirical, p_value_empirical = func(bs_x, bs_y)
        # Add to permuted results
        bs_replicates[i] = spearman_r_empirical

    # Plot the histogram of the replicates
    plt.hist(bs_replicates, bins=50)
    plt.xlabel('x_label')
    plt.ylabel('y_label')
    plt.show()

    # Get confidence intervals
    confidence_intervals = np.percentile(bs_replicates, percentiles)

    return bs_replicates, confidence_intervals


# =============================================================================
# BOOSTRAP (LINEAR REGRESSION)
# =============================================================================


def boostrap_pairs_lin_reg(data_1, data_2, func, iterations=1, ci=95):

    """
    Perform pairs bootstrap for linear regression.
    Pairs bootstrap:
    1. Resample data in pairs.
    2. Compute slope and intercept from resampled data.
    3. Each slope and intercept is a boostrap replicate.
    4. Compute CI from percentiles of bootstrap replicates.

    Important:
    Code taken and modified from Statistical Thinking in Python (Part 2) - Datacamp
    course - by Justin Bois (Lecturer at the California Institute of Technology).

    Input:
        data_1: array of data.
        data_2: array of data.
        func: function to apply.
        iterations (int): number of iterations. Default set to 1.
        ci (int): percentage of confidence intervals. Default set to 95%.

    Output:
        bs_slope_reps: boostrapped replicates from the slope.
        bs_intercept_reps: boostrapped replicates from the intercept.
        confidence_intervals: confidence intervals for bootstrapped sample
                              from the slope.
    """

    # Set up array of indices to sample from: inds
    inds = np.arange(0, len(data_1))

    # Initialize replicates: bs_slope_reps, bs_intercept_reps
    bs_slope_reps = np.empty(iterations)
    bs_intercept_reps = np.empty(iterations)

    # Chose percentiles based on the percentage of confidence intervals
    if ci == 95:
        percentiles = [2.5, 97.5]
    elif ci == 99:
        percentiles = [0.5, 99.5]

    # Generate replicates
    for i in range(iterations):
        # Chose randomly as many indices as the length of indices
        bs_inds = np.random.choice(inds, size=len(inds))
        # Select from x and y the choosen indices
        bs_x, bs_y = data_1.iloc[bs_inds], data_2.iloc[bs_inds]
        # Apply regression to the boostrapped samples
        bs_slope_reps[i], bs_intercept_reps[i] = func(bs_x, bs_y)

    # Compute confidence intervals (slope)
    confidence_intervals = np.percentile(bs_slope_reps, percentiles)

    return bs_slope_reps, bs_intercept_reps, confidence_intervals


# =============================================================================
# DEFINE FUNCTION TO PERFORM TEST IF NEEDED
# =============================================================================


def plot_bs_reg(bs_slope_reps, bs_intercept_reps, data_1, data_2, n_reg_lines):

    """
    Plot multiple regression lines from a bootstrapped sample.

    Input:
        bs_slope_reps: bootstrapped slope.
        bs_intercept_reps: bootstrapped intercept.
        data_1: original x data.
        data_2: original y data.
        n_reg_lines: number of regression lines to be plotted.
    Output:
        Scatter plot with multiple regression lines fitted form bootstrapped
        sample.
    """

    # Plot the regression lines
    for i in range(n_reg_lines):
        plt.plot(data_1, bs_slope_reps[i] * data_1 + bs_intercept_reps[i], linewidth=0.5, alpha=0.2, color='red')

    # Plot the empirical data
    plt.plot(data_1, data_2, marker='.', linestyle='none')
    plt.xlabel('x_label')
    plt.ylabel('y_label')
    plt.show()


def ecdf(data):

    """
    Compute ECDF for a one-dimensional array of measurements.
    """

    # Number of data points: n
    n = len(data)

    # x-data for the ECDF: x
    x = np.sort(data)

    # y-data for the ECDF: y
    y = np.arange(1, len(data)+1) / n

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


def lin_reg(data_1, data_2):

    """
    Perform linear regression removing NaN values listwise.

    Input:
        data_1: array or Series of data (data set 1).
        data_2: array or Series of data (data set 2).

    Output:
        slope: value for the slope.
        intercept: value for the intercept.
    """

    # Remove NaN values listwise
    df_results = pg.linear_regression(data_1, data_2, remove_na=True)
    # Extract slope and intercept
    slope, intercept = df_results['coef'].iloc[1], df_results['coef'].iloc[0]

    return slope, intercept


# =============================================================================
# EXAMPLE
# =============================================================================


bs, ci = bootstrap(DataFrame['column'], np.mean, 10000, 95)
bs, ci = bootstrap_pairs(DataFrame['column_1'], DataFrame['column_2'], spearman_r, 10000, 95)
bs_slope, bs_intercept, ci = boostrap_pairs_lin_reg(DataFrame['column_1'], DataFrame['column_2'], lin_reg, 10000, 95)
plot_bs_reg(bs_slope, bs_intercept, DataFrame['column_1'], DataFrame['column_2'], 100)
