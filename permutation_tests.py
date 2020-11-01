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

Important:
Code taken and modified from Statistical Thinking in Python (Part 2) - Datacamp
course - by Justin Bois (Lecturer at the California Institute of Technology).
"""

# =============================================================================
# IMPORT LIBRARIES
# =============================================================================

import numpy as np
import scipy.stats as stats

# =============================================================================
# PERMUTATION
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
        To compute the p-value, compare every permutation value with the empirical value.
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

    return p_value, confidence_intervals

# =============================================================================
# DEFINE FUNCTION TO PERFORM TEST IF NEEDED
# =============================================================================


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


p, ci = draw_perm_reps_spearman(DataFrame['column_name_1'], DataFrame['column_name_2'], spearman_r, 10000, 95)
print(p, ci)
