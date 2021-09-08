"""
error_metrics.py

Michael Bloem
03 February 2020

This file contains functions that compute error metrics.
"""

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error


def mean_absolute_percentage_error(y_true, y_pred):
    # See: https://stats.stackexchange.com/questions/58391/mean-absolute-percentage-error-mape-in-scikit-learn
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def median_absolute_percentage_error(y_true, y_pred):
    return np.median(np.abs((y_true - y_pred) / y_true)) * 100


def percent_within_n(y_true, y_pred, n=1):
    return np.mean(np.abs(y_true-y_pred) < n) * 100


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


# TODO: change code that calls this to dynamically pass in the quantile
# rather than relying on this default
def tilted_loss(y_true, y_pred, quantile=0.2):
    return np.mean(
        np.max(
            np.vstack(
                (
                    quantile*(y_true-y_pred).values,
                    (quantile-1)*(y_true-y_pred).values
                )
            ),
            axis=0
        )
    )


def fraction_less_than_actual(y_true, y_pred):
    return np.mean(y_pred < y_true)

def madres(y_true, y_pred):
    return np.median(np.abs((y_true-y_pred)-np.median(y_true-y_pred)))*1.4826

def medianres(y_true, y_pred):
    return np.median(y_true-y_pred)
 


METRIC_NAME_TO_FUNCTION_DICT = {
    'mean_absolute_error': mean_absolute_error,
    'mean_absolute_percentage_error': mean_absolute_percentage_error,
    'median_absolute_percentage_error': median_absolute_percentage_error,
    'percent_within_n': percent_within_n,
    'rmse': rmse,
    'tilted_loss': tilted_loss,
    'fraction_less_than_actual': fraction_less_than_actual,
    'MAD_residual':madres,
    'median_residual':medianres,
}
