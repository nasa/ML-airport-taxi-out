"""Pipelines for unimpeded taxi-in prediction test
"""
import math
import numbers
import random
from copy import deepcopy
from typing import Any, Dict
import pandas as pd
import mlflow
import numpy as np


def test_missing_core_feature_none(trained_model, data, parameters, init_mlflow):
    """a method for testing whether the model returns a  default value when a
       missing  core features are passed in as none
       """

    # remove core feature
    idx = random_not_null_idx(data)

    all_features = get_detailed_features(data.columns, parameters['inputs'])
    core_features = [f for f in all_features if all_features[f]['is_core']]

    for core_feature in core_features:
        data_copy = deepcopy(data.loc[[idx]])

        data_copy[core_feature] = None
        output = trained_model.predict(data_copy)
        np.testing.assert_equal(trained_model.default_response, output)

        output = trained_model.predict_df(data_copy)
        np.testing.assert_equal(trained_model.default_response, output['pred'].iloc[0])
        assert core_feature in output['error_msg'].iloc[0]

    mlflow.log_metric('test_missing_core_feature_none_passed', 1)
    mlflow.log_metric('tests_passed', 1)


def test_missing_core_feature_nan(trained_model, data, parameters):
    """a method for testing whether the model returns a  default value when a
           missing  core features are passed in as nan.
           """
    # remove core feature
    idx = random_not_null_idx(data)

    all_features = get_detailed_features(data.columns, parameters['inputs'])
    core_features = [f for f in all_features if all_features[f]['is_core']]

    for core_feature in core_features:
        data_copy = deepcopy(data.loc[[idx]])

        data_copy[core_feature] = math.nan
        output = trained_model.predict(data_copy)
        np.testing.assert_equal(trained_model.default_response, output)

        output = trained_model.predict_df(data_copy)
        np.testing.assert_equal(trained_model.default_response, output['pred'].iloc[0])
        assert core_feature in output['error_msg'].iloc[0]

    mlflow.log_metric('test_missing_core_feature_none_passed', 1)
    mlflow.log_metric('tests_passed', 1)


def test_missing_core_feature_emptystring(trained_model, data, parameters):
    """a method for testing whether the model returns a  default value when a
               missing  core features are passed in as empty string.
               """
    # remove core feature
    idx = random_not_null_idx(data)

    all_features = get_detailed_features(data.columns, parameters['inputs'])
    core_features = [f for f in all_features if all_features[f]['is_core']]

    for core_feature in core_features:
        data_copy = deepcopy(data.loc[[idx]])

        data_copy[core_feature] = ''
        output = trained_model.predict(data_copy)
        np.testing.assert_equal(trained_model.default_response, output)

        output = trained_model.predict_df(data_copy)
        np.testing.assert_equal(trained_model.default_response, output['pred'].iloc[0])
        assert core_feature in output['error_msg'].iloc[0]

    mlflow.log_metric('test_core_feature_emptystring_passed', 1)
    mlflow.log_metric('tests_passed', 1)


def test_pipeline_features_order(trained_model, data):
    """a method for testing that the model will provide the same prediction even when changing the column order
    """
    idx = random_not_null_idx(data)

    expected_prediction = trained_model.predict(  data.loc[[idx]])[0]
    # shuffle data
    inverted_data = data[data.columns[::-1]].loc[[idx]]
    actual_prediction = trained_model.predict(inverted_data)[0]

    np.testing.assert_equal(expected_prediction, actual_prediction )
    #assert (expected_prediction['value'] == actual_prediction['value']).all()

    mlflow.log_metric('test_feature_order_passed', 1)
    mlflow.log_metric('tests_passed', 1)


def test_features_type(trained_model, data, parameters):
    """a method for testing that the model will return the default value if a wrong type is passed in.
       """
    idx = random_not_null_idx(data)

    all_features = get_detailed_features(data.columns, parameters['inputs'])
    core_features = [f for f in all_features if all_features[f]['is_core']]

    for core_feature in core_features:
        for bad_value in get_bad_types(all_features[core_feature]['type'], all_features[core_feature]['constraints']):
            data_copy = deepcopy(data.loc[[idx]])

            data_copy[core_feature] = bad_value
            print("testing: {}({}) with {}".format(core_feature, all_features[core_feature]['type'], bad_value))

            output = trained_model.predict(data_copy)
            np.testing.assert_equal(trained_model.default_response, output)

            output = trained_model.predict_df(data_copy)
            np.testing.assert_equal(trained_model.default_response, output['pred'].iloc[0])
            assert core_feature in output['error_msg'].iloc[0]

    mlflow.log_metric('test_feature_type_passed', 1)
    mlflow.log_metric('tests_passed', 1)


def test_target_type(data, trained_model, parameters, N=10):
    """a method for verifying that the predicted target type is the same as the configured target type.
       """
    target_type = parameters['type']
    target_type_rule = type_rules()[target_type]
    predictions = trained_model.predict(data.head(N))

    for prediction in predictions:
        for val in prediction['value']:
            assert target_type_rule(val)

    mlflow.log_metric('test_target_type_passed', 1)
    mlflow.log_metric('tests_passed', 1)


def random_not_null_idx(data):
    not_null_idx = data.isnull() == False
    return random.choice(data[not_null_idx].index)


def get_detailed_features(
        columns: np.array,
        inputs_params: Dict[str, Any]
) -> Dict[str, Any]:
    # Formats features info and identifies all the features for type series (e.g. wind_15, wind_30, wind_45, etc.)
    features = {}

    for feature in inputs_params:
        if not inputs_params[feature]['series']:
            features[feature] = {}
            features[feature]['type'] = inputs_params[feature]['type']
            features[feature]['encoder'] = inputs_params[feature]['encoder']
            features[feature]['is_core'] = inputs_params[feature].get('core', False)
            features[feature]['original_feature'] = feature
            features[feature]['lat'] = None
            features[feature]['constraints'] = {}
            if 'constraints' in inputs_params[feature]:
                features[feature]['constraints'] = {k: v for k, v in inputs_params[feature]['constraints'].items()}
        else:
            for v in [c for c in columns if feature in c]:
                x = v.split(feature + '_')
                if (len(x) == 2) and x[1].isnumeric():
                    features[v] = {}
                    features[v]['type'] = inputs_params[feature]['type']
                    features[v]['encoder'] = inputs_params[feature]['encoder']
                    features[v]['is_core'] = inputs_params[feature].get('core', False)
                    features[v]['original_feature'] = feature
                    features[v]['lat'] = int(x[-1])
                    features[v]['constraints'] = {}
                    if 'constraints' in inputs_params[feature]:
                        features[v]['constraints'] = {k: v for k, v in inputs_params[feature]['constraints'].items()}

    return features


def type_rules():
    return {
        'categorical': lambda x: isinstance(x, str),
        'numeric': lambda x: isinstance(x, numbers.Number),
        'bool': lambda x: isinstance(x, bool),
        'datetime': lambda x: isinstance(x,np.datetime64) or isinstance(x,pd._libs.tslibs.timestamps.Timestamp)
    }


def get_bad_types(feature_type, constraints):
    if feature_type == 'numeric':
        bad_vals = ['unknown']
        if constraints:
            if 'min' in constraints:
                bad_vals.append(constraints['min'] - 10)
            if 'max' in constraints:
                bad_vals.append(constraints['min'] + 10)
        return bad_vals
    elif feature_type == 'categorical':
        return ['unknown_category', 10, -10]
    elif feature_type == 'bool':
        return ['unknown', 10, -10]
    elif feature_type == 'datetime':
        return ['unknown', 10, -10]
