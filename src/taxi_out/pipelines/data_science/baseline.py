from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
from typing import Any, Dict


class GroupByModel(BaseEstimator, TransformerMixin):

    def __init__(
        self,
        config: Dict[str, Any]
    ) -> None:
        self.baseline_config = config

    def fit(
        self,
        data: pd.DataFrame,
        y: pd.Series
    )  -> object:

        if self.baseline_config['group_by_agg']['metric'] == 'quantile':
            f = lambda x: np.nanquantile(x, self.baseline_config['group_by_agg']['param'])
        else:
            f = lambda x: np.nanmean(x)

        if len(self.baseline_config['group_by_features']) > 0:
            # Average target values grouped by  "group_by_features"
            data1 = data.copy()
            data1['y'] = y

            self.baseline_model = data1.groupby(
                self.baseline_config['group_by_features'], as_index=False)[['y']].agg(f). \
                rename(columns={'y': 'predicted'})
        else:
            self.baseline_model = f(y)
        # set a default in case data not in training
        self.default_value = np.nanmedian(self.baseline_model['predicted'])
 
        return self

    def predict(
        self,
        data: pd.DataFrame,
    ) -> np.array:

        if len(self.baseline_config['group_by_features']) > 0:
            predicted = pd.merge(data, self.baseline_model, how='left',
                                 on=self.baseline_config['group_by_features'])['predicted'].values
        else:
            predicted = self.baseline_model * np.ones(len(data.index))
        # remove Nan from predicted (it is an array)
        predicted[np.where(np.isfinite(predicted) == False)] = self.default_value
            
        return predicted
