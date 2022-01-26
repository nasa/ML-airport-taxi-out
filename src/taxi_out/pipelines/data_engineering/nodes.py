# Copyright 2020 QuantumBlack Visual Analytics Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND
# NONINFRINGEMENT. IN NO EVENT WILL THE LICENSOR OR OTHER CONTRIBUTORS
# BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF, OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# The QuantumBlack Visual Analytics Limited ("QuantumBlack") name and logo
# (either separately or in combination, "QuantumBlack Trademarks") are
# trademarks of QuantumBlack. The License does not grant you any right or
# license to the QuantumBlack Trademarks. You may not use the QuantumBlack
# Trademarks or any confusingly similar mark as a trademark for your product,
# or use the QuantumBlack Trademarks in any other manner that might cause
# confusion in the marketplace, including but not limited to in advertising,
# on websites, or on software.
#
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This is a boilerplate pipeline 'data_engineering'
generated using Kedro 0.16.2
"""

"""General purpose (impeded or unimpeded) data engineering nodes
"""

from typing import Any, Dict, Union

import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import KFold


def add_train_test_group(
    data_in: pd.DataFrame,
    test_size: float,
    random_seed: int,
    all_params: Dict[str, Any],
) -> pd.DataFrame:

    data = data_in.copy()
    log = logging.getLogger(__name__)
    # Set random seed
    if random_seed == 'RANDOM':
        random_seed = int(np.random.uniform()*100000)
    np.random.seed(random_seed)

    # Check if one wants multiple splits to do uncertainty calculation
    if ('cross_val' in all_params) :
        if (not(isinstance(all_params['cross_val'], type(None)))) and ('algo' in all_params['cross_val']) :
            algo = all_params['cross_val']['algo']
            if (algo == 'kfold') :
                if ('n_cv' in all_params['cross_val']) :
                    n_cv = all_params['cross_val']['n_cv']
                    if (np.abs(n_cv*test_size-1) > 0.01) :
                        log.warning(f'inconsistent test_size {test_size} with number of kfold {n_cv}, using n_cv value')
                    kf = KFold(n_splits=n_cv, shuffle=True, random_state=random_seed)
                    splits = list(kf.split(data))
                    # Select "main" model
                    data['group'] = 'train'
                    data['group'].iloc[splits[0][1]] = 'test'
                    for k in range(n_cv-1) :
                        data['test_'+str(k)] = False
                        data['test_'+str(k)].iloc[splits[k+1][1]] = True
                    return data    
                else :
                    log.warning('CV algo set to kfold but n_cv not set, use single default sampling')
            if (algo == 'montecarlo') :    
                if ('n_cv' in all_params['cross_val']) :
                    data['group'] = data.apply(
                        lambda row: 'test' if np.random.uniform() < test_size else 'train',
                        axis=1,
                    )
                    n_cv = all_params['cross_val']['n_cv']
                    for k in range(n_cv-1) :
                        data['test_'+str(k)] = data.apply(
                            lambda row: True if np.random.uniform() < test_size else False,
                            axis=1,
                        )
                    return data    
                else :
                    log.warning('CV algo set to montecarlo but n_cv not set, use single default sampling')
            else :
                log.warning(f'algo set to {algo} but function only takes kfold or montecarlo, default to single sampling')
        else :    
            log.warning('cross val section does not contain an algo, default to single sampling')   

    ############ Default simple train/test split :                
    # Apply group
    data['group'] = data.apply(
        lambda row: 'test' if np.random.uniform() < test_size else 'train',
        axis=1,
    )
       
    return data

def train_test_group_logging(
    data: pd.DataFrame,
):
    log = logging.getLogger(__name__)

    log.info('train group has {} instances'.format(sum(data.group == 'train')))
    log.info('test group has {} instances'.format(sum(data.group == 'test')))


def set_index(
    data: pd.DataFrame,
    new_index="gufi",
) -> pd.DataFrame:
    data = data.sort_values(by=['gufi'],ascending=[1]).reset_index(drop=True)
    data.set_index(new_index, inplace=True)

    return data


def keep_first_of_duplicate_index(
    data: pd.DataFrame,
) -> pd.DataFrame:
    data.loc[~data.index.duplicated(keep='first')]

    return data


def keep_second_of_duplicate_gufi(
        data: pd.DataFrame,
        ) -> pd.DataFrame:
    data_out = data.loc[~data.gufi.duplicated(keep='first')]
    return data_out



def compute_total_taxi_time(
    data: pd.DataFrame,
) -> pd.DataFrame:
    data['actual_departure_total_taxi_time'] = \
        (
            data.departure_runway_actual_time -
            data.departure_stand_actual_time
        ).dt.total_seconds()

    return data


def left_join_on_index(
    data_0: pd.DataFrame,
    data_1: pd.DataFrame,
) -> pd.DataFrame:
    return data_0.join(data_1)


def replace_runway_actuals(
        data: pd.DataFrame, data_runway_actuals: pd.DataFrame
) -> pd.DataFrame:
    # Replace arrival/departure runway, on times info at selected airport (arrival/departure times at other airports not modified)

    log = logging.getLogger(__name__)
    if (len(data_runway_actuals) == 0) :
        log.warning('No actual runways data, skipping the data update')
        return data
    
    data_runway_actuals['isdeparture'] = data_runway_actuals['departure_runway_actual'].isna() == False
    data_runway_actuals['isarrival'] = data_runway_actuals['arrival_runway_actual'].isna() == False

    on_fields = ['gufi', 'isarrival', 'isdeparture']
    update_fields = [v for v in list(data_runway_actuals) if v not in on_fields]
    new_suffix = '_new'
    data_merged = pd.merge(data, data_runway_actuals, on=on_fields, how='left', suffixes=['', new_suffix], sort=False)

    for field in update_fields:
        bIndex = data_merged[field + new_suffix].isna() == False
        data_merged.loc[bIndex, field] = data_merged.loc[bIndex, field + new_suffix]

    data_merged.drop(columns=[v + new_suffix for v in update_fields], inplace=True)

    # Update derived fields
    data_merged['actual_departure_ama_taxi_time'] = (data_merged['departure_runway_actual_time'] - data_merged['departure_movement_area_actual_time']).astype('timedelta64[s]')
    data_merged['actual_departure_full_taxi_time'] = (data_merged['departure_runway_actual_time'] - data_merged['departure_stand_actual_time']).astype('timedelta64[s]')

    # patch to make sure all departures are runways :
    data_merged['arrival_runway_actual'] = data_merged['arrival_runway_actual'].astype(str)
    data_merged['departure_runway_actual'] = data_merged['departure_runway_actual'].astype(str)
    
    # make sure there is not trailing .0 in the name of the runways
    # because of conversion from float to string, remove 0 at the beginning
    # of the name as well
    data_merged['arrival_runway_actual'] = data_merged['arrival_runway_actual'].str.replace(r'\.0$','').str.replace(r'^0+','')
    data_merged['departure_runway_actual'] = data_merged['departure_runway_actual'].str.replace(r'\.0$','').str.replace(r'^0+','')

    # need to put back NaN as NaN and not string to avoid keeping them later on :
    data_merged.loc[data_merged['arrival_runway_actual'] == 'nan', 'arrival_runway_actual'] = np.nan
    data_merged.loc[data_merged['departure_runway_actual'] == 'nan', 'departure_runway_actual'] = np.nan

    return data_merged

def merge_STBO(data: pd.DataFrame, data_STBO: pd.DataFrame
) -> pd.DataFrame:

    log = logging.getLogger(__name__)

    if data_STBO.empty:
        log.info('No STBO data available.')
        return data
    else:
        on_fields = ['acid', 'isarrival', 'isdeparture', 'departure_stand_initial_time', 'departure_aerodrome_icao_name', 'arrival_aerodrome_icao_name']
        new_suffix = '_ffs'

        merged_STBO_data = pd.merge(data, data_STBO, on = on_fields, how = 'left', suffixes =['', new_suffix])

        log.info('{:.1f}% of flights have STBO data elements available'.format(
        (merged_STBO_data['gufi_ffs'].notnull().sum()/len(merged_STBO_data))*100))
    
        return merged_STBO_data

######### Bunch of filters :


def apply_filter_only_departures(
    data: pd.DataFrame,
) -> pd.DataFrame:
    initial_row_count = data.shape[0]

    data = data[data.isdeparture]

    final_row_count = data.shape[0]

    log = logging.getLogger(__name__)
    log.info('Kept {:.1f}% of flights when filtering to keep only departures'.format(
        (final_row_count/initial_row_count)*100
    ))

    return data

def apply_filter_null_times(
        data: pd.DataFrame,
        time_list : list = [], 
) -> pd.DataFrame:
    initial_row_count = data.shape[0]

    null_times = pd.Series(data=[False]*len(data),index=data.index)
    for column in time_list :
        null_times = (null_times | pd.isna(data[column]))
    data = data.drop(data.index[null_times])

    final_row_count = data.shape[0]

    log = logging.getLogger(__name__)
    log.info('Kept {:.1f}% of flights when filtering to keep only departures with non-nulls for all required actual times'.format(
        (final_row_count/initial_row_count)*100
    ))

    return data



def apply_filter_req_dep_stand_and_runway(
    data_in: pd.DataFrame,
    model_params: Dict[str, Any],
    freight_airlines: list = [],
) -> pd.DataFrame:

    data  = data_in.copy()
    initial_row_count = data.shape[0]
    log = logging.getLogger(__name__)

    # Remove rows with no stand
    # if stand is in the core features
    # Temporary remove it to let Fedex, UPS go through, maybe I should only pass it if it is Fedex or UPS
    if ("departure_stand_actual" in model_params["features_core"]) :
        data = data[
            data.departure_stand_actual.notnull()
        ]
        interim_row_count = data.shape[0]
        log.info(('Kept {:.1f}% of departures when filtering'+
                 'to keep only departures with non-null'+
                 'departure stand').format(
                     (interim_row_count/initial_row_count)*100)
                 )
    else :
        if ((freight_airlines == None) or (len(freight_airlines) == 0)) :
            log.warning('Departure Stand Not Filtered for Null value because not in core features')
            interim_row_count = data.shape[0]
        else :
            # only pass the null value if on the freight airlines list
            data = data[
                data.departure_stand_actual.notnull() | data.carrier.isin(freight_airlines)
                ]
            interim_row_count = data.shape[0]
            log.warning(('Kept {:.1f}% of departures when filtering'+
                         'to keep only departures with non-null'+
                         'departure stand (freight not filtered)').format(
                             (interim_row_count/initial_row_count)*100)
                        )


    # Remove rows with no runway if they are core parameters :
    if ("departure_runway_actual" in model_params["features_core"]) :    
        data = data[
            data.departure_runway_actual.notnull()
        ]    
        final_row_count = data.shape[0]
        log.info(('Kept {:.1f}% of departures when filtering'+
                 'to keep only departures with non-null'+
                 'departure runway').format(
                     (final_row_count/interim_row_count)*100)
                 )
    else :
        if ((freight_airlines == None) or (len(freight_airlines) == 0)) :
            log.warning('Departure Runway Not Filtered for Null value because not in core features')
        else :
            # only pass the null value if on the freight airlines list
            data = data[
                data.departure_runway_actual.notnull() | data.carrier.isin(freight_airlines)
                ]
            final_row_count = data.shape[0]
            log.info(('Kept {:.1f}% of departures when filtering'+
                      'to keep only departures with non-null'+
                      'departure runway (freight not filtered)').format(
                          (final_row_count/interim_row_count)*100)
                     )
        
    return data




def _apply_filter_req_niqr_dep_any_taxi_times(
        data_in: pd.DataFrame,
        niqr : Union [float,list],
        target_col : str,
        ) -> pd.DataFrame:

    log = logging.getLogger(__name__)
    initial_row_count = data_in.shape[0]

    if (niqr == -1) :
        log.info('NIQR set to -1, skipping the req/niqr and > 0.0 filter on {}'.format(target_col))
        return data_in

    if (not(target_col in data_in)) :
        log.info('{} is not in the data so skipping its req/niqr filtering'.format(target_col))
        return data_in
    
    # Remove rows if 0 or negative total or AMA or ramp taxi times
    data = data_in[
        data_in[target_col] > 0.0
    ].copy()

    final_row_count = data.shape[0]

    log.info('Kept {:.1f}% of departures when filtering to keep only departures with valid {} taxi time'.format(
        (final_row_count/initial_row_count)*100, target_col
    ))

    # Remove rows if value outside median +/- niqr * IQR if niqr is a float
    # Remove rows outside percentile_low, percentile_high if niqr is a list
    initial_row_count = data.shape[0]
    if isinstance(niqr, float) :
        median = data[target_col].median()
        iqr = data[target_col].quantile(0.75) - data[target_col].quantile(0.25)
        data = data[
            (data[target_col] > median-niqr*iqr) &
            (data[target_col] < median+niqr*iqr)
        ]
    elif isinstance(niqr, list) :
        if (len(niqr) == 2) :
            print('NIQR : ',niqr)
            range_value = data[target_col].quantile(np.array(niqr).astype(float)).values
            print('range_value : ',range_value) 
            data = data[
                (data[target_col] > range_value[0]) &
                (data[target_col] < range_value[1])
            ]
        else :
            log.error('A list NIQR needs to be a list of two elements : low_percentile, high_percentile')
    else :
        log.error('NIQR needs to be a float or a 2-element list of floats')
        
    final_row_count = data.shape[0]

    log.info('Kept {:.1f}% of departures when filtering to keep only departures with value within {} +-IQR/QUANTILE for {}'.format(
        (final_row_count/initial_row_count)*100,", ".join(np.array(niqr).flatten().astype(str)),target_col
    ))
    
    return data

def apply_filter_req_niqr_dep_ramp_taxi_times(
    data: pd.DataFrame,
    niqr_ramp_taxi: float,
) -> pd.DataFrame:
    return _apply_filter_req_niqr_dep_any_taxi_times(data, niqr_ramp_taxi, 'actual_departure_ramp_taxi_time')


def apply_filter_req_niqr_dep_full_taxi_times(
    data: pd.DataFrame,
    niqr_full_taxi: float,
) -> pd.DataFrame:
    return _apply_filter_req_niqr_dep_any_taxi_times(data, niqr_full_taxi, 'actual_departure_full_taxi_time')


def apply_filter_req_niqr_dep_ama_taxi_times(
    data: pd.DataFrame,
    niqr_full_taxi: float,
) -> pd.DataFrame:
    return _apply_filter_req_niqr_dep_any_taxi_times(data, niqr_full_taxi, 'actual_departure_ama_taxi_time')



def _apply_filter_surface_counts(
        data_in: pd.DataFrame,
        upperlimits : Dict[str, Any],
) -> pd.DataFrame:

    log = logging.getLogger(__name__)
    data = data_in.copy()
    
    if any(map(lambda x : x <= 0, upperlimits.values())) :
        log.info('An upperlimit is smaller or equal to 0, skipping the surface count filter, upperlimits : {}'.format(upperlimits))
        return data
    filter_ct = pd.Series(True,index=data.index)
    initial_row_count = data.shape[0]
    for key_up, value_up in upperlimits.items():
        if (value_up < 1.0) :
            # assume it is a percentile upper limit
            value_up_num = data[key_up].quantile(value_up)
            filter_ct = filter_ct & (data[key_up] < value_up_num)
        else :
            # assume it a number of aircrafts upper limit
            value_up_num = value_up
            filter_ct = filter_ct & (data[key_up] < value_up_num)
        final_row_count = filter_ct.sum()
        log.info('Kept {:.1f}% of departures when filtering to keep only departures with this count filter : {} {} {}'.format((final_row_count/initial_row_count)*100, key_up, value_up, value_up_num))
    data = data[filter_ct].copy()
    
    return data



def apply_filter_surface_counts_full(
        data_in: pd.DataFrame,
        params : Dict[str, Any],
) -> pd.DataFrame:

    log = logging.getLogger(__name__)
    if ('upperlimit_counts' in params) :
        upperlimit_counts = params['upperlimit_counts']
        if ('full' in upperlimit_counts) and (upperlimit_counts['full'] != None) :
            return _apply_filter_surface_counts(data_in, upperlimit_counts['full'])

    log.info('No filtering from surface counts')          
    return data_in    



def apply_filter_surface_counts_ama(
        data_in: pd.DataFrame,
        params : Dict[str, Any],
) -> pd.DataFrame:

    log = logging.getLogger(__name__)
    if ('upperlimit_counts' in params) :
        upperlimit_counts = params['upperlimit_counts']
        if ('AMA' in upperlimit_counts) and (upperlimit_counts['AMA'] != None) :
            return _apply_filter_surface_counts(data_in, upperlimit_counts['AMA'])

    log.info('No filtering from surface counts')          
    return data_in    
    

    
def apply_filter_surface_counts_ramp(
        data_in: pd.DataFrame,
        params : Dict[str, Any],
) -> pd.DataFrame:

    log = logging.getLogger(__name__)
    if ('upperlimit_counts' in params) :
        upperlimit_counts = params['upperlimit_counts']
        if ('ramp' in upperlimit_counts) and (upperlimit_counts['ramp'] != None) :
            return _apply_filter_surface_counts(data_in, upperlimit_counts['ramp'])

    log.info('No filtering from surface counts')          
    return data_in    
     


##################

def join_fraction_speed_gte_threshold_and_filter(
    data: pd.DataFrame,
    data_fraction_speed_gte_threshold: pd.DataFrame,
) -> pd.DataFrame:
    initial_row_count = data.shape[0]

    data = data.join(data_fraction_speed_gte_threshold)

    # don't remove data without fraction speed, if Nan it will be filtered by unimpeded_AMA
    #data = data[data.fraction_speed_gte_threshold.notnull()]

    final_row_count = data.shape[0]

    log = logging.getLogger(__name__)
    log.info('Matched {:.1f}% of departures when joining fraction of taxi-out trajectory with speed greater than or equal to threshold'.format(
        (final_row_count/initial_row_count)*100
    ))

    return data

def calculate_unimpeded_AMA(
    data: pd.DataFrame,
    model_params: Dict[str, Any],
):

    log = logging.getLogger(__name__)
    if ('unimpeded_ama_fraction_gte' in model_params) :
        data['unimpeded_AMA'] = (
            data.fraction_speed_gte_threshold >=
            model_params['unimpeded_ama_fraction_gte']
        )
        log.info('{:.1f}% of departures were unimpeded in the AMA (because fraction of trajectory measurements with speed above threshold >= {})'.format(
            data.unimpeded_AMA.mean()*100,
            model_params['unimpeded_ama_fraction_gte'],
        ))
    else :
        log.warning('unimpeded_ama_fraction_gte is not defined in the parameter file, this filter is skipped')

    return data


def check_and_filter_start_end_dates(
        data: pd.DataFrame,
        global_params: Dict[str, Any],
        ):

    start_time = global_params['start_time']
    end_time = global_params['end_time']

    ref_key = 'departure_runway_actual_time'
    start_data = data[ref_key].min()
    end_data = data[ref_key].max()

    log = logging.getLogger(__name__)

    # Make sure we have enough data :
    delta_start = start_data - start_time
    delta_end = end_time - end_data 
    if ((delta_start > 12*pd.Timedelta('1 hour')) |
        (delta_end > 12*pd.Timedelta('1 hour'))) :
        log.warning(('\033[0;31mThe requested time range is not well covered by \n'+
        'the csv data : csv data range : {} {} ; \n'+
        'requested data range : {} {} \033[0m').format(
            start_data,end_data,start_time,end_time))

    filter_range = (data[ref_key] > start_time) & \
        (data[ref_key] <= end_time)

    num_sample = filter_range.sum()

    if (num_sample < 1) :
        log.error(('No valid data in the requested time range : '+ 
        'the csv data : csv data range : {} {}; '+
        'requested data range : {} {}').format(
            start_data,end_data,
            start_time,end_time))
        raise(TypeError(('No valid data in the requested time range : '+
        'the csv data : csv data range : {} {}; '+
        'requested data range : {} {}').format(
            start_data,end_data,
            start_time,end_time)))

    data_out = data.copy()
    data_out = data_out[filter_range]
        
    log.info('{:.1f}% of departures were selected in the date range'.format(
        len(data_out)/len(data)*100.0
    ))

                        
    return data_out
