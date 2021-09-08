#!/usr/bin/env python

"""
Functions for computing surface counts for departures

Using the data_services.compute_surface_counts

        Name of the new columns created by 
        compute_arrival_departure_count_at_pushback
        arr_runway_AMA_count
        dep_AMA_runway_count
        AMA_gate_count
        dep_stand_AMA_count
        total_arrivals_on_surface
        total_departures_on_surface
        total_flights_on_surface

"""

from data_services.compute_surface_counts import *

calculation_time = 'departure_stand_actual_time'


def compute_arrival_departure_count_at_pushback(
    data: pd.DataFrame
) -> pd.DataFrame:
    '''
    Rollup function to compute entire set of surface counts

    Parameters
    ----------
    data : pd.DataFrame
        DESCRIPTION.

    Returns
    -------
    DataFrame with surface counts

    '''

    data0 = compute_arrival_count_in_ramp_at_pushback(data)
    data1 = compute_arrival_count_in_ama_at_pushback(data0)
    data2 = compute_arrival_count_on_surface_at_pushback(data1)
    
    data3 = compute_departure_count_in_ramp_at_pushback(data2)
    data4 = compute_departure_count_in_ama_at_pushback(data3)
    data5 = compute_departure_count_on_surface_at_pushback(data4)
    
    data6 = compute_total_count_on_surface_at_pushback(data5)
    
    return data6


def compute_total_count_on_surface_at_pushback(
    data: pd.DataFrame
) -> pd.DataFrame:
    
    # Only interested in counts @ pushback, so grab just the departures
    dat_arrivals = data[['gufi',calculation_time]]\
        [data['isdeparture'] & data[calculation_time].notnull()].copy().\
        sort_values(by=calculation_time).\
        reset_index(drop=True)
            
    departure_count = compute_departure_count_timeseries(data,
                                               'total',
                                               use_original_times=True,
                                               timeout_flights=True,
                                               infer_flight_times=True)
    
    # Merge just the arrival flights on to the counts, using merge_asof
    dat0 = pd.merge_asof(
        dat_arrivals,
        departure_count,
        left_on=calculation_time,
        right_on='timestamp')
    
    arrival_count = compute_arrival_count_timeseries(data,
                                               'total',
                                               use_original_times=True,
                                               timeout_flights=True,
                                               infer_flight_times=True)
    
    # Merge just the arrival flights on to the counts, using merge_asof
    dat1 = pd.merge_asof(
        dat0,
        arrival_count,
        left_on=calculation_time,
        right_on='timestamp')
    
    # Add arrival and departure counts
    dat1['total_flights_on_surface'] = (dat1['total_arrivals_on_surface'] +
                                        dat1['total_departures_on_surface'])
        
    # Merge back to original dataframe
    dat2 = data.merge(
        dat1[["gufi","total_flights_on_surface"]],
        on='gufi',
        how='left')
    
    return dat2

def compute_arrival_count_in_ama_at_pushback(
    data: pd.DataFrame
) -> pd.DataFrame:
    
    # Calculate Arrival Counts in AMA @ pushback
    dat_arrivals = data[['gufi',calculation_time]]\
        [data['isdeparture'] & data[calculation_time].notnull()].copy().\
        sort_values(by=calculation_time).\
        reset_index(drop=True)
            
    arr_ama_count = compute_arrival_count_timeseries(data,
                                                     'ama',
                                                     use_original_times=True,
                                                     timeout_flights=True,
                                                     infer_flight_times=True)
    
    # Merge just the arrival flights on to the counts, using merge_asof
    dat0 = pd.merge_asof(
        dat_arrivals,
        arr_ama_count,
        left_on=calculation_time,
        right_on='timestamp')
        
    # Merge back to original dataframe
    dat1 = data.merge(
        dat0[["gufi","arr_runway_AMA_count"]],
        on='gufi',
        how='left')
    
    return dat1

def compute_arrival_count_in_ramp_at_pushback(
    data: pd.DataFrame
) -> pd.DataFrame:
    
    # Calculate Arrival Counts in AMA @ departures
    dat_arrivals = data[['gufi',calculation_time]]\
        [data['isdeparture'] & data[calculation_time].notnull()].copy().\
        sort_values(by=calculation_time).\
        reset_index(drop=True)
            
    arr_ramp_count = compute_arrival_count_timeseries(data,
                                                      'ramp',
                                                      use_original_times=True,
                                                      timeout_flights=True,
                                                      infer_flight_times=True)
    
    # Merge just the arrival flights on to the counts, using merge_asof
    dat0 = pd.merge_asof(
        dat_arrivals,
        arr_ramp_count,
        left_on=calculation_time,
        right_on='timestamp')
        
    # Merge back to original dataframe
    dat1 = data.merge(
        dat0[["gufi","AMA_gate_count"]],
        on='gufi',
        how='left')
    
    return dat1

def compute_arrival_count_on_surface_at_pushback(
    data: pd.DataFrame
) -> pd.DataFrame:
    
    # Calculate Arrival Counts in AMA @ pushback
    dat_arrivals = data[['gufi',calculation_time]]\
        [data['isdeparture'] & data[calculation_time].notnull()].copy().\
        sort_values(by=calculation_time).\
        reset_index(drop=True)
            
    arr_count = compute_arrival_count_timeseries(data,
                                                 'total',
                                                 use_original_times=True,
                                                 timeout_flights=True,
                                                 infer_flight_times=True)
    
    # Merge just the arrival flights on to the counts, using merge_asof
    dat0 = pd.merge_asof(
        dat_arrivals,
        arr_count,
        left_on=calculation_time,
        right_on='timestamp')
        
    # Merge back to original dataframe
    dat1 = data.merge(
        dat0[["gufi","total_arrivals_on_surface"]],
        on='gufi',
        how='left')
    
    return dat1

def compute_departure_count_in_ama_at_pushback(
    data: pd.DataFrame
) -> pd.DataFrame:
    
    # Only interested in counts @ pushback, so grab just the departures
    dat_arrivals = data[['gufi',calculation_time]]\
        [data['isdeparture'] & data[calculation_time].notnull()].copy().\
        sort_values(by=calculation_time).\
        reset_index(drop=True)
            
    count = compute_departure_count_timeseries(data,
                                               'ama',
                                               use_original_times=True,
                                               timeout_flights=True,
                                               infer_flight_times=True)
    
    # Merge just the arrival flights on to the counts, using merge_asof
    dat0 = pd.merge_asof(
        dat_arrivals,
        count,
        left_on=calculation_time,
        right_on='timestamp')
        
    # Merge back to original dataframe
    dat1 = data.merge(
        dat0[["gufi","dep_AMA_runway_count"]],
        on='gufi',
        how='left')
    
    return dat1

def compute_departure_count_in_ramp_at_pushback(
    data: pd.DataFrame
) -> pd.DataFrame:
    
    # Only interested in counts @ pushback, so grab just the departures
    dat_arrivals = data[['gufi',calculation_time]]\
        [data['isdeparture'] & data[calculation_time].notnull()].copy().\
        sort_values(by=calculation_time).\
        reset_index(drop=True)
            
    count = compute_departure_count_timeseries(data,
                                               'ramp',
                                               use_original_times=True,
                                               timeout_flights=True,
                                               infer_flight_times=True)
    
    # Merge just the arrival flights on to the counts, using merge_asof
    dat0 = pd.merge_asof(
        dat_arrivals,
        count,
        left_on=calculation_time,
        right_on='timestamp')
        
    # Merge back to original dataframe
    dat1 = data.merge(
        dat0[["gufi","dep_stand_AMA_count"]],
        on='gufi',
        how='left')
    
    return dat1

def compute_departure_count_on_surface_at_pushback(
    data: pd.DataFrame
) -> pd.DataFrame:
    
    # Only interested in counts @ pushback, so grab just the departures
    dat_arrivals = data[['gufi',calculation_time]]\
        [data['isdeparture'] & data[calculation_time].notnull()].copy().\
        sort_values(by=calculation_time).\
        reset_index(drop=True)
            
    count = compute_departure_count_timeseries(data,
                                               'total',
                                               use_original_times=True,
                                               timeout_flights=True,
                                               infer_flight_times=True)
    
    # Merge just the arrival flights on to the counts, using merge_asof
    dat0 = pd.merge_asof(
        dat_arrivals,
        count,
        left_on=calculation_time,
        right_on='timestamp')
        
    # Merge back to original dataframe
    dat1 = data.merge(
        dat0[["gufi","total_departures_on_surface"]],
        on='gufi',
        how='left')
    
    return dat1
