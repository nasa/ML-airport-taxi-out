import pytest
import pandas as pd
import numpy as np
from kedro.config import TemplatedConfigLoader



@pytest.fixture(scope='module')
def config():
    """ 
    Load Names of Files in a Configuration
    """
    return TemplatedConfigLoader(['conf/base', 'conf/local'], globals_pattern='globals*')


@pytest.fixture(scope='module')
def parameters(config):
    """
    Extract Parameters from parameters.yml stored in the configuration
    """
    return config.get('parameters.yml')


@pytest.fixture(scope='module')
def data_fit() :
    """
    Define some basic dataframes for testing the fit of the pipeline
    """

    data=[]
    carriers = ['UAL','DAL','SKW','RPA','AAL']
    runways = ['36R','18L','36C','18R','17L']
    # need enough stand to create stand clusters nstand >= ncluster in the parameters.yml 
    stands = ['C10','A5','B8','A2','D1','D13','E4','C3','E15','B2','A10','C1']
    aircrafts = ['A320','B738','CRJ9','MD90','B737']
    groups = ['test','train']
    arr_run_AMA = [1,5,10,20,30]
    dep_AMA_run = [1,5,10,20,30]
    arr_AMA_gate = [1,5,10,20,30]
    dep_gate_AMA = [1,5,10,20,30]
    timestamps = pd.date_range(start='06-01-2020 08:00:00',end='06-30-2020 08:00:00',freq='5 min')
    # need enough if using the gate_cluster
    ndata = 5000
    taxi_time = 500.
    # in case, one need not quite identical target
    epsilon = 0.0
    # define count variables consistent with each other
    arr_runway_AMA_count = np.random.choice(arr_run_AMA,ndata)
    dep_AMA_runway_count = np.random.choice(dep_AMA_run,ndata)
    AMA_gate_count =  np.random.choice(arr_AMA_gate,ndata)
    dep_stand_AMA_count = np.random.choice(dep_gate_AMA,ndata)
    total_arrivals_on_surface = arr_runway_AMA_count + AMA_gate_count
    total_departures_on_surface = dep_AMA_runway_count + dep_stand_AMA_count
    total_flights_on_surface = total_arrivals_on_surface + total_departures_on_surface
    # Case #1 all taxis are the sames :
    data.append(
        pd.DataFrame({'carrier':np.random.choice(carriers,ndata),
                      'departure_stand_actual':np.random.choice(stands,ndata),
                      'departure_runway_actual':np.random.choice(runways,ndata),
                      'aircraft_type':np.random.choice(aircrafts,ndata),
                      'unimpeded_AMA':np.array([True]*ndata),
                      'group':np.random.choice(groups,ndata,p=[0.2,0.8]),
                      'arr_runway_AMA_count':arr_runway_AMA_count,
                      'dep_AMA_runway_count':dep_AMA_runway_count,
                      'AMA_gate_count':AMA_gate_count,
                      'dep_stand_AMA_count':dep_stand_AMA_count,
                      'total_arrivals_on_surface':total_arrivals_on_surface,
                      'total_departures_on_surface':total_departures_on_surface,
                      'total_flights_on_surface':total_flights_on_surface,
                      'departure_stand_airline_time':np.random.choice(timestamps,ndata),
                      'target':np.ones(ndata)*taxi_time+np.random.randn(ndata)*epsilon,
                      }
                     )
    )
    # Case #2 with a different order for the columns
    reorder_keys = np.random.choice(data[0].keys(),size=len(data[0].keys()),replace=False)
    data_deordered = data[0][reorder_keys].copy()
    data.append(data_deordered)       
                      

    return data    
                      

@pytest.fixture(scope='module')
def data_pred() :
    """
    Define a basic dataframe for testing prediction of the pipelines
    """
    return None
