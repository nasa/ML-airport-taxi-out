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
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.16.2
"""
from typing import Any, Dict, Union, List

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor

#from sklearn.compose import ColumnTransformer
# new wrapper develp. by Michael to return a DF :
from data_services.sklearn_wrapper import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline as sklearn_Pipeline
import xgboost as xgb
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyClassifier
from sklearn.dummy import DummyRegressor

import time
import logging
import os

import mlflow
from mlflow import sklearn as mlf_sklearn


# extra file that I will put in the same directory as this file
from data_services.stand_cluster_encoder import StandEncoder
from data_services.terminal_encoder import TerminalEncoder

# It seems kedro knows where to find the data_services package
#from data_services.FilterPipeline_mod import FilterPipeline
from data_services.FilterPipeline import FilterPipeline

from data_services.OrderFeatures import OrderFeatures
from data_services.utils import FormatMissingData
from sklearn.preprocessing import FunctionTransformer
from data_services.mlflow_utils import add_environment_specs_to_conda_file


#### metric libraries for model evaluation
from sklearn import metrics
from . import error_metrics
from .error_metrics import METRIC_NAME_TO_FUNCTION_DICT
## distribution artifacts
from . import training_set_distribution  as tsd

#### import code for baseline fit
from .baseline import GroupByModel
from data_services.gate_cluster_encoder import GateClusterEncoder

#### import optimization function
from sklearn.model_selection import GridSearchCV
from copy import deepcopy


#### import hyperopt wrapper
from .hyperopt_wrapper import OptimizerHOP


from sklearn.base import clone as sk_clone
from sklearn.model_selection import ShuffleSplit

#### 'GENERIC' FUNCTIONS USEFUL FOR NODES :

from sklearn.model_selection import KFold


def list_from_nested_dict(d):
  """
  provide a flatten list from a nested dictionary
  """
  def recur_search (keys, value) :
    if isinstance(value, dict):
      for k, v in value.items() :
        for i in recur_search(keys + [k], v): 
          yield i
    else:
      yield (keys, value)
  return recur_search([], d)


def flatten_dict(d, sep = '_') :
  """
     flatten a dictionary with specific "sep" separator from nested dictionary
  """
  return dict( (sep.join(ks), v) for ks, v in list_from_nested_dict(d))


def string_to_list(in_string, sep = '..') :
    """
    Take a string and split it with the separator, returns the element
    if each is a string, if 2 numbers return the range instead

    """

    values = in_string.split(sep)
    # Two cases : numeric or categorical :
    nvalues = len(values)
    log = logging.getLogger(__name__)
    if (nvalues < 2) :
        log.error("Expect more than 1 value separated by {} but got this {} number of\
 values for the following string : {}".format(sep,nvalues,in_string))

    # not perfect test, for instance wont work on exponent
    if (values[0].strip().replace(".","",1).isnumeric()) :
        if (nvalues == 2) :
            values_out = np.arange(float(values[0]),float(values[1]))
        elif (nvalues == 3) :
            values_out = np.arange(float(values[0]),float(values[1]),float(values[2]))
        else :
            values_out = np.array(values).astype(float)
    else :
        values_out = [x.strip() for x in values]
    # convert back to int if number are integers
    if (values[0].strip().isdigit()) :
        values_out = values_out.astype(int)

    # need to return list not np array, otherwise issue with KerasRegressor    
    return values_out.tolist()
    

#### USEFUL FUNCTIONs for the OPTIMIZATION NODE

def create_param_grid(
        model_params: Dict[str,Any],
        sep : str
        ):
    """
    Take the parameter dictionary, identify the varying parameters,
    and create a parameter grid 
    """
    flat_model_params = flatten_dict(model_params,sep='__')
    # Create list of varying parameters :
    variable_params = {}
    for keyi,valuei in flat_model_params.items() :
        if isinstance(valuei,str) :
            if (sep in valuei) :
                values_out = string_to_list(valuei,sep=sep)
                if (len(values_out) > 1) :
                    variable_params[keyi]=values_out

    # for now only allow model and gate_cluster parameters tuning :
    # for model, we would just need to change model_params into model :
    # Add prefix core_pipeline__ to run inside the FilterPipeline
    param_grid = {}
    for variable_i in variable_params :
        if  variable_i.startswith("model_params") :
          #param_grid[variable_i.replace('model_params','model__regressor')]=variable_params[variable_i]
          param_grid[variable_i.replace('model_params','core_pipeline__model__regressor')]=variable_params[variable_i]
        if  variable_i.startswith("gate_cluster") :
          #param_grid[variable_i.replace('gate_cluster','gate_encoder')]=variable_params[variable_i]            
          param_grid[variable_i.replace('gate_cluster','core_pipeline__gate_encoder')]=variable_params[variable_i]
    return param_grid


def replace_model_params(
        model_params : Dict[str, Any],
        params_to_replace : Dict[str,Any]
)-> Dict[str, Any] :

    out_model_params = deepcopy(model_params)
    for keyi,valuei in params_to_replace.items() :
        # need to remove the prefix when optimzing FilterPipeline
        keyi = keyi.replace('core_pipeline__','')
        if keyi.startswith('gate_encoder') :
            out_model_params['gate_cluster'][keyi.replace('gate_encoder__','')] = valuei
        if keyi.startswith('model__regressor') :
            out_model_params['model_params'][keyi.replace('model__regressor__','')] = valuei

    return out_model_params



###  Function to define a model for KerasRegressor (sklearn keras wrapper)
###  with a wrapper to define the number of input features

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import Adam
from keras.wrappers.scikit_learn import KerasRegressor

def keras_model_definition_ndim(input_dim=50) :
  def keras_model_definition(n_hidden = 64, n_layers = 3, activation = 'relu',
                             drop_out = 0.5, loss = 'mean_absolute_error',
                             learning_rate = 0.1) :
      model = Sequential()
      model.add(Dense(n_hidden, activation=activation, input_dim=input_dim))
      model.add(Dropout(drop_out))
      for k in range(n_layers-1):
        model.add(Dense(n_hidden, activation=activation))
        model.add(Dropout(drop_out))
      model.add(Dense(1, activation='linear'))
      adam = Adam(learning_rate=learning_rate)
      model.compile(loss=loss,
                    optimizer=adam)
      return model
  return keras_model_definition  



################################  UNIMPEDED /IMPEDED CORE FUNCTIONS ##########


### GENERIC NODE FUNCTIONS CALLED BY SPECIFIC (full, AMA, ramp) NODE FUNCTIONS

def _define_any_model(
    data: pd.DataFrame,
    model_params: Dict[str, Any],
    global_params: Dict[str, Any],
    random_seed: str,
) -> FilterPipeline:


    # Test if the model should be a constant :
    if (model_params['model'] == 'Constant') :
      constant_value = model_params['model_params']['constant_value']
      # no filter just return a constant
      filt_pipeline = FilterPipeline(
        sklearn_Pipeline([
          ('sanitize',FunctionTransformer(lambda x : x.apply(lambda col: np.zeros(len(col))), validate=False)),
          ('model',DummyRegressor(strategy='constant', constant=constant_value))
        ]),
        constant_value)
      return filt_pipeline
    
    # Define filters first, so I can set the core_features rules for the rest of the transformations
    # FilterPipeline is a new class inheriting from sklearn Pipeline class
    # its attribute core_pipeline is the scikit learn pipeline
    # it defines behavior for rows not satisfying some criteria and log the errors
    # it applies for features_core because there is no imputing for these features
    # for each feature, they add an include rule and an exclude rule
    # why do they have an include rule feature_values + missing_values and then exclude missing_values ???
    # answer : because they want to log the error differently
    # 1st rule excludes category_exclusions and category not in training set
    # 2nd rule excludes Nan, None and ''
    # I just realized this test can be in the general function, ie does not have to be in the wrapper
    if ('unimpeded_AMA' in data) :
       train_selection = data.unimpeded_AMA
    else :
       train_selection = True
    default_answer = np.nanmedian(data.loc[(data.group == 'train') & train_selection , model_params['target']])
    filt_pipeline = FilterPipeline(sklearn_Pipeline([('dummy',FunctionTransformer(validate=False))]), default_answer)
    missing_values = [np.nan, None, '']
    ## for now only features with one-hot-encoder can only predict known categories
    nonewcat_features = model_params['OneHotEncoder_features']
    for feature_name in model_params['features_core']:
        if feature_name in nonewcat_features :
          feature_values = [
            c for c in data.loc[(data.group == 'train') & (data[feature_name].notnull()), feature_name].unique()
            if (feature_name not in global_params['category_exclusions']) or
            (str(c) not in [str(v) for v in global_params['category_exclusions'][feature_name]])
          ]
          # Rules flagging unknown values (ignoring missing values) : Here do not include values not in training
          # set or that are in the category_exclusions or that are null wo being in missing_values
          filt_pipeline.add_include_rule(feature_name, feature_values + missing_values, 'Unknown ' + feature_name)
        # Rules flagging missing values : now remove the missing values
        filt_pipeline.add_exclude_rule(feature_name, missing_values, 'Missing ' + feature_name)
        # these are used on input to _fit,_predict,_score
    # Rules flagging invalid predictions/target : it is applied on _fit, _predict
    filt_pipeline.add_exclude_rule_preds(lambda x: x < 0, 'Negative prediction')
    filter_sel_x , _, filter_sel_y, _ = filt_pipeline.filter(data.loc[data.group == 'train',model_params['features']],
                                                             data.loc[data.group == 'train',model_params['target']])

    pip_filter = filter_sel_x & filter_sel_y
    
    
    # Replaces miscellaneous missing values with expected values (takes care of the Nan,
    # only for not "features_core"
    format_missing_data = FormatMissingData(model_params['features_core'])    
    
    # Orders feature columns : when you fit you look at the column order, transform will put
    # column back in the same order. 
    order_features = OrderFeatures()


    # One hot encoding : possible categories are taken from training set with not null value
    # No 'category_exclusions' handled here
    # first dropna is to remove categories that will be removed by the wrapper on features_core
    features_onehot = model_params['OneHotEncoder_features']
    one_hot_enc = OneHotEncoder(
        categories = [
            [
                c for c in data[data.group == 'train'].loc[pip_filter,feature].dropna().unique()
                if (feature not in global_params['category_exclusions']) or \
                (str(c) not in [str(v) for v in global_params['category_exclusions'][feature]])
            ]
            for feature in features_onehot
        ],
        sparse=False,handle_unknown="ignore")
  
    # ColumnTransformer takes a list of (name,transformers,columns)
    # the tranformer to be applied need to have a fit and transform function
    # remainder indicates here that feature without a defined transformer
    # just pass through unchanged (default is to drop them)
    # sparse_threshold =0 prevents to compress data into sparse matrix format
    # a column with two transformers will have 2 output columns (no consecutive transformation)
    # Order of the tranformed columns is the order of the transformation

    onehot_transformer = ColumnTransformer(
         [('one_hot_encoder',one_hot_enc,features_onehot)],         
        remainder='passthrough',
        sparse_threshold=0,
    )
    
    ############
    # StandEncoder : for now transform a stand like B12 into an extra column group_B with a value 12
    # stand A10 would be an extra column group_A with a value 10, all stands not in the "group" would have
    # a value of 0 for the "group_?" feature
    ###########
    # Gate encoder that is going directly in the pipeline instead of col transformer
    # because it needs several columns
    # Gate encoder needs to go before one-hot-encoder because it needs
    # some of these features as string not encoded
    # add active_run_id to log plot into the mlflow server

   
    if ('gate_cluster' in model_params.keys()) :
        gate_encoder = GateClusterEncoder(**model_params['gate_cluster'], active_run_id=None)
        ninputs = gate_encoder.nclusters
    else :
        gate_encoder = ColumnTransformer(
            [('stand_encoder',StandEncoder(),['departure_stand_actual'])],
            remainder='passthrough',
            sparse_threshold=0,
        )
        # need to try out the stand encoder to define the output size (for Keras)
        test = StandEncoder()
        test.fit(data.loc[train_selection & (data.group == 'train'), model_params['features']])
        ninputs = len(test.colnames)

    # Binned time transform (temp):
    if ('Binned_features' in model_params) :
      Binned_features = model_params['Binned_features']
    else:
      Binned_features = []
      
    def func_trans(x):
      return x.apply(lambda col : pd.to_datetime(col).apply( lambda t : (t.hour*60+t.minute)//10) if col.name in Binned_features else col)
    binned_time_transf = FunctionTransformer(func_trans,validate=False)

    # Calculated the # of features for Keras :
    ninputs = ninputs + np.sum([len(x) for x in onehot_transformer.transformers[0][1].categories])
    if (model_params['model'] == 'KerasNN') :
      model_params['model_params'].update({'build_fn' : keras_model_definition_ndim(ninputs)})
      
    # Define Possible models :
    model_dict = {
      'XGBRegressor': xgb.XGBRegressor,
      'GradientBoostingRegressor':  GradientBoostingRegressor,
      'RandomForestRegressor' : RandomForestRegressor,
      'KerasNN' : KerasRegressor
    }

     
    if (model_params['model'] in model_dict) :
      model = model_dict[model_params['model']](**model_params['model_params'])
    else :
      log.error("model_params['model'] is not in the model_dict, given \
      value is {}\n".format(model_params['model']))

    if ('Constant_shift' in model_params) :
      constant_shift = model_params['Constant_shift']
    else:
      constant_shift = 0.0
      
    model = TransformedTargetRegressor(
        regressor = model, 
        inverse_func = lambda x: np.round(x-constant_shift), 
        check_inverse = False, 
    )


    # Make pipeline,
    pipeline = sklearn_Pipeline(
        steps=[
            ('order_features', order_features),
            ('format_missing_data', format_missing_data),
            ('binned_time_transf',binned_time_transf),
            ('gate_encoder', gate_encoder),
            ('onehot_transformer', onehot_transformer),
            ('to_array', FunctionTransformer(lambda x : x.values,
                                             validate=False)),
            ('model', model),
        ]
    )

    # Replace dummy pipeline with the real one :
    filt_pipeline.core_pipeline = pipeline
    filt_pipeline.steps = pipeline.steps

    return filt_pipeline


            
def _optimize_any_model(
    data: pd.DataFrame,
    model_params: Dict[str, Any],
    global_params: Dict[str, Any],
    active_run_id: str,
    random_seed: str,
    hyper_params: Dict[str, str],
    filt_pipeline: FilterPipeline,
    train_selection: Union[pd.Series, bool],
) -> FilterPipeline:

    sep = hyper_params['sep']
    param_grid = create_param_grid(model_params, sep)
    hypertune_dict = {
      'gridsearch' : GridSearchCV,
      'hyperopt'  :  OptimizerHOP,
      }
    
    
    if (len(param_grid) > 0) :
        def neg_mad(y_t,y_p):
            log = logging.getLogger(__name__)
            res = y_t-y_p
            log.info(f"Score function - percentage of NaN residual : {np.isnan(res).sum()/100./len(res)}%")
            return -1*np.nanmedian(np.abs(res-np.nanmedian(res)))*1.4826

        from sklearn.metrics import make_scorer
        #!!!!!corepip
        #search = GridSearchCV(filt_pipeline.core_pipeline, param_grid=param_grid,scoring=make_scorer(neg_mad), cv=3)
        if (hyper_params['method'] in hypertune_dict) :
          search = hypertune_dict[hyper_params['method']](
            filt_pipeline, param_grid=param_grid, scoring=make_scorer(neg_mad), cv=KFold(n_splits=5,shuffle=True,random_state=53)
          )
        else :
          log.error("Unknown Optimization method : {}\n".format(hyper_params['method']))

        search.fit(
            data.loc[
                train_selection & (data.group == 'train'),
                model_params['features']
            ],
            data.loc[
                train_selection & (data.group == 'train'),
                model_params['target']
            ]
        )

        if (active_run_id != None) :
            results = search.cv_results_
            params = results['params']
            mad_val = results['mean_test_score']
            std_mad_val = results['std_test_score']
            with mlflow.start_run(run_id=active_run_id) as parent_run :
                # JSONize through pandas to avoid numpy type issue
                pd.DataFrame(results).to_json('./results_cv.json')
                mlflow.log_artifact('./results_cv.json')
                os.remove('./results_cv.json')
                for parami,mad_vali,std_mad_vali in zip(params, mad_val, std_mad_val) :
                    with mlflow.start_run(nested=True) as child_run :
                        mlflow.log_params(flatten_dict(parami,sep='__'))
                        mlflow.log_metric('MAD_residual',-1*mad_vali) # log back the + value
                        mlflow.log_metric('STD_MAD_residual',std_mad_vali)

        # Need to return the best_model params to run with
        best_model_params = replace_model_params(model_params, search.best_params_)
        # Setup the best model
        #!!!!!corepip
        #filt_pipeline.core_pipeline.set_params(**search.best_params_)
        filt_pipeline.set_params(**search.best_params_)
    else :
        # No parameter to optimize then 
        best_model_params = deepcopy(model_params)
        
    # Put the active_run_id in gate_encoder of the Pipeline (default is set to None)
    if ('gate_cluster' in model_params) :
        filt_pipeline.core_pipeline.set_params(gate_encoder__active_run_id=active_run_id)

    return best_model_params, filt_pipeline
                

def _train_any_model(
    data: pd.DataFrame,
    model_params: Dict[str, Any],
    global_params: Dict[str, Any],
    active_run_id: str,
    random_seed: str,
    filt_pipeline : FilterPipeline,
    train_selection: Union[pd.Series, bool],
) -> List[FilterPipeline]:


    log = logging.getLogger(__name__)
    tic = time.time()    
    if (model_params['model'] == 'Constant') :
      # to avoid numerical error when fitting a predefined constant model
      constant_value = model_params['model_params']['constant_value']
      data.loc[:, model_params['target']] = constant_value
      log.info('Constant model : set y of the fit to {}'.format(constant_value))         
    
    if ('features' in model_params) & ('target' in model_params) :
      filt_pipeline.fit(
        data.loc[
          train_selection & (data.group == 'train'),
          model_params['features']
        ],
        data.loc[
          train_selection & (data.group == 'train'),
          model_params['target']
        ],
      )
      
      # Check if there is some CV splits in the data, hopefully nothing else is called test_[0-9]+
      # Here the main model is already subtracted given we look at the dataframe column names
      n_cv = data.keys().str.match('test_[0-9]+').sum()
      out_models = [filt_pipeline] # first model ran on all training data
      # load name of the function/class for each steps :
      step_func_names = [x[1].__class__.__name__ for x in filt_pipeline.steps] 
      for kk in range(n_cv) :
        train_index = train_selection & (data['test_'+str(kk)]  == False)
        clone_filt_pipeline = sk_clone(filt_pipeline)
        # Don't log the gatecluster results in MLFlow
        if ('GateClusterEncoder' in step_func_names) :
          clone_filt_pipeline.core_pipeline.set_params(gate_encoder__active_run_id=None)
        out_models.append(
          clone_filt_pipeline.fit(data.loc[train_index, model_params['features']],
                                  data.loc[train_index, model_params['target']])
        )      
    else :
      log.warning('No features or target , skipping the model fitting')
      out_models = [filt_pipeline] # now we return a list of models even if only 1 model
    toc = time.time()

    log.info('training {} model took {:.1f} minutes'.format(
        model_params['name'], (toc - tic) / 60)
    )
  
    
    if (active_run_id != None) :
        with mlflow.start_run(run_id=active_run_id):
            # Log trained model
            mlf_sklearn.log_model(
                sk_model=filt_pipeline,
                artifact_path='model',
                conda_env=add_environment_specs_to_conda_file())
            # Set tags
            mlflow.set_tag('airport_icao', global_params['airport_icao'])
            mlflow.set_tag('Model Version', 0)
            # Log model parameters one at a time so that character limit is
            # 500 instead of 250
            for key, value in model_params.items():
                mlflow.log_param(key, value)
            # Still hoping we can do all the globals in one shot
            mlflow.log_params(global_params)
            mlflow.log_param("Train_test_random_seed",random_seed)
            mlflow.log_artifact('./conf/base/globals.yml')
            mlflow.log_artifact('./conf/base/parameters.yml')
            mlflow.log_artifact('./conf/base/catalog.yml')
            sha_commit = os.popen('git rev-parse HEAD').read()
            mlflow.set_tag('mlflow.source.git.commit',sha_commit)
            os.system('git status &> git_status.txt')
            os.system('pwd >> git_status.txt')
            mlflow.log_artifact('git_status.txt')
            os.remove('git_status.txt')
            os.system('git diff &> git_diff.txt')
            mlflow.log_artifact('git_diff.txt')
            os.remove('git_diff.txt')
            # log number of samples :
            mlflow.log_param("Number of training samples",(train_selection & (data.group == 'train')).sum())
            mlflow.log_param("Number of testing samples",(train_selection & (data.group == 'test')).sum())
            mlflow.log_param("Total number of samples",len(data))
            ### Calculate distribution of features and log it
            distribution_files = tsd.training_set_feature_distributions(data, model_params)
            for file_i in distribution_files:
              mlflow.log_artifact(file_i)
              os.remove(file_i)
            
    return out_models

################################## UNIMPEDED FUNCTIONs/NODEs
  
### FULL TAXI

 
def define_unimp_full_model(
    data: pd.DataFrame,
    model_params: Dict[str, Any],
    global_params: Dict[str, Any],
    random_seed: str,
)-> FilterPipeline :
    return _define_any_model(data, model_params, global_params, random_seed)



def optimize_unimp_full_model(
    data: pd.DataFrame,
    model_params: Dict[str, Any],
    global_params: Dict[str, Any],
    active_run_id: str,
    random_seed: str,
    hyper_params: Dict[str,str],
    filt_pipeline: FilterPipeline,
) :
    if ('unimpeded_AMA' in data) :
       train_selection = data.unimpeded_AMA
    else :
       train_selection = True
    return _optimize_any_model(data, model_params, global_params, active_run_id,
                               random_seed, hyper_params, filt_pipeline, train_selection)


def train_unimp_full_model(
    data: pd.DataFrame,
    model_params: Dict[str, Any],
    global_params: Dict[str, Any],
    active_run_id: str,
    random_seed: str,
    filt_pipeline: FilterPipeline,
) -> List[FilterPipeline]:
    if ('unimpeded_AMA' in data) :
       train_selection = data.unimpeded_AMA
    else :
       train_selection = True
    return _train_any_model(data, model_params, global_params, active_run_id,
                            random_seed, filt_pipeline, train_selection)


  

### AMA TAXI
  
def define_unimp_ama_model(
    data: pd.DataFrame,
    model_params: Dict[str, Any],
    global_params: Dict[str, Any],
    random_seed: str,
)-> FilterPipeline :
    return _define_any_model(data, model_params, global_params, random_seed)



def optimize_unimp_ama_model(
    data: pd.DataFrame,
    model_params: Dict[str, Any],
    global_params: Dict[str, Any],
    active_run_id: str,
    random_seed: str,
    hyper_params: Dict[str,str],
    filt_pipeline: FilterPipeline,
) :
    if ('unimpeded_AMA' in data) :
       train_selection = data.unimpeded_AMA
    else :
       train_selection = True
    return _optimize_any_model(data, model_params, global_params, active_run_id,
                               random_seed, hyper_params, filt_pipeline, train_selection)


def train_unimp_ama_model(
    data: pd.DataFrame,
    model_params: Dict[str, Any],
    global_params: Dict[str, Any],
    active_run_id: str,
    random_seed: str,
    filt_pipeline : FilterPipeline,
) -> List[FilterPipeline]:
    if ('unimpeded_AMA' in data) :
      train_selection = data.unimpeded_AMA
    else :
      train_selection = True
    return _train_any_model(data, model_params, global_params, active_run_id,
                            random_seed, filt_pipeline, train_selection)


### RAMP TAXI
  
def define_unimp_ramp_model(
    data: pd.DataFrame,
    model_params: Dict[str, Any],
    global_params: Dict[str, Any],
    random_seed: str,
)-> FilterPipeline :
    return _define_any_model(data, model_params, global_params, random_seed)



def optimize_unimp_ramp_model(
    data: pd.DataFrame,
    model_params: Dict[str, Any],
    global_params: Dict[str, Any],
    active_run_id: str,
    random_seed: str,
    hyper_params: Dict[str,str],
    filt_pipeline: FilterPipeline,
) :
    train_selection = True
    return _optimize_any_model(data, model_params, global_params, active_run_id,
                               random_seed, hyper_params, filt_pipeline, train_selection)


  
def train_unimp_ramp_model(
    data: pd.DataFrame,
    model_params: Dict[str, Any],
    global_params: Dict[str, Any],
    active_run_id: str,
    random_seed: str,
    filt_pipeline: FilterPipeline,
) -> List[FilterPipeline]:
    train_selection = True
    return _train_any_model(data, model_params, global_params, active_run_id,
                            random_seed, filt_pipeline, train_selection)
  
    
  


################################  IMPEDED ##########


##### AMA

def define_imp_ama_model(
    data: pd.DataFrame,
    model_params: Dict[str, Any],
    global_params: Dict[str, Any],
    random_seed: str,
)-> FilterPipeline :
    return _define_any_model(data, model_params, global_params, random_seed)
  


def optimize_imp_ama_model(
    data: pd.DataFrame,
    model_params: Dict[str, Any],
    global_params: Dict[str, Any],
    active_run_id: str,
    random_seed: str,
    hyper_params: Dict[str,str],
    filt_pipeline: FilterPipeline,
) :
    train_selection = True
    return _optimize_any_model(data, model_params, global_params, active_run_id,
                               random_seed, hyper_params, filt_pipeline, train_selection)


def train_imp_ama_model(
    data: pd.DataFrame,
    model_params: Dict[str, Any],
    global_params: Dict[str, Any],
    active_run_id: str,
    random_seed: str,
    filt_pipeline : FilterPipeline,
) -> List[FilterPipeline]:
    train_selection = True
    return _train_any_model(data, model_params, global_params, active_run_id,
                            random_seed, filt_pipeline, train_selection)

##################

####### full

def define_imp_full_model(
    data: pd.DataFrame,
    model_params: Dict[str, Any],
    global_params: Dict[str, Any],
    random_seed: str,
)-> FilterPipeline :
    return _define_any_model(data, model_params, global_params, random_seed)



def optimize_imp_full_model(
    data: pd.DataFrame,
    model_params: Dict[str, Any],
    global_params: Dict[str, Any],
    active_run_id: str,
    random_seed: str,
    hyper_params: Dict[str,str],
    filt_pipeline: FilterPipeline,
) :
    train_selection = True
    return _optimize_any_model(data, model_params, global_params, active_run_id,
                               random_seed, hyper_params, filt_pipeline, train_selection)


def train_imp_full_model(
    data: pd.DataFrame,
    model_params: Dict[str, Any],
    global_params: Dict[str, Any],
    active_run_id: str,
    random_seed: str,
    filt_pipeline : FilterPipeline,
) -> List[FilterPipeline]:
    train_selection = True
    return _train_any_model(data, model_params, global_params, active_run_id,
                            random_seed, filt_pipeline, train_selection)


####### ramp

def define_imp_ramp_model(
    data: pd.DataFrame,
    model_params: Dict[str, Any],
    global_params: Dict[str, Any],
    random_seed: str,
)-> FilterPipeline :
    return _define_any_model(data, model_params, global_params, random_seed)
  


def optimize_imp_ramp_model(
    data: pd.DataFrame,
    model_params: Dict[str, Any],
    global_params: Dict[str, Any],
    active_run_id: str,
    random_seed: str,
    hyper_params: Dict[str,str],
    filt_pipeline: FilterPipeline,
) :
    train_selection = True
    return _optimize_any_model(data, model_params, global_params, active_run_id,
                               random_seed, hyper_params, filt_pipeline, train_selection)


def train_imp_ramp_model(
    data: pd.DataFrame,
    model_params: Dict[str, Any],
    global_params: Dict[str, Any],
    active_run_id: str,
    random_seed: str,
    filt_pipeline : FilterPipeline,
) -> List[FilterPipeline]:
    train_selection = True
    return _train_any_model(data, model_params, global_params, active_run_id,
                            random_seed, filt_pipeline, train_selection)




  
################################################################
################################  BASELINE MODELS ##############
################################################################

def _train_any_baseline(
    data: pd.DataFrame,
    model_params: Dict[str, Any],
    global_params: Dict[str, Any],
    active_run_id: str,
    train_selection: Union[pd.Series, bool],
) -> FilterPipeline:

    pip_steps = []


    # Get also a non-nan default answer for baseline model
    default_answer = np.nanmedian(data.loc[(data.group == 'train') & train_selection, model_params['target']])

    # Better to follow the same basic filtering/wrapping as the full estimator
    filt_pipeline = FilterPipeline(sklearn_Pipeline([('dummy',FunctionTransformer(validate=False))]), default_answer)
    if not('baseline' in model_params) :
      return filt_pipeline
    missing_values = [np.nan, None, '']
    for feature_name in model_params['features_core']:
        feature_values = [
            c for c in data.loc[(data.group == 'train') & (data[feature_name].notnull()), feature_name].unique()
            if (feature_name not in global_params['category_exclusions']) or
            (str(c) not in [str(v) for v in global_params['category_exclusions'][feature_name]])
        ]
        filt_pipeline.add_include_rule(feature_name, feature_values + missing_values, 'Unknown ' + feature_name)
        filt_pipeline.add_exclude_rule(feature_name, missing_values, 'Missing ' + feature_name)
    filt_pipeline.add_exclude_rule_preds(lambda x: x < 0, 'Negative prediction')


    # Orders feature columns : when you fit you look at the column order, transform will put
    # column back in the same order. Why would column order be different ?
    order_features = OrderFeatures()
    pip_steps.append(('order_features', order_features))

    # Replaces miscellaneous missing values with expected values (takes care of the Nan,
    # only for not "features_core"
    format_missing_data = FormatMissingData(model_params['features_core'])    
    pip_steps.append(('format_missing_data', format_missing_data))
     
    # Get a value for the Terminal if terminal used
    if ('terminal' in model_params['baseline']['group_by_features']) :
      features_stand = ['departure_stand_actual']
      terminal_encoder = TerminalEncoder(features_stand)
      pip_steps.append(('terminal_encoder', terminal_encoder))
          
        
    if 'baseline' in model_params:
        baseline_model = GroupByModel(model_params['baseline'])
    else:
        baseline_model = DummyClassifier(strategy='most_frequent') 
    pip_steps.append(('baseline_model',baseline_model))
        

    # Make pipeline
    pipeline = sklearn_Pipeline(
        steps=pip_steps
    )

    # Replace dummy pipeline with the real one :
    filt_pipeline.core_pipeline = pipeline        

    tic = time.time()
    filt_pipeline.fit(
        data.loc[
            train_selection & (data.group == 'train'),
            model_params['features']
        ],
        data.loc[
            train_selection & (data.group == 'train'),
            model_params['target']
        ],
    )
    toc = time.time()

    log = logging.getLogger(__name__)
    log.info('training unimpeded baseline for {} took {:.1f} minutes'.format(
        model_params['target'], (toc - tic) / 60)
    )
    
    if (active_run_id != None) :
        with mlflow.start_run(run_id=active_run_id):
            # Log trained model
            mlf_sklearn.log_model(
                sk_model=baseline_model,
                artifact_path='baseline',
                conda_env=add_environment_specs_to_conda_file())
        
    return filt_pipeline


#### Baseline unimpeded

def train_unimp_full_baseline(
    data: pd.DataFrame,
    model_params: Dict[str, Any],
    global_params: Dict[str, Any],
    active_run_id: str,
) -> FilterPipeline:
    if ('unimpeded_AMA' in data) :
      train_selection = data.unimpeded_AMA
    else :
      train_selection = True
    return _train_any_baseline(data, model_params, global_params, active_run_id, train_selection)


def train_unimp_ama_baseline(
    data: pd.DataFrame,
    model_params: Dict[str, Any],
    global_params: Dict[str, Any],
    active_run_id: str,
) -> FilterPipeline:
    if ('unimpeded_AMA' in data) :
      train_selection = data.unimpeded_AMA
    else :
      train_selection = True
    return _train_any_baseline(data, model_params, global_params, active_run_id, train_selection)



def train_unimp_ramp_baseline(
    data: pd.DataFrame,
    model_params: Dict[str, Any],
    global_params: Dict[str, Any],
    active_run_id: str,
) -> FilterPipeline:
    train_selection = True
    return _train_any_baseline(data, model_params, global_params, active_run_id, train_selection)


#### Baseline impeded
 
def train_imp_full_baseline(
    data: pd.DataFrame,
    model_params: Dict[str, Any],
    global_params: Dict[str, Any],
    active_run_id: str,
) -> FilterPipeline:
    train_selection = True
    return _train_any_baseline(data, model_params, global_params, active_run_id, train_selection)


def train_imp_ama_baseline(
    data: pd.DataFrame,
    model_params: Dict[str, Any],
    global_params: Dict[str, Any],
    active_run_id: str,
) -> FilterPipeline:
    train_selection = True
    return _train_any_baseline(data, model_params, global_params, active_run_id, train_selection)



def train_imp_ramp_baseline(
    data: pd.DataFrame,
    model_params: Dict[str, Any],
    global_params: Dict[str, Any],
    active_run_id: str,
) -> FilterPipeline:
    train_selection = True
    return _train_any_baseline(data, model_params, global_params, active_run_id, train_selection)

 
  

################################################################
################################  PREDICTION ###################
################################################################

def predict(
    pipeline: Union[List[FilterPipeline], FilterPipeline],
    data: pd.DataFrame,
    model_params: Dict[str, Any],
) -> pd.DataFrame:
    
    # Run model

    log = logging.getLogger(__name__) 
    data_out = data.copy()

    # Now it should be a list, but just in case I am checking
    if isinstance(pipeline, list) :
      pipeline_one = pipeline[0]
    else :
      pipeline_one = pipeline

      
    if (('features' in model_params) & (model_params['model'] != 'Constant')) :
      tic = time.time()
      predictions = pipeline_one.predict(
        data_out[model_params['features']]
      )
      toc = time.time()
      log.info('predicting took {:.1f} minutes'.format(
        (toc-tic)/60)
               )
      # Add predictions to dataframe for convenience
      data_out['predicted_{}'.format(model_params['name'])] = predictions
      # Flag predictions with missing core features
      data_out['missing_core_features'] = pipeline_one.filter(data_out[model_params['features']])[0] == False
    else:
      log.warning('no features in model_params or model = Constant, skipping predictions')

    # Add prediction from CV fitting
    if isinstance(pipeline, list) :
      for kk,pipeline_i in enumerate(pipeline[1:]) :
        predictions = pipeline_i.predict(data_out[model_params['features']])
        data_out['predictedCV{}_{}'.format(kk, model_params['name'])] = predictions
      
    return data_out



def predict_baseline(
    pipeline: sklearn_Pipeline,
    data: pd.DataFrame,
    model_params: Dict[str, Any]
) -> pd.DataFrame:

    log = logging.getLogger(__name__)
    data_out = data.copy()
    if 'baseline' in model_params:        
        # Run model
        tic = time.time()
        predictions = pipeline.predict(
            data_out[model_params['features']]
        )
        toc = time.time()
        
        log.info('predicting baseline took {:.1f} minutes'.format(
            (toc-tic)/60)
                 )
        # Add predictions to dataframe for convenience
        data_out['predicted_baseline'] = predictions
        # Flag predictions with missing core features
        data_out['missing_core_features_baseline'] = pipeline.filter(data_out)[0] == False
    else :
      log.warning('no baseline in model_params, skipping baseline predictions')

    return data_out



################################################################
################################  METRICS ###################
################################################################


def report_performance_metrics(
    data: pd.DataFrame,
    model_params: Dict[str, Any],
    active_run_id: str,
) -> None:
    """Node for reporting performance metrics. Notice that this function has no
    outputs.
    """
    log = logging.getLogger(__name__)

    # Predictive model
    if 'predicted_{}'.format(model_params['name']) in data :
      report_model_metrics(data,
                           model_params,
                           'predicted_{}'.format(model_params['name']),
                           active_run_id)
    else :
      log.warning('no model predictions, skipping metrics calculation')
      
    # Baseline
    if 'predicted_baseline' in data.columns:
        report_model_metrics(data, 
                             model_params, 
                             'predicted_baseline',
                             active_run_id,
                             'baseline_',
                             ['train','test']) # I add 'train' to see how it does
    else :
      log.warning('no baseline predictions, skipping metrics calculation')

    # STBO as truth data
    if ('label' in model_params) :
      if 'undelayed_departure_{}_transit_time'.format(model_params['label']) in data.columns:
        report_model_metrics_STBO(data, 
                                  model_params, 
                                  'predicted_{}'.format(model_params['name']),
                                  active_run_id,
                                  'STBO_',
                                  ['train','test'])
      else:
        log.warning('no undelayed_departure transit time, skipping metrics calculation')
        

def report_model_metrics(
    data: pd.DataFrame,
    model_params: Dict[str, Any],
    y_pred: str,
    active_run_id: str,
    name_prefix: str = '',
    group_values : list = ['train','test']
) -> None:
    """Node for reporting the performance metrics of the predictions performed
    by the previous node. Notice that this function has no outputs, except
    logging.
    """

    metrics_dict = {
        metric_name: METRIC_NAME_TO_FUNCTION_DICT[metric_name]
        for metric_name in model_params['metrics']
    }

    evaluation_df = evaluate_predictions(
        data[(data.missing_core_features == False) &
              (data['predicted_{}'.format(model_params['name'])].isna() == False) &
              (data['predicted_baseline'].isna() == False) &
               data.group.isin(group_values)],
        y_true=model_params['target'],
        y_pred=y_pred,
        metrics_dict=metrics_dict,
    )
  
    # Log the accuracy of the model
    log = logging.getLogger(__name__)

    if (active_run_id != None) :
        with mlflow.start_run(run_id=active_run_id):
            # Set the metrics
            # for metric_name in metrics_dict.keys():
            # use evaluation_df to get the unimpeded_AMA metrics as well (if calculated)
            for metric_name in evaluation_df.keys() :
                log.info("metric {}:".format(name_prefix + metric_name))
                for group in [v for v in data.group.unique() if v in group_values]:
                    log.info("{} group: {}".format(
                        group,
                        evaluation_df.loc[group, metric_name]
                    ))
                    mlflow.log_metric(
                        name_prefix + metric_name + '_' + group,
                        evaluation_df.loc[group, metric_name]
                    )


def report_model_metrics_STBO(
    data: pd.DataFrame,
    model_params: Dict[str, Any],
    y_pred: str,
    active_run_id: str,
    name_prefix: str = '',
    group_values : list = ['train','test']
) -> None:
    """Node for reporting the performance metrics of the predictions performed
    by the previous node. Notice that this function has no outputs, except
    logging.
    """

    metrics_dict = {
        metric_name: METRIC_NAME_TO_FUNCTION_DICT[metric_name]
        for metric_name in model_params['metrics']
    }

    if 'undelayed_departure_{}_transit_time'.format(model_params['label']) in data.columns:
        data_filtered = data[(data.missing_core_features == False) &
              (data['predicted_{}'.format(model_params['name'])].isna() == False) &
              (data['predicted_baseline'].isna() == False) &
              (data['undelayed_departure_{}_transit_time'.format(model_params['label'])].isna() == False) &
               data.group.isin(group_values)]
        evaluation_df_STBO = evaluate_predictions(
            data_filtered,
            y_true='undelayed_departure_{}_transit_time'.format(model_params['label']),
            y_pred=y_pred,
            metrics_dict=metrics_dict,
        )

    # Log the accuracy of the model
    log = logging.getLogger(__name__)

    if (active_run_id != None) :
        with mlflow.start_run(run_id=active_run_id):
            # Set the metrics
            # for metric_name in metrics_dict.keys():
            # use evaluation_df to get the unimpeded_AMA metrics as well (if calculated)
            if 'undelayed_departure_{}_transit_time'.format(model_params['label']) in data.columns:
                for metric_name in evaluation_df_STBO.keys() :
                    log.info("metric {}:".format(name_prefix + metric_name))
                    for group in [v for v in data.group.unique() if v in group_values]:
                        log.info("{} group: {}".format(
                            group,
                            evaluation_df_STBO.loc[group, metric_name]
                        ))
                        mlflow.log_metric(
                            name_prefix + metric_name + '_' + group,
                            evaluation_df_STBO.loc[group, metric_name]
                        )




def evaluate_predictions(
    df,
    y_true,
    y_pred,
    metrics_dict={'mean_absolute_error': metrics.mean_absolute_error},
):
    evaluation_df = pd.DataFrame(
        index=df.group.unique(),
    )

    # Check if several predictions are in the dataframe for this predictions
    # the keys should be in increasing order because they were created that way
    CV_keys = df.keys()[df.keys().str.match(y_pred.replace('predicted','predictedCV[0-9]+'))]
    
    for metric_name, metric_func in metrics_dict.items():
        if metric_name == 'percent within n':
            continue  # Handled by separate function

        evaluation_df[metric_name] = None

        for group in df.group.unique():
            evaluation_df.loc[group, metric_name] =\
                metric_func(
                    df.loc[df.group == group, y_true],
                    df.loc[df.group == group, y_pred],
                )
            # if several CV predictions in df, calculate the spread of the metrics
            if (len(CV_keys) > 0) :
              nkeys = len(CV_keys)
              group_bool = (group == 'test')
              evaluation_df.loc[group, metric_name+'_STD'] =\
                np.nanstd(
                  list(map(lambda x : metric_func(df.loc[(df['test_'+str(x)] == group_bool) , y_true],
                                                  df.loc[(df['test_'+str(x)] == group_bool) , CV_keys[x]]),
                           range(nkeys)))
                )
            
            # Add an extra metrics if unimpeded AMA was calculated
            if ('unimpeded_AMA' in df) :
                sel_group = (df.group == group) & df.unimpeded_AMA
                evaluation_df.loc[group, 'unimpeded_AMA_'+metric_name] = \
                  metric_func( df.loc[sel_group, y_true],
                               df.loc[sel_group, y_pred] )
                # if several CV predictions in df, calculate the spread of the metrics
                if (len(CV_keys) > 0) :
                  group_bool = (group == 'test')
                  evaluation_df.loc[group, 'unimpeded_AMA_'+metric_name+'_STD'] =\
                    np.nanstd(
                      list(map(lambda x : metric_func(df.loc[(df['test_'+str(x)] == group_bool) & df.unimpeded_AMA, y_true],
                                                      df.loc[(df['test_'+str(x)] == group_bool) & df.unimpeded_AMA, CV_keys[x]]),
                               range(nkeys)))
                    )

                

    return evaluation_df

