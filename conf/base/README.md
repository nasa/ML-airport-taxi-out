# Configuration Files

-------------
## globals.yml
This file contains general information about the training,
all the parameters are sub-parameter to the parameter **globals**.

### globals 
* **airport_icao** : the airport icao code
* **start time** : Start time and date of the training/testing dataset, for instance _2020-08-01 08:00:00_
* **end time** : End time and date of the training/testing dataset
* **fuser_db_credentials** : name of the credentials that contains the information on how to access databases, the information is located in the conf/local/credentials.yml file,
* **STBO_data_credentials** : name of the credentials to use to connect to the database containing STBO information (for KDFW and KCLT)
* **category_exclusions** : it contains a dictionary of features, each entry list the values that needs to be filtered out by the FilterPipeline wrapper. Example of feature **departure_runway_actual**.

----------------
## parameters.yml

This file contains the parameters of the models including the
data engineering and feature engineering.

### globals
This parameter copy variables from globals.yml file, and contains
general information

### ntx_connection
Parameters describing the connection to the MLFlow server to copy artifacts : **host**, **username**, **port**. In general one will set up a tunnel between a port of the localhost and the MLFlow server port 22.  

### TEST_SIZE
Fraction of the data set used for testing (between 0 and 1), the rest is for training/validation.

### RANDOM_SEED
Random seed used to split the dataset between test/train, one can use a
special value _'RANDOM'_ and the seed will be chosen at random.

### NIQR_*_TAXI
Parameters of the data engineering N-InterQuartile Range for taxi time in the ramp, AMA and total (* = RAMP, AMA, FULL). If the parameter is set to -1, the filter is skipped. If the parameter is a single float, the data are filtered around the median with +- NIQR*IQR. If the parameter is a list of 2 floats, the data are filtered between those two percentiles ([0.1,0.2] : select data between 10th and 20th percentile).


### *_non_null
For each * (full, ama, ramp) pipelines, this parameter indicates which timestamps need to be defined (not null). In general, one would need to have the beginning and ending timestamps of the corresponding taxi valid (for a full taxi pipeline, it would be  _departure_runway_actual_time_, _departure_stand_actual_time_).


### upperlimit_counts
This variable contains a dictionary of pipelines (ramp, full). Each entry contains another dictionary of low-pass filter to be applied on the aircraft counts, either in number of aircrafts (value greater or equal to 1) or in percentile (value less than 1). The possible aircraft counts are arr_runway_AMA_count, dep_AMA_runway_count,
AMA_gate_count, dep_stand_AMA_count, total_arrivals_on_surface, total_departures_on_surface, total_flights_on_surface. This filter intent is to select unimpeded taxis, especially in the ramp, for the ramp taxi time estimate one would in principle switch from a quantile regressor to a simple regressor if the selection is successful enough.


### hypertune
Parameters for tuning the hyper-parameters. For now only parameters in **[imp/unimp]_*_model_params.gate_cluster** and  **[imp/unimp]_*_model_params.model_params** can be tuned.
* **sep** is setting the separator to describe the range in which to tune the parameters. One can either gives a list of parameters separated by **sep**, a range set by 2 int/float with a +1 increment (excluding the end boundary), a range set by 3 int/float with the last number indicating the increment.
* **method** is setting the tuning technique, for now only _gridsearch_ and _hyperopt_ are availble.

### cross_val
Parameters to run a cross-validation and calculate uncertainty on the metrics.
* **algo** is the algorithm used to sample the different training/testing sets, it can either be _montecarlo_ (random assignment) or _kfold_ (k-fold with shuffle sampling)
* **n_cv** is the number of time the data will be sampled (for _kfold_ it is also the number of fold)


### [imp/unimp]_*_model_params :
Parameters of the predictive model, metrics, output visualization. * stands for AMA, ramp or full.
* **name** : name of the model
* **label** [optional] : use to match the STBO taxi time to pipeline estimates to calculate the STBO metrics (CLT and DFW airports), it has to be _ama_, _ramp_ or _total_ for AMA, ramp and full pipelines
* **target** : column on which to apply the regression model
* **features** : list of features of the model
* **features_core** : list of features that are considered essential to make a prediction,
* **OneHotEncoder_features** : list of features to be one-hot encoded
* **gate_cluster** [optional] : parameters to describe an encoding of the stand
by clustering the target taxi time when going to **nrunways** runways. **nclusters** is the number of clusters, **scaler** is the method to scale the taxi time ("" : no scaling, _standard_ : StandardScaler, _minmax_ : MinMaxScaler), **clust** is the clustering algorithm (_hierch_ : AgglomerativeClustering , _kmeans_: KMeans). If gate_cluster is not provided, the model is using StandEncoder has the default encoder for the gate number.
* **model** : algorithm to apply the regression, for now it can be : _XGBRegressor_: XGBRegressor from xgboost, _GradientBoostingRegressor_ and _RandomForestRegressor_ from the sklearn library (Random Forest can be extremly long to train).
* **model_params** : parameters used to initiate the model
* **metrics** : list of metrics calculated on the model predictions, can be _mean_absolute_error_, _mean_absolute_percentage_error_, _median_absolute_percentage_error_, _percent_within_n_, _rmse_, _tilted_loss_, _fraction_less_than_actual_, _MAD_residual_, _median_residual_.
* **baseline** : baseline model parameters to compare with the model, **group_by_features** : list of features to aggregate the data with, **group_by_agg.metric** : _quantile_ or _mean_, if _quantile_ one needs to set **group_by_agg.param** as the quantile value.
* **mlflow** : parameters to connect to the MLFlow database, **tracking_uri** is the http address of the MLFlow server, **experiment_name** : name of the registered experiment (one per pipeline so far), **run_name** : name of the run
* **visual** [optional] : parameters to output some plots on the metrics, **plots** : list of plots to save (_residual_histogram_, _estimate_vs_truth_), **datasets** : list of groups to plot (_train_, _test_), **estimates** : which estimator to include in the plots (main model : _predicted__**name**, baseline : _predicted_baseline_)

-------------
## catalog.yml
This file contains the list of the input/intermediate/output data, for now these data are either contained in csv files on a local disk or on a remote database accessed with a SQL request.


### MFS_data_set@DB
MATM flight summary table to get summary values for each flights (like departure_stand_actual, arrival_stand_actual, departure_movement_area_actual_time, ...).
* **credentials** : name of the credentials to access the DB
* **load_args** : the DB is accessed with SQLQueryFIleChunkedDataSet class which allows to query data by chunk, one needs to provide the chunk size in days (**chunk_size_days**). **params** contains a dictionary of variables to change in the SQL request, the class is using read_sql_query from the pandas library to do so. The variable used here are  **airport_icao**, **end_time**, **start_time**.
* **sqlfilepath** contains the location of the SQL request file,
* **type** defines the class through which the data are going to be interacted with.
* **airport** is an airport ICAO code to inform the SQL reader about which credentials to use to access the data

### runway_actuals_data_set@DB
runway table data that will override values in the **MFS_data_set@DB**
The sub-parameters of runway_actuals_data_set@DB are the same as MFS_data_set@DB, although their values could differ.

### fraction_speed_gte_threshold_data_set@DB:
Data request that aims to return the fraction of time a flight has a ground speed greater than a threshold speed, it queries ASDEX data from the matm_flight_all and matm_flight_summary tables.
The sub-parameters of fraction_speed_gte_threshold_data_set@DB are the similar to MFS_data_set@DB, although their values could differ. There is an additional entry to the **load_args.params** : **threshold_knots** which is the minimum taxi speed.

### ffs_data_set@DB
STBO data query on flight_summary_*_v3_1 tables (where * is kdfw or kclt), to gain some knowledged in STBO estimated unimpeded taxi time. The sub-parameters are similar to MFS_data_set@DB, there is one less entry in the **load_args.params** : **airport_icao** since the table is airport-specific.


### [MFS_data_set, runway_actuals_data_set, fraction_speed_gte_threshold_data_set, ffs_data_set]@CSV
Same as previous ones but saved in a local csv file.

### _outcsv
Template for output of training/testing predictions as a csv file

### _model
Template for output of models as a pickle file


### data_predicted_*_baseline
csv file containing the final dataframe calculated by a pipeline, including its prediction.

### [imp/unimp]_*_model_pipeline
pickle file containing the fitted model