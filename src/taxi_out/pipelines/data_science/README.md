# Pipeline data_science

> *Note:* This is a `README.md` boilerplate generated using `Kedro 0.16.2`.

## Overview

<!---
Please describe your modular pipeline here.
-->

This pipeline processes the data from data_engineering
and for now only applies some feature engineering to these
data.\




## Pipeline inputs 
</br>
<!---
The list of pipeline inputs.
-->

### `data_engred_general`
|      |                    |
| ---- | ------------------ |
| Type | ``pandas.DataFrame`` |
| Description | DataFrame containing a row for each selected flights with multiple columns|
</br>


### `params:unimp_taxi_model_params`
|      |                    |
| ---- | ------------------ |
| Type | ``Dict``           |
| Description | Dictionary containing feat. eng. and model parameters|
</br>

### `params:globals`
|      |                    |
| ---- | ------------------ |
| Type | ``Dict``           |
| Description | Dictionary containing general parameters |
</br>
</br>




## Pipeline outputs
</br>
<!---
The list of pipeline outputs.
-->

### `unimp_taxi_model_pipeline`
|      |                    |
| ---- | ------------------ |
| Type | ``data_services.FilterPipeline``           |
| Description | Pipeline of feat. eng. and model with filtering


