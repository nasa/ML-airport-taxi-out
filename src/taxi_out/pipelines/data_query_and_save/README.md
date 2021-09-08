# Pipeline data_query_and_save

> *Note:* This is a `README.md` boilerplate generated using `Kedro 0.16.2`.

## Overview

<!---
Please describe your modular pipeline here.
-->

This pipeline issue SQL queries to two databases and
save the returned tables into CSV files using kedro transcoding
capability.
</br>


## Pipeline inputs
</br>
<!---
The list of pipeline inputs.
-->

### `MFS_data_set@DB`
|      |                    |
| ---- | ------------------ |
| Type | `data_services.kedro_extensions.io.sqlfile_dataset.SQLQueryFileChunkedDataSet` |
| Description | Input data from matm_flight_summary : arrival, departure time @ stand, spot and runway for params:airport_icao|
</br>

### `runway_actuals_data_set@DB`
|      |                    |
| ---- | ------------------ |
| Type | `data_services.kedro_extensions.io.sqlfile_dataset.SQLQueryFileChunkedDataSet` |
| Description | Input data from runway_actuals : arrival, depature time @ runway for params:airport_icao |
</br>
</br>

## Pipeline outputs
</br>
<!---
The list of pipeline outputs.
-->

### `MFS_data_set@CSV`
|      |                    |
| ---- | ------------------ |
| Type | `pandas.CSVDataSet`
| Description | Input data from matm_flight_summary : arrival, departure time @ stand, spot and runway for params:airport_icao|
</br>

### `runway_actuals_data_set@CSV`
|      |                    |
| ---- | ------------------ |
| Type | `pandas.CSVDataSet`
| Description | Input data from runway_actuals : arrival, depature time @ runway for params:airport_icao |
