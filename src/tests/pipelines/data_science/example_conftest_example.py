import mlflow
import pytest
from kedro.config import TemplatedConfigLoader
from kedro.io import DataCatalog
from kedro.runner import SequentialRunner
from mlflow import sklearn, entities
import random
import string
from data_services.mlflow_utils import get_model_by_run_id
import pickle
from os import path
from kedro.pipeline import Pipeline
import importlib

@pytest.fixture(scope='module')
def run_name():
    """generating a unique name for each  test run
    """

    letters = string.ascii_lowercase
    return 'run-' + ''.join(random.choice(letters) for i in range(10))


@pytest.fixture(scope='module')
def init_mlflow(run_name, parameters, run_id):
    """a method that initiates an <experiment_name>_test associated with the model <experiment_name>.
    <experiment_name>_test contains unit test runs
    Each test run is linked to a run_id found in the <experiement_name>.
    Each test run is tagged with the modeler name, and the original run_id of the model passed from the original experiment.
    """
    mlflow.set_tracking_uri(parameters['mlflow']['tracking_uri'])
    mlflow.set_experiment(parameters['mlflow']['experiment_name']+'_test')

    mlflow.start_run(run_name=run_name)
    mlflow.set_tag('modeler_name', parameters['mlflow']['modeler_name'])
    mlflow.set_tag('original_run_id', run_id)
    mlflow.set_tag('run_type', 'unittest')

    for key, value in parameters.items():
        mlflow.log_param(key, value)


@pytest.fixture(scope='module')
def run_id(parameters):
    """a method that run the unit test for a  certain run_id if the run_id is passed in the parameters.yml.
    if the run_id is not specified, the most recent run for the experiment will be taken as the default value

        """

    run_id = parameters['unit_tests'].get('run_id', None)
    if not run_id:
        mlflow.set_tracking_uri(parameters['mlflow']['tracking_uri'])
        experiment = mlflow.get_experiment_by_name(parameters['mlflow']['experiment_name'])

        # The default ordering is to sort by start_time DESC, then run_id.
        runs = mlflow.search_runs([experiment.experiment_id], max_results=1)
        return runs.iloc[0].run_id

    if run_id:
        return run_id
    raise Exception('No run_id found in parameters and cannot find latest run in experiment {}'
                    .format(parameters['mlflow']['experiment_name']))


@pytest.fixture(scope='module')
def config():
    return TemplatedConfigLoader(['conf/base', 'conf/local'], globals_pattern='globals*')


@pytest.fixture(scope='module')
def parameters(config):
    return config.get('parameters.yml')


@pytest.fixture(scope='module')
def data(config, parameters):
    """ method returning  input data to run tests
        """

    catalog = DataCatalog.from_config(catalog=config.get('catalog*'), credentials=config.get('credentials*'))
    catalog.add_feed_dict({'parameters': parameters, 'params:model': parameters['model']})

    if 'nodes_to_run' in parameters['unit_tests']['input_data']:
        nodes_to_run = parameters['unit_tests']['input_data']['nodes_to_run']
        pipeline_ds = importlib.import_module(nodes_to_run['pipeline_path'])
        pipeline_nodes = pipeline_ds.create_pipelines()[nodes_to_run['pipeline']]._nodes_by_name

        nodes = []
        for v in pipeline_nodes:
            if v in nodes_to_run['nodes_name']:
                nodes.append(pipeline_nodes[v])

        data = SequentialRunner().run(Pipeline(nodes), catalog)[nodes_to_run['nodes_output']]
    else:
        data = catalog.load(parameters['unit_tests']['input_data']['catalog_item']).head()

    return data


@pytest.fixture(scope='module')
def trained_model(run_id,parameters):
    """ previously trained model downloaded from mlflow server
    """
    mlflow.set_tracking_uri(parameters['mlflow']['tracking_uri'])
    model_run = get_model_by_run_id(run_id)
    model_run_dir = path.dirname(model_run)
    model = sklearn.load_model(model_run_dir)

    return model


@pytest.fixture(scope='session')
def teardown():
    yield None
    mlflow.end_run()


def pytest_runtest_logreport(report):
    """a method that log failed test as a separate metric, each with a unique name
    """
    if report.when == 'call' and report.failed:
        mlflow.log_metric('test_failed', 1)
        mlflow.log_metric(report.head_line + '_failed', 1)
