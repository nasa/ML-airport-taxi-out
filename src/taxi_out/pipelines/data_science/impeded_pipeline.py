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

from kedro.pipeline import Pipeline, node
from .nodes import *
from data_services.mlflow_utils import init_mlflow, init_mlflow_run
from data_services.analytics_visualization_artifacts import visualization_caller
from data_services.network import copy_artifacts_to_ntx


def create_pipeline(**kwargs):
   
    imp_full_pipeline = Pipeline(
        [
            node(
                func=init_mlflow,
                inputs="params:imp_full_model_params",
                outputs="experiment_id",
           ),
            node(
                func=init_mlflow_run,
                inputs=[
                    "params:imp_full_model_params",
                    "experiment_id",
                ],
                outputs="active_run_id",
            ),
            node(
                func=define_imp_full_model,
                inputs=[
                    "data_engred_imp_full",
                    "params:imp_full_model_params",
                    "params:globals",
                    "params:RANDOM_SEED",
                ],
                outputs="untrained_imp_full_model_pipeline"
            ),
            node(
                func=optimize_imp_full_model,
                inputs=[
                    "data_engred_imp_full",
                    "params:imp_full_model_params",
                    "params:globals",
                    "active_run_id",
                    "params:RANDOM_SEED",
                    "params:hypertune",
                    "untrained_imp_full_model_pipeline"
                ],
                outputs=[
                    "best_model_params",
                    "best_untrained_imp_full_model_pipeline",
                ]
                ),
            node(
                func=train_imp_full_model,
                inputs=[
                    "data_engred_imp_full",
                    "best_model_params",
                    "params:globals",
                    "active_run_id",
                    "params:RANDOM_SEED",
                    "best_untrained_imp_full_model_pipeline"
                ],
                outputs="imp_full_model_pipeline",
                ),
            node(
                func=predict,
                inputs=[
                    "imp_full_model_pipeline",
                    "data_engred_imp_full",
                    "params:imp_full_model_params",
                ],
                outputs="data_predicted_full_taxi",
            ),
            node(
                func=train_imp_full_baseline,
                inputs=[
                    "data_engred_imp_full",
                    "params:imp_full_model_params",
                    "params:globals",
                    "active_run_id",
                ],
                outputs="baseline_pipeline",
            ),
            node(
                func=predict_baseline,
                inputs=[
                    "baseline_pipeline",
                    "data_predicted_full_taxi", # full_taxi because we want to add more preds to the dataframe
                    "params:imp_full_model_params",
                ],
                outputs="data_predicted_full_baseline",
            ),
            node(
                func=report_performance_metrics,
                inputs=[
                    "data_predicted_full_baseline",
                    "params:imp_full_model_params",
                    "active_run_id",
                ],
                outputs=None,
            ),
            node(
                func=visualization_caller,
                inputs=[
                    "data_predicted_full_baseline",
                    "baseline_pipeline",
                    "params:imp_full_model_params",
                    "params:globals",
                    "active_run_id",
                ],
                outputs="artifacts_ready",
            )
            ,
            node(
                func=copy_artifacts_to_ntx,
                inputs=[
                    "experiment_id",
                    "active_run_id",
                    "params:ntx_connection",
                    "artifacts_ready",
                ],
                outputs=None,
            )
        ]
    )

    imp_ama_pipeline = Pipeline(
        [
            node(
                func=init_mlflow,
                inputs="params:imp_ama_model_params",
                outputs="experiment_id",
            ),
            node(
                func=init_mlflow_run,
                inputs=[
                    "params:imp_ama_model_params",
                    "experiment_id",
                ],
                outputs="active_run_id",
            ),
            node(
                func=define_imp_ama_model,
                inputs=[
                    "data_engred_imp_ama", # verify data_engred_imp_ama exists
                    "params:imp_ama_model_params",
                    "params:globals",
                    "params:RANDOM_SEED",
                ],
                outputs="untrained_imp_ama_model_pipeline"
            ),
            node(
                func=optimize_imp_ama_model,
                inputs=[
                    "data_engred_imp_ama",
                    "params:imp_ama_model_params",
                    "params:globals",
                    "active_run_id",
                    "params:RANDOM_SEED",
                    "params:hypertune",
                    "untrained_imp_ama_model_pipeline"
                ],
                outputs=[
                    "best_model_params",
                    "best_untrained_imp_ama_model_pipeline",
                ]
            ),
            node(
                func=train_imp_ama_model,
                inputs=[
                    "data_engred_imp_ama",
                    "best_model_params",
                    "params:globals",
                    "active_run_id",
                    "params:RANDOM_SEED",
                    "best_untrained_imp_ama_model_pipeline"
                ],
                outputs="imp_ama_model_pipeline",
            ),
            node(
                func=predict,
                inputs=[
                    "imp_ama_model_pipeline",
                    "data_engred_imp_ama",
                    "params:imp_ama_model_params",
                ],
                outputs="data_predicted_ama_taxi",
            ),
            node(
                func=train_imp_ama_baseline,
                inputs=[
                    "data_engred_imp_ama",
                    "params:imp_ama_model_params",
                    "params:globals",
                    "active_run_id",
                ],
                outputs="baseline_pipeline",
            ),
            node(
                func=predict_baseline,
                inputs=[
                    "baseline_pipeline",
                    "data_predicted_ama_taxi", 
                    "params:imp_ama_model_params",
                ],
                outputs="data_predicted_ama_baseline",
            ),
            node(
                func=report_performance_metrics,
                inputs=[
                    "data_predicted_ama_baseline",
                    "params:imp_ama_model_params",
                    "active_run_id",
                ],
                outputs=None,
            ),
            node(
                func=visualization_caller,
                inputs=[
                    "data_predicted_ama_baseline",
                    "baseline_pipeline",
                    "params:imp_ama_model_params",
                    "params:globals",
                    "active_run_id",
                ],
                outputs="artifacts_ready",
            )
            ,
            node(
                func=copy_artifacts_to_ntx,
                inputs=[
                    "experiment_id",
                    "active_run_id",
                    "params:ntx_connection",
                    "artifacts_ready",
                ],
                outputs=None,
            )
             
        ]
    )


    imp_ramp_pipeline = Pipeline(
        [
            node(
                func=init_mlflow,
                inputs="params:imp_ramp_model_params",
                outputs="experiment_id",
            ),
            node(
                func=init_mlflow_run,
                inputs=[
                    "params:imp_ramp_model_params",
                    "experiment_id",
                ],
                outputs="active_run_id",
            ),
            node(
                func=define_imp_ramp_model,
                inputs=[
                    "data_engred_imp_ramp",
                    "params:imp_ramp_model_params",
                    "params:globals",
                    "params:RANDOM_SEED",
                ],
                outputs="untrained_imp_ramp_model_pipeline"
            ),
            node(
                func=optimize_imp_ramp_model,
                inputs=[
                    "data_engred_imp_ramp",
                    "params:imp_ramp_model_params",
                    "params:globals",
                    "active_run_id",
                    "params:RANDOM_SEED",
                    "params:hypertune",
                    "untrained_imp_ramp_model_pipeline"
                ],
                outputs=[
                    "best_model_params",
                    "best_untrained_imp_ramp_model_pipeline",
                ]
                ),
            node(
                func=train_imp_ramp_model,
                inputs=[
                    "data_engred_imp_ramp",
                    "best_model_params",
                    "params:globals",
                    "active_run_id",
                    "params:RANDOM_SEED",
                    "best_untrained_imp_ramp_model_pipeline"
                ],
                outputs="imp_ramp_model_pipeline",
                ),
            node(
                func=predict,
                inputs=[
                    "imp_ramp_model_pipeline",
                    "data_engred_imp_ramp",
                    "params:imp_ramp_model_params",
                ],
                outputs="data_predicted_ramp_taxi",
            ),
            node(
                func=train_imp_ramp_baseline,
                inputs=[
                    "data_engred_imp_ramp",
                    "params:imp_ramp_model_params",
                    "params:globals",
                    "active_run_id",
                ],
                outputs="baseline_pipeline",
            ),
            node(
                func=predict_baseline,
                inputs=[
                    "baseline_pipeline",
                    "data_predicted_ramp_taxi", 
                    "params:imp_ramp_model_params",
                ],
                outputs="data_predicted_ramp_baseline",
            ),
            node(
                func=report_performance_metrics,
                inputs=[
                    "data_predicted_ramp_baseline",
                    "params:imp_ramp_model_params",
                    "active_run_id",
                ],
                outputs=None,
            ),
            node(
                func=visualization_caller,
                inputs=[
                    "data_predicted_ramp_baseline",
                    "baseline_pipeline",
                    "params:imp_ramp_model_params",
                    "params:globals",
                    "active_run_id",
                ],
                outputs="artifacts_ready",
            )
            ,
            node(
                func=copy_artifacts_to_ntx,
                inputs=[
                    "experiment_id",
                    "active_run_id",
                    "params:ntx_connection",
                    "artifacts_ready",
                ],
                outputs=None,
            )
        ]
    )

    
    return {
        'imp_full': imp_full_pipeline,
        'imp_ama' : imp_ama_pipeline,
        'imp_ramp' : imp_ramp_pipeline,
        }
