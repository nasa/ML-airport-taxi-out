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

from kedro.pipeline import Pipeline, node
from .nodes import *
from .compute_surface_counts_departures import *



def create_pipeline(**kwargs):
    unimp_pip_core_nodes =  Pipeline(
        [
            node(
                func=replace_runway_actuals,
                inputs=["MFS_data_set@CSV", "runway_actuals_data_set@CSV"],
                outputs="data_init",
            ),
            node(
                func=merge_STBO,
                inputs=["data_init", "ffs_data_set@CSV"],
                outputs="data_0"
            ),
            node(
                func=keep_second_of_duplicate_gufi,
                inputs="data_0",
                outputs="data_1",
            ),
            node(
                func=compute_arrival_departure_count_at_pushback,
                inputs="data_1",
                outputs="data_2",
            ),
            node(
                func=set_index,
                inputs="data_2",
                outputs="data_3"
            ),
            node(
                func=compute_total_taxi_time,
                inputs="data_3",
                outputs="data_4"
            ),
            node(
                func=check_and_filter_start_end_dates,
                inputs=["data_4","params:globals"],
                outputs="data_5"
            ),
            node(
                func=add_train_test_group,
                inputs=["data_5", "params:TEST_SIZE", "params:RANDOM_SEED", "parameters"],
                outputs="data_engred_general",
            ),
        ]
    )

    unimp_extra_nodes = Pipeline(
        [
            node(
                func=apply_filter_only_departures,
                inputs="data_engred_general",
                outputs="data_engred_general_departures",
            ),
        ]
    )

    unimp_full_extra_nodes = Pipeline(
        [
            node(
            func=apply_filter_req_dep_stand_and_runway,
                inputs=["data_engred_general_departures","params:unimp_full_model_params",
                        "params:freight_airlines"],
                outputs="data_engred_general_departures_filtered",
            ),
            node(
                func=apply_filter_req_niqr_dep_full_taxi_times,
                inputs=[
                    "data_engred_general_departures_filtered",
                    "params:NIQR_FULL_TAXI",
                ],
                outputs="data_full_niqrfiltered_unimp",
            ),
            node(
                func=apply_filter_null_times,
                inputs= [ "data_full_niqrfiltered_unimp",
                          "params:full_non_null"
                         ],
                outputs="data_engred_general_departures_non_null_full"
            ), 
            node(
                func=set_index,
                inputs="fraction_speed_gte_threshold_data_set@CSV",
                outputs="fraction_speed_gte_threshold",
            ),
            node(
                func=join_fraction_speed_gte_threshold_and_filter,
                inputs=[
                    "data_engred_general_departures_non_null_full",
                    "fraction_speed_gte_threshold"
                ],
                outputs="data_joined_unimp_full",
            ),
            node(
                func=apply_filter_surface_counts_full,
                inputs=["data_joined_unimp_full","parameters"],
                outputs="data_surface_filter_full"
            ),            
            node(
                func=calculate_unimpeded_AMA,
                inputs=[
                    "data_surface_filter_full",
                    "params:unimp_full_model_params",
                ],
                outputs="data_engred_unimp_full",
            ),
            node(
                func=train_test_group_logging,
                inputs="data_engred_unimp_full",
                outputs=None,
            ),
        ]
    )


    unimp_ramp_extra_nodes = Pipeline(
        [
            node(
            func=apply_filter_req_dep_stand_and_runway,
                inputs=["data_engred_general_departures","params:unimp_ramp_model_params",
                        "params:freight_airlines"],
                outputs="data_engred_general_departures_filtered",
            ),
            node(
                func=apply_filter_req_niqr_dep_ramp_taxi_times,
                inputs=[
                    "data_engred_general_departures_filtered",
                    "params:NIQR_RAMP_TAXI",
                ],
                outputs="data_ramp_niqrfiltered_unimp",
            ),
            node(
                func=apply_filter_surface_counts_ramp,
                inputs=["data_ramp_niqrfiltered_unimp","parameters"],
                outputs="data_surface_filter_ramp"
            ),            
            node(
                func=apply_filter_null_times,
                inputs= [ "data_surface_filter_ramp",
                          "params:ramp_non_null"
                         ],
                outputs="data_engred_unimp_ramp"
            ), 
            node(
                func=train_test_group_logging,
                inputs="data_engred_unimp_ramp",
                outputs=None,
            ),
        ]
    )

    unimp_ama_extra_nodes = Pipeline(
        [
            node(
            func=apply_filter_req_dep_stand_and_runway,
                inputs=["data_engred_general_departures","params:unimp_ama_model_params",
                        "params:freight_airlines"],
                outputs="data_engred_general_departures_filtered",
            ),
            node(
                func=apply_filter_req_niqr_dep_ama_taxi_times,
                inputs=["data_engred_general_departures_filtered",
                        "params:NIQR_AMA_TAXI",
                ],
                outputs="data_ama_niqrfiltered_unimp",
            ),
            node(
                func=apply_filter_null_times,
                inputs= [ "data_ama_niqrfiltered_unimp",
                          "params:ama_non_null"
                         ],
                outputs="data_engred_general_departures_non_null_ama"
            ),
            node(
                func=set_index,
                inputs="fraction_speed_gte_threshold_data_set@CSV",
                outputs="fraction_speed_gte_threshold",
            ),
            node(
                func=join_fraction_speed_gte_threshold_and_filter,
                inputs=[
                    "data_engred_general_departures_non_null_ama",
                    "fraction_speed_gte_threshold"
                ],
                outputs="data_joined_unimp_ama",
            ),
            node(
                func=apply_filter_surface_counts_ama,
                inputs=["data_joined_unimp_ama","parameters"],
                outputs="data_surface_filter_ama"
            ),            
            node(
                func=calculate_unimpeded_AMA,
                inputs=[
                    "data_surface_filter_ama",
                    "params:unimp_ama_model_params",
                ],
                outputs="data_engred_unimp_ama",
            ),
            node(
                func=train_test_group_logging,
                inputs="data_engred_unimp_ama",
                outputs=None,
            ),
        ]
    )

    return {
        'unimp_full_pip':
        unimp_pip_core_nodes+
        unimp_extra_nodes+
        unimp_full_extra_nodes,
        'unimp_ramp_pip':
        unimp_pip_core_nodes+
        unimp_extra_nodes+
        unimp_ramp_extra_nodes,
        'unimp_ama_pip':
        unimp_pip_core_nodes+
        unimp_extra_nodes+
        unimp_ama_extra_nodes,
    }
