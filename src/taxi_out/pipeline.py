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

"""Construction of the master pipeline.
"""

from typing import Dict

from kedro.pipeline import Pipeline

from taxi_out.pipelines import data_query_and_save as dqs
from taxi_out.pipelines import data_engineering as de
from taxi_out.pipelines import data_science as ds

from data_services.conda_environment_test import check_environment

def create_pipelines(**kwargs) -> Dict[str, Pipeline]:
    """Create the project's pipeline.

    Args:
        kwargs: Ignore any additional arguments added in the future.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.

    """

    check_environment()

    
    dqs_pipelines = dqs.create_pipeline()
    de_pipelines = de.create_pipeline()  
    ds_pipelines = ds.create_pipeline()
    
    return {
        "dqs" : dqs_pipelines['unimp_data_sql'] + dqs_pipelines['unimp_threshold_data_sql'],
        "dqs_STBO" : dqs_pipelines['unimp_data_sql'] + dqs_pipelines['unimp_threshold_data_sql']+ dqs_pipelines['unimp_STBO_data_sql'],
        "dqs_nothreshold" : dqs_pipelines['unimp_data_sql'] + dqs_pipelines['unimp_STBO_data_sql'],
        "dqs_threshold" : dqs_pipelines['unimp_threshold_data_sql'],
        "dqs_unimp_data" : dqs_pipelines['unimp_data_sql'],
        "dqs_only_STBO" : dqs_pipelines['unimp_STBO_data_sql'],  
#    	"de" : de_pipelines['unimp_full_pip'] + de_pipelines['unimp_ramp_pip'] + de_pipelines['unimp_ama_pip'] + de_pipelines['imp_ama_pip'] + de_pipelines['imp_full_pip'] + de_pipelines['imp_ramp_pip'],
        "unimp_full_de" : de_pipelines['unimp_full_pip'],
        "unimp_full_taxi" : de_pipelines['unimp_full_pip'] + ds_pipelines['unimp_full'],
        "unimp_ama_de" : de_pipelines['unimp_ama_pip'],
        "unimp_ama_taxi" : de_pipelines['unimp_ama_pip'] + ds_pipelines['unimp_ama'],
        "unimp_ramp_de" : de_pipelines['unimp_ramp_pip'],
        "unimp_ramp_taxi" : de_pipelines['unimp_ramp_pip'] + ds_pipelines['unimp_ramp'],
        "imp_ama_de" : de_pipelines['imp_ama_pip'],
        "imp_full_de" : de_pipelines['imp_full_pip'],
        "imp_ramp_de" : de_pipelines["imp_ramp_pip"],
        "imp_ama_taxi" : de_pipelines["imp_ama_pip"] + ds_pipelines["imp_ama"],
        "imp_full_taxi" : de_pipelines["imp_full_pip"] + ds_pipelines["imp_full"],
        "imp_ramp_taxi" : de_pipelines["imp_ramp_pip"] + ds_pipelines["imp_ramp"],        
        "__default__": dqs_pipelines
    }

