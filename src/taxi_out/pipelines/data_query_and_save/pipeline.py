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
This is a boilerplate pipeline 'data_query_and_save'
generated using Kedro 0.16.2
"""

from kedro.pipeline import Pipeline, node
from .nodes import *


def create_pipeline(**kwargs):
    unimp_data_sql = Pipeline([
        node(
            func= lambda x,y : [x,y],
            inputs=[
                "MFS_data_set@DB","runway_actuals_data_set@DB"
            ],
            outputs=[
                "MFS_data_set@CSV","runway_actuals_data_set@CSV"
            ],
        ),

    ])


    unimp_threshold_data_sql = Pipeline(
        [
            node(
                func= lambda x : [x],
                inputs=[
                    "fraction_speed_gte_threshold_data_set@DB"
                ],
                outputs=[
                    "fraction_speed_gte_threshold_data_set@CSV"
                ],
            ),
        ]
    )

    unimp_STBO_data_sql = Pipeline(
        [
            node(
                func= lambda x : [x],
                inputs=[
                    "ffs_data_set@DB"
                ],
                outputs=[
                    "ffs_data_set@CSV"
                ],
            ),
        ]
    )


    return {
        'unimp_data_sql': unimp_data_sql,
        'unimp_threshold_data_sql': unimp_threshold_data_sql,
        'unimp_STBO_data_sql': unimp_STBO_data_sql
    }
