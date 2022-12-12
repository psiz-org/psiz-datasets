# -*- coding: utf-8 -*-
# Copyright 2022 The PsiZ Datasets Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Module of utility functions.

Functions:
    parse_asset_id: Parse asset ID.

"""


def parse_asset_id(asset_id):
    """Parse asset ID.

    Args:
        asset_id: A string denoting the asset ID.

    Returns:
        tuple(domain_id, subdomain_id, local_id)

    """
    # Settings.
    domain_length = 2
    subdomain_length = 3

    domain_id = asset_id[0:domain_length]
    subdomain_id = asset_id[domain_length:(domain_length + subdomain_length)]
    local_id = asset_id[(domain_length + subdomain_length):]
    return domain_id, subdomain_id, local_id
