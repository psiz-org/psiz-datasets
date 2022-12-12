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
    parse_rank_timestep: Parse rank trial from timestep data.

"""

import numpy as np
from psiz.data import Rank

from psiz_datasets.utils.parse_asset_id import parse_asset_id


def parse_rank_timestep(timestep):
    """Parse rank trial from timestep data.

    Args:
        timestep: A dictionary of timestep data for a rank trial.

    Returns:
        A tuple (stimulus_set, outcome_idx, rt_ms), where
            `stimulus_set` is a 1D array of stimulus indices,
            `outcome_idx` is an integer corresponding to the sparse
            categorical outcome, and `rt_ms` is a dictionary of
            response times.

    """
    # Settings
    base = 36

    references = []
    reference_order = []
    selections = []
    selection_order = []

    # Keep stimulus order preserved and infer outcome index.
    for interaction in timestep['interactions']:
        if interaction['kind'] == 'content:query':
            _, _, local_id = parse_asset_id(interaction['detail'])
            query = int(local_id, base)
        elif 'content:reference' in interaction['kind']:
            _, _, local_id = parse_asset_id(interaction['detail'])
            local_idx = int(local_id, base)
            references.append(local_idx)
            reference_order.append(int(interaction['kind'].split('_')[1]))
        elif 'behavior:rank' in interaction['kind']:
            _, _, local_id = parse_asset_id(interaction['detail'])
            local_idx = int(local_id, base)
            selections.append(local_idx)
            selection_order.append(int(interaction['kind'].split('_')[1]))

    # Convert lists to arrays.
    references = np.array(references, dtype=np.int32)
    reference_order = np.array(reference_order, dtype=np.int32)
    selections = np.array(selections, dtype=np.int32)
    selection_order = np.array(selection_order, dtype=np.int32)
    n_reference = len(references)

    # Make sure references and selections are in correct order. This should
    # already be true, but we are being careful.
    references = references[reference_order]
    selections = selections[selection_order]

    # Determine index locations of selections.
    dmy_idx = np.arange(n_reference)
    selection_indices = []
    for selection in selections:
        selection_indices.append(
            dmy_idx[np.equal(references, selection)][0]
        )

    # Finalize pieces.
    stimulus_set = np.hstack([np.array([query]), references]).astype(np.int32)
    outcome_idx = Rank.as_sparse_outcome(n_reference, selection_indices)
    rt_ms = {}
    rt_ms['total'] = timestep['response_time_ms']

    return stimulus_set, outcome_idx, rt_ms
