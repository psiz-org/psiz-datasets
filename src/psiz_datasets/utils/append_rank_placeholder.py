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
    append_rank_placeholder: Append placeholder rank trial.

"""

import numpy as np

from psiz_datasets.utils.one_hot import one_hot

_N_OUTCOME_2RANK1 = 2
_N_OUTCOME_8RANK2 = 56


def append_rank_placeholder(
    formatted_sequence, n_reference=None, n_select=None
):
    """Append placeholder rank trial."""
    prefix = 'given{0}rank{1}'.format(n_reference, n_select)
    # Handle content.
    stimulus_set = np.zeros([n_reference + 1], dtype=np.int32)
    formatted_sequence[prefix + '_stimulus_set'].append(stimulus_set)

    # Handle outcome.
    # NOTE: Can not make outcome all zeros because it will result in nan's
    # when computing categorical crossentropy loss.
    outcome_idx = 0
    if n_reference == 8 and n_select == 2:
        n_outcome = _N_OUTCOME_8RANK2
    elif n_reference == 2 and n_select == 1:
        n_outcome = _N_OUTCOME_2RANK1
    else:
        raise NotImplementedError('Unrecognized rank configuration.')
    formatted_sequence[prefix + '_outcome'].append(
        one_hot(outcome_idx, n_outcome)
    )
    rt_ms = 0.0
    formatted_sequence[prefix + '_response_time_ms'].append(rt_ms)
    weight = 0.0
    formatted_sequence[prefix + '_sample_weight'].append(weight)

    return formatted_sequence
