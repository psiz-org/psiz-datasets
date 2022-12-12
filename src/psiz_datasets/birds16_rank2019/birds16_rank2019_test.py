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
"""birds16_rank2019 dataset."""

import tensorflow_datasets as tfds
from psiz_datasets import birds16_rank2019


class Birds16Rank2019WithTimestepTest(
    tfds.testing.DatasetBuilderTestCase
):
    """Tests for birds16_rank2019 dataset."""
    DATASET_CLASS = birds16_rank2019.Birds16Rank2019
    BUILDER_CONFIG_NAMES_TO_TEST = ["with_timestep"]
    SPLITS = {
        'train': 2,
    }


class Birds16Rank2019WithoutTimestepTest(
    tfds.testing.DatasetBuilderTestCase
):
    """Tests for birds16_rank2019 dataset."""
    DATASET_CLASS = birds16_rank2019.Birds16Rank2019
    BUILDER_CONFIG_NAMES_TO_TEST = ["without_timestep"]
    SPLITS = {
        'train': 60,
    }


if __name__ == '__main__':
    tfds.testing.test_main()
