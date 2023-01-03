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

import csv
import json
import pkgutil

from etils import epath
import tensorflow as tf
import tensorflow_datasets as tfds

from psiz_datasets.utils import append_rank_placeholder
from psiz_datasets.utils import one_hot
from psiz_datasets.utils import parse_rank_timestep


_VERSION = tfds.core.Version('1.0.0')
_DESCRIPTION = pkgutil.get_data(__name__, 'DESCRIPTION.md').decode("utf-8")
_HOMEPAGE = 'https://psiz.readthedocs.io/en/latest/src/datasets/datasets.html'
_CITATION = pkgutil.get_data(__name__, 'CITATIONS.bib').decode("utf-8")
_N_OUTCOME_2RANK1 = 2
_N_OUTCOME_8RANK2 = 56
_MAX_TIMESTEP = 120


def read_metadata_file(path):
    """Reads the tab-separated metadata from the path."""
    base = 36
    metadata = {}
    with epath.Path(path).open() as f:
        reader = csv.DictReader(f, delimiter="|")
        for row in reader:
            # Collect metadata for each asset.
            stimulus_id = int(row['local_id'], base)
            metadata[stimulus_id] = {
                "filepath": row["filepath"],
                "common_name": row["common_name"],
                "taxonomic_family": row["taxonomic_family"],
            }
    return metadata


class Birds16Rank2019Config(tfds.core.BuilderConfig):
    """BuilderConfig for Birds16Rank2019."""

    def __init__(self, *, with_timestep_axis=None, **kwargs):
        """BuilderConfig for Birds16Rank2019.

        Args:
            with_timestep_axis: Boolean indicating if dataset should be
                returned with a timestep axis. If `True`, dataset
                includes timestep axis.
            **kwargs: keyword arguments forwarded to super.

        """
        super(Birds16Rank2019Config, self).__init__(version=_VERSION, **kwargs)
        self.with_timestep_axis = with_timestep_axis
        self.max_timestep = _MAX_TIMESTEP
        self.data_url = 'https://osf.io/43rmk/download'


class Birds16Rank2019(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for birds16_rank2019 dataset."""

    BUILDER_CONFIGS = [
        Birds16Rank2019Config(
            name="with_timestep",
            description=(
                "Human-provided ranked similarity judgments for the Birds16 "
                "image dataset. Ranked judgments were first published in "
                "Roads & Mozer, 2019. Dataset is formatted with a timestep "
                "axis to facilitate sequence modeling."
            ),
            with_timestep_axis=True
        ),
        Birds16Rank2019Config(
            name="without_timestep",
            description=(
                "Human-provided ranked similarity judgments for the Birds16 "
                "image dataset. Ranked judgments were first published in "
                "Roads & Mozer, 2019. Dataset is formatted without a "
                "timestep axis by 'unrolling' all sequences and dropping "
                "placeholder trials."
            ),
            with_timestep_axis=False,
        ),
    ]
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
    }

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        encoding = tfds.features.Encoding.NONE
        if self.builder_config.name == 'with_timestep':
            features = tfds.features.FeaturesDict({
                'given2rank1_stimulus_set': tfds.features.Sequence(
                    tfds.features.Tensor(
                        shape=(3,), dtype=tf.int32, encoding=encoding
                    ), length=self.builder_config.max_timestep
                ),
                'given8rank2_stimulus_set': tfds.features.Sequence(
                    tfds.features.Tensor(
                        shape=(9,), dtype=tf.int32, encoding=encoding
                    ), length=self.builder_config.max_timestep
                ),
                'anonymous_id': tfds.features.Sequence(
                    tfds.features.Text(),
                    length=self.builder_config.max_timestep
                ),
                'given2rank1_outcome': tfds.features.Sequence(
                    tfds.features.Tensor(
                        shape=(_N_OUTCOME_2RANK1,), dtype=tf.float32,
                        encoding=encoding
                    ), length=self.builder_config.max_timestep
                ),
                'given2rank1_response_time_ms': tfds.features.Sequence(
                    tf.float32, length=self.builder_config.max_timestep
                ),
                'given2rank1_sample_weight': tfds.features.Sequence(
                    tf.float32, length=self.builder_config.max_timestep
                ),
                'given8rank2_outcome': tfds.features.Sequence(
                    tfds.features.Tensor(
                        shape=(_N_OUTCOME_8RANK2,), dtype=tf.float32,
                        encoding=encoding
                    ), length=self.builder_config.max_timestep
                ),
                'given8rank2_response_time_ms': tfds.features.Sequence(
                    tf.float32, length=self.builder_config.max_timestep
                ),
                'given8rank2_sample_weight': tfds.features.Sequence(
                    tf.float32, length=self.builder_config.max_timestep
                )
            })
            supervised_keys = (
                {
                    'given2rank1_stimulus_set': 'given2rank1_stimulus_set',
                    'given8rank2_stimulus_set': 'given8rank2_stimulus_set',
                    'anonymous_id': 'anonymous_id',
                },
                {
                    'given2rank1_outcome': 'given2rank1_outcome',
                    'given2rank1_response_time_ms':
                        'given2rank1_response_time_ms',
                    'given8rank2_outcome': 'given8rank2_outcome',
                    'given8rank2_response_time_ms':
                        'given8rank2_response_time_ms',
                },
                {
                    'given2rank1_sample_weight': 'given2rank1_sample_weight',
                    'given8rank2_sample_weight': 'given8rank2_sample_weight',
                }
            )
        elif self.builder_config.name == 'without_timestep':
            features = tfds.features.FeaturesDict({
                'given2rank1_stimulus_set': tfds.features.Tensor(
                    shape=(3,), dtype=tf.int32, encoding=encoding
                ),
                'given8rank2_stimulus_set': tfds.features.Tensor(
                    shape=(9,), dtype=tf.int32, encoding=encoding
                ),
                'anonymous_id': tfds.features.Text(),
                'given2rank1_outcome': tfds.features.Tensor(
                    shape=(_N_OUTCOME_2RANK1,), dtype=tf.float32,
                    encoding=encoding
                ),
                'given2rank1_response_time_ms': tf.float32,
                'given2rank1_sample_weight': tf.float32,
                'given8rank2_outcome': tfds.features.Tensor(
                    shape=(_N_OUTCOME_8RANK2,), dtype=tf.float32,
                    encoding=encoding
                ),
                'given8rank2_response_time_ms': tf.float32,
                'given8rank2_sample_weight': tf.float32,
            })
            supervised_keys = (
                {
                    'given2rank1_stimulus_set': 'given2rank1_stimulus_set',
                    'given8rank2_stimulus_set': 'given8rank2_stimulus_set',
                    'anonymous_id': 'anonymous_id',
                },
                {
                    'given2rank1_outcome': 'given2rank1_outcome',
                    'given2rank1_response_time_ms':
                        'given2rank1_response_time_ms',
                    'given8rank2_outcome': 'given8rank2_outcome',
                    'given8rank2_response_time_ms':
                        'given8rank2_response_time_ms',
                },
                {
                    'given2rank1_sample_weight': 'given2rank1_sample_weight',
                    'given8rank2_sample_weight': 'given8rank2_sample_weight',
                }
            )

        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=features,
            supervised_keys=supervised_keys,
            homepage=_HOMEPAGE,
            citation=_CITATION,
            # Empty metadata object that will be filled dynamically.
            metadata=tfds.core.MetadataDict()
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        extracted_path = dl_manager.download_and_extract(
            self.builder_config.data_url
        )

        # Load metadata file.
        metadata_file = extracted_path / 'stimuli.txt'
        self.info.metadata["stimuli"] = read_metadata_file(metadata_file)

        return {
            'train': self._generate_examples(extracted_path / 'train_seqs'),
        }

    def _generate_examples(self, seqs_path):
        """Yields examples."""
        builder_config = self.builder_config

        for sequence_path in seqs_path.glob('seq_*.json'):
            data = json.loads(sequence_path.read_text())
            # version = data['version']
            data = data['data'][0]
            anonymous_id = data['anonymous_id']
            # design_id = data['design_id']
            # project = data['project']
            # protocol = data['protocol']
            grade = data['grade']
            sequence = data['sequence']

            groups = {
                'anonymous_id': anonymous_id,
            }
            formatted_sequence = format_sequence(groups, grade, sequence)

            # Pad sequence if preserving timestep axis.
            if builder_config.with_timestep_axis:
                formatted_sequence = pad_sequence(
                    builder_config, formatted_sequence, groups
                )
                example_id = data["sequence_id"]
                yield example_id, formatted_sequence
            else:
                # Yield single timesteps.
                example_id_prefix = "{0}".format(data["sequence_id"])

                n_timestep = len(formatted_sequence['anonymous_id'])
                for idx_timestep in range(n_timestep):
                    example_id = (
                        example_id_prefix + "/{0}".format(idx_timestep)
                    )
                    formatted_timestep = select_timestep(
                        formatted_sequence, idx_timestep
                    )
                    yield example_id, formatted_timestep


def select_timestep(sequence, idx_timestep):
    timestep = {}
    for key, value in sequence.items():
        timestep[key] = value[idx_timestep]
    return timestep


def format_sequence(groups, grade, sequence):
    """Format sequence."""
    formatted_sequence = None
    for timestep in sequence:
        formatted_sequence = format_timestep(
            formatted_sequence, timestep, groups, grade
        )
    return formatted_sequence


def format_timestep(formatted_sequence, timestep, groups, grade):
    """Format timestep."""
    if formatted_sequence is None:
        formatted_sequence = {
            'given8rank2_stimulus_set': [],
            'given2rank1_stimulus_set': [],
            'anonymous_id': [],
            'given8rank2_outcome': [],
            'given8rank2_response_time_ms': [],
            'given8rank2_sample_weight': [],
            'given2rank1_outcome': [],
            'given2rank1_response_time_ms': [],
            'given2rank1_sample_weight': [],
        }

    if timestep['kind'] == "rank:8rank2":
        stimulus_set, outcome_idx, rt_ms = parse_rank_timestep(timestep)

        # Handle content.
        formatted_sequence['given8rank2_stimulus_set'].append(stimulus_set)
        # Handle group information.
        formatted_sequence['anonymous_id'].append(groups['anonymous_id'])
        # Handle outcome.
        formatted_sequence['given8rank2_outcome'].append(
            one_hot(outcome_idx, _N_OUTCOME_8RANK2)
        )
        formatted_sequence[
            'given8rank2_response_time_ms'
        ].append(rt_ms['total'])
        formatted_sequence['given8rank2_sample_weight'].append(grade / 100)

        # Handle placeholder trials.
        formatted_sequence = append_rank_placeholder(
            formatted_sequence, n_reference=2, n_select=1
        )

    elif timestep['kind'] == "rank:2rank1":
        stimulus_set, outcome_idx, rt_ms = parse_rank_timestep(timestep)

        # Handle content.
        formatted_sequence['given2rank1_stimulus_set'].append(stimulus_set)
        # Handle group information.
        formatted_sequence['anonymous_id'].append(groups['anonymous_id'])
        # Handle outcome.
        formatted_sequence['given2rank1_outcome'].append(
            one_hot(outcome_idx, _N_OUTCOME_2RANK1)
        )
        formatted_sequence[
            'given2rank1_response_time_ms'
        ].append(rt_ms['total'])
        formatted_sequence['given2rank1_sample_weight'].append(grade / 100)

        # Handle placeholder trials.
        formatted_sequence = append_rank_placeholder(
            formatted_sequence, n_reference=8, n_select=2
        )

    elif timestep['kind'] == "questionnaire:feedback":
        pass
    else:
        raise NotImplementedError(
            "Unrecognized timestep['kind']={}".format(
                timestep['kind']
            )
        )

    return formatted_sequence


def pad_sequence(builder_config, formatted_sequence, groups):
    """Pad sequence with placeholder timesteps."""
    n_timestep = len(formatted_sequence['anonymous_id'])
    n_timestep_pad = builder_config.max_timestep - n_timestep
    for _ in range(n_timestep_pad):
        formatted_sequence = append_rank_placeholder(
            formatted_sequence, n_reference=2, n_select=1
        )
        formatted_sequence = append_rank_placeholder(
            formatted_sequence, n_reference=8, n_select=2
        )
        formatted_sequence['anonymous_id'].append(groups['anonymous_id'])
    return formatted_sequence
