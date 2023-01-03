**Birds-16 Rank-2019** dataset of human similarity judgments.

Human participants viewed trials composed of multiple images taken from the Birds-16 image dataset. The Birds-16 image dataset is composed of 208 images carefully selected from the CUB-200 dataset, yielding four taxonomic families, with four taxonomic species per family, and 13 images per species.

Participants were shown two types of trials: *2-rank-1* and *8-rank-2* trials. The dataset is formatted into three pieces: the *content* of the trial, the *group* membership of the participant, and the participant's observed *behavior*.

* Content:
    * `_stimulus_set`: An array of indices that map to images. Indexing starts at `1`.
* Group:
    * `_anonymous_id`: A unique anonymous identifier for each participant.
* Behavior:
    * `_outcome`: A one-hot encoding representing the unique outcome of the participant's selection(s).
    * `_sample_weight`: The weight of a trial, which is derived from participant performance on catch trials.
