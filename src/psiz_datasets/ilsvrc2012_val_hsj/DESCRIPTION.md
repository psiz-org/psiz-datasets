**ILSVRC-2012-val HSJ** dataset of human similarity judgments.

Human participants viewed trials composed of multiple images taken from the ILSVRC-2012 validation image dataset. The ILSVRC-2012 validation set is composed of 50,000 images from 1000 classes.

Participants were shown *8-rank-2* trials. The dataset is formatted into three pieces: the *content* of the trial, the *group* membership of the participant, and the participant's observed *behavior*.

* Content:
    * `_stimulus_set`: An array of indices that map to images. Indexing starts at `1`.
* Group:
    * `_anonymous_id`: A unique anonymous identifier for each participant.
* Behavior:
    * `_outcome`: A one-hot encoding representing the unique outcome of the participant's selection(s).
    * `_sample_weight`: The weight of a trial, which is derived from participant performance on catch trials.
