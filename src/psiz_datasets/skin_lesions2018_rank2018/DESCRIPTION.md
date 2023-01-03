**Skin-Lesion-2018 Rank-2018** dataset of human similarity judgments.

Human participants viewed trials composed of multiple images taken from the Skin Lesions 2018 image dataset. The Skin Lesions 2018 image dataset is composed of 237 images. There are 120 melanoma images representing four types: acral lentiginous, lentigo maligna, nodular, superficial spreading. There are 117 benign images representing four types: blue nevi, lentigo, melanocytic nevi, seborrheic keratoses. Images were collected via Google image search and validated by an expert dermatologist. All images were scaled to fit within a frame of 300 × 300 pixels and cropped to remove any body part information.

Participants were shown *8-rank-2* trials. The dataset is formatted into three pieces: the *content* of the trial, the *group* membership of the participant, and the participant's observed *behavior*.

* Content:
    * `_stimulus_set`: An array of indices that map to images. Indexing starts at `1`.
* Group:
    * `_anonymous_id`: A unique anonymous identifier for each participant.
* Behavior:
    * `_outcome`: A one-hot encoding representing the unique outcome of the participant's selection(s).
    * `_sample_weight`: The weight of a trial, which is derived from participant performance on catch trials.
