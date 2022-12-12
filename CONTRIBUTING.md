# Contributing

When contributing to this repository, please first discuss the change you wish to make via issue or email with the owners of this repository before making a change.

Please note we have a code of conduct, please follow it in all your interactions with the project.

## Guiding Principles
The dataset generators should be written such that minimal code changes are necessary, but users are afforded maximum flexibility to achieve their modeling goals.

## Common Tasks
* Build and overwrite: `tfds build <dataset_dir_name> --overwrite`
* Build, overwrite, and register checksums: `tfds build <dataset_dir_name> --overwrite --register_checksums`
