# Data Processing Contribution

This folder contains the data preprocessing work for COS30019 Assignment 2 Part B.

## Included files
- `A2B_Data_Preprocessing.ipynb`: documented notebook showing data loading, quality checks, time-series transformation, duplicate resolution, chronological split, scaling and sequence preparation.
- `data_preprocessing.py`: reusable preprocessing script for the model-development team.
- `data_quality_and_processing_summary.csv`: summary of data validation results including post-reshape duplicate detection and resolution.
- `chronological_split_summary.csv`: summary of the chronological training, validation and test split.

## Processing completed
- Loaded the supplied SCATS October 2006 dataset.
- Checked missing, invalid, negative and duplicate traffic records.
- Converted daily `V00` to `V95` volume fields into timestamp-based 15-minute observations.
- **Detected and resolved 2,976 duplicate site-location-timestamp records** (caused by source rows sharing the same SCATS Number, Location and Date without being exact row duplicates). Resolved by averaging the affected traffic-flow values.
- Created chronological train, validation and test splits.
- Applied min-max normalisation using training data only.
- Prepared configurable model input sequences.

## Known behaviours (not bugs)
- **Scaled values above 1.0:** Validation and test splits may produce scaled values above 1.0 because their traffic-flow peaks can exceed the training-set maximum. This is expected and is not a data-leakage issue.
- **Cross-midnight windows:** The sliding window in `create_sequences` does not restart at midnight. It operates continuously over each (split, site, location) group, so the first window of a new calendar day includes observations from the previous day. This is intentional for continuous time-series modelling.

## Note on generated files
Large processed CSV and NumPy output files were not uploaded to GitHub because they are generated outputs and can be reproduced by running `data_preprocessing.py`. The current demonstration sequence window is 12 intervals (3 hours); the group may adjust this to match final model settings.
