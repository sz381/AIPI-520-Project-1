# Deprecated scripts

Earlier versions of the pipeline, kept for reference. Notebooks 01-05 replace
them, and they are not part of the run order in the main README.

They were written for the layout before they were moved into this folder and do
not run from here: `mos_lib.py` resolves the project root as `scripts/`, and it
reads `data/interim/rdu_hourly_features.csv`, which is not in the repository.

`results/` holds what they produced at the time, including the scores of every
model variant on the two validation folds (`mos_scores.csv`) and on the test
window (`test_scores.csv`).
