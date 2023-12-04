#!/bin/bash

# Evaluating previously trained models to predict seeing a psychiatrist within 12 months using toy data
#python -m models.bow --target "dspln_PSYCHIATRY_12" --table "demo" --eval_only --model-file "./results/final_results/dspln_PSYCHIATRY_12/BoW/BoW_20230220-1206_e0.pbz2" --data-dir "./demo" --results-dir "./demo"
python -m models.bow --target "demo" --eval_only --table "toy_data" --data-dir "./data" --results-dir "./data"