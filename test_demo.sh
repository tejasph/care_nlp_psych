#!/bin/bash

# Evaluating previously trained models to predict seeing a psychiatrist within 12 months using toy data
#python -m models.bow --target "dspln_PSYCHIATRY_12" --table "demo" --eval_only --model-file "./results/final_results/dspln_PSYCHIATRY_12/BoW/BoW_20230220-1206_e0.pbz2" --data-dir "./demo" --results-dir "./demo"

# Testing toy_data with BoW Model
#python -m models.bow --target "demo" --table "toy_data" --data-dir "./data" --results-dir "results"

# Testing CNN model with toy data
python -m models.cnn  --target "demo" --eval_only --model-file "/Users/tejasphaterpekar/Documents/Data-Projects/cancer_nlp/care_nlp_psych/results/demo/CNN/CNN_20231214-1656.pt" --data-dir "./data" --results-dir "results"