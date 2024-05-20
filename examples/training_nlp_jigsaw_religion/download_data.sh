#!/bin/bash
mkdir -p data
mkdir -p data/jigsawreligion/
mkdir -p output
wget https://anonymous.4open.science/r/example_data_fairness-15BA/train.tsv -P data/jigsawreligion
wget https://anonymous.4open.science/r/example_data_fairness-15BA/test.tsv -P data/jigsawreligion
wget https://anonymous.4open.science/r/example_data_fairness-15BA/valid.tsv -P data/jigsawreligion