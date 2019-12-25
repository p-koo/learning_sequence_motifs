#!/bin/bash

python 1_train_synthetic.py
bash 2_tomtom_synthetic.sh
python 3_match_statistics_synthetic.py
python 4_2ndlayer_filter_analysis_synthetic.py
bash 5_2ndlayer_tomtom_synthetic.sh
python 6_2ndlayer_match_statistics_synthetic.py

