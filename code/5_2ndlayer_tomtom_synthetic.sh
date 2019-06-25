#!/bin/bash


for TRIAL in {0..4}
do
    dirpath="../results/synthetic_$TRIAL/conv_filters"
 
    # 2nd layer filters of cnn-1
    tomtom -evalue -thresh 0.1 -o $dirpath/cnn_1_2ndlayer $dirpath/cnn_1_2nd_layer_clip_$TRIAL.meme ../data/JASPAR_CORE_2016_vertebrates.meme

    # 2nd and 3rd layer filters of cnn-1-3
    tomtom -evalue -thresh 0.1 -o $dirpath/cnn_1_3_2ndlayer $dirpath/cnn_1_3_2nd_layer_clip_$TRIAL.meme ../data/JASPAR_CORE_2016_vertebrates.meme
    tomtom -evalue -thresh 0.1 -o $dirpath/cnn_1_3_3rdlayer $dirpath/cnn_1_3_3rd_layer_clip_$TRIAL.meme ../data/JASPAR_CORE_2016_vertebrates.meme

done

