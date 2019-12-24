#!/bin/bash


for TRIAL in {0..4}
do
    dirpath="../results/synthetic_$TRIAL/conv_filters"
    
    # tomtom for most models
    for MODEL in cnn_1 cnn_2 cnn_4 cnn_10 cnn_25 cnn_50 cnn_100 cnn9_4 cnn9_25 cnn3_50 cnn3_2 cnn_50_2 cnn_2_1
    do 
        tomtom -evalue -thresh 0.1 -o $dirpath/$MODEL $dirpath/${MODEL}_clip.meme ../data/JASPAR_CORE_2016_vertebrates.meme
    done
 
    # 2nd layer filters of cnn-1
    tomtom -evalue -thresh 0.1 -o $dirpath/cnn_1_2ndlayer $dirpath/cnn_1_2nd_layer_clip_$TRIAL.meme ../data/JASPAR_CORE_2016_vertebrates.meme

    # 2nd and 3rd layer filters of cnn-1-3
    tomtom -evalue -thresh 0.1 -o $dirpath/cnn_1_3_2ndlayer $dirpath/cnn_1_3_2nd_layer_clip_$TRIAL.meme ../data/JASPAR_CORE_2016_vertebrates.meme
    tomtom -evalue -thresh 0.1 -o $dirpath/cnn_1_3_3rdlayer $dirpath/cnn_1_3_3rd_layer_clip_$TRIAL.meme ../data/JASPAR_CORE_2016_vertebrates.meme

    # for CNN-25 with varying numbers of filters
    for FILTERS in 60 90 120
    do
        tomtom -evalue -thresh 0.1 -o $dirpath/cnn_25_$FILTERS $dirpath/cnn_50_${FILTERS}_clip.meme ../data/JASPAR_CORE_2016_vertebrates.meme
    done
done

