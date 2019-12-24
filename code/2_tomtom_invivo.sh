#!/bin/bash

for TRIAL in {0..4}
do
    dirpath="../results/invivo_$TRIAL/conv_filters"
    
    # tomtom for most models
    for MODEL in cnn_1 cnn_2 cnn_4 cnn_10 cnn_25 cnn_50 cnn_100 cnn9_4 cnn9_25 cnn3_50 cnn3_2 cnn_50_2 cnn_2_1
    do 
        tomtom -evalue -thresh 0.1 -o $dirpath/$MODEL $dirpath/${MODEL}_clip.meme ../data/JASPAR_CORE_2016_vertebrates.meme
    done
 
    # for CNN-25 with varying numbers of filters
    for FILTERS in 60 90 120
    do
        tomtom -evalue -thresh 0.1 -o $dirpath/cnn_25_$FILTERS $dirpath/cnn_25_${FILTERS}_clip.meme ../data/JASPAR_CORE_2016_vertebrates.meme
    done
done

