#!/bin/bash


tomtom -evalue -thresh 0.1 -o ../results/multitask/conv_filters/cnn_2_50 ../results/multitask/conv_filters/cnn_2_50.meme ../data/JASPAR_CORE_2016_vertebrates.meme
tomtom -evalue -thresh 0.1 -o ../results/multitask/conv_filters/cnn_4_25 ../results/multitask/conv_filters/cnn_4_25.meme ../data/JASPAR_CORE_2016_vertebrates.meme
tomtom -evalue -thresh 0.1 -o ../results/multitask/conv_filters/cnn_10_10 ../results/multitask/conv_filters/cnn_10_10.meme ../data/JASPAR_CORE_2016_vertebrates.meme
tomtom -evalue -thresh 0.1 -o ../results/multitask/conv_filters/cnn_25_4 ../results/multitask/conv_filters/cnn_25_4.meme ../data/JASPAR_CORE_2016_vertebrates.meme
tomtom -evalue -thresh 0.1 -o ../results/multitask/conv_filters/cnn_50_2 ../results/multitask/conv_filters/cnn_50_2.meme ../data/JASPAR_CORE_2016_vertebrates.meme
tomtom -evalue -thresh 0.1 -o ../results/multitask/conv_filters/cnn_100_1 ../results/multitask/conv_filters/cnn_100_1.meme ../data/JASPAR_CORE_2016_vertebrates.meme

tomtom -evalue -thresh 0.1 -o ../results/multitask/conv_filters/scnn_25_4 ../results/multitask/conv_filters/scnn_25_4.meme ../data/JASPAR_CORE_2016_vertebrates.meme
tomtom -evalue -thresh 0.1 -o ../results/multitask/conv_filters/scnn_4_25 ../results/multitask/conv_filters/scnn_4_25.meme ../data/JASPAR_CORE_2016_vertebrates.meme

tomtom -evalue -thresh 0.1 -o ../results/multitask/conv_filters/cnn_2_50_invariant ../results/multitask/conv_filters/cnn_2_50_invariant.meme ../data/JASPAR_CORE_2016_vertebrates.meme


tomtom -evalue -thresh 0.1 -o ../results/deepsea/conv_filters/cnn_2_50 ../results/deepsea/conv_filters/cnn_2_50.meme ../data/JASPAR_CORE_2016_vertebrates.meme
tomtom -evalue -thresh 0.1 -o ../results/deepsea/conv_filters/cnn_4_25 ../results/deepsea/conv_filters/cnn_4_25.meme ../data/JASPAR_CORE_2016_vertebrates.meme
tomtom -evalue -thresh 0.1 -o ../results/deepsea/conv_filters/cnn_10_10 ../results/deepsea/conv_filters/cnn_10_10.meme ../data/JASPAR_CORE_2016_vertebrates.meme
tomtom -evalue -thresh 0.1 -o ../results/deepsea/conv_filters/cnn_25_4 ../results/deepsea/conv_filters/cnn_25_4.meme ../data/JASPAR_CORE_2016_vertebrates.meme
tomtom -evalue -thresh 0.1 -o ../results/deepsea/conv_filters/cnn_50_2 ../results/deepsea/conv_filters/cnn_50_2.meme ../data/JASPAR_CORE_2016_vertebrates.meme
tomtom -evalue -thresh 0.1 -o ../results/deepsea/conv_filters/cnn_100_1 ../results/deepsea/conv_filters/cnn_100_1.meme ../data/JASPAR_CORE_2016_vertebrates.meme

tomtom -evalue -thresh 0.1 -o ../results/deepsea/conv_filters/cnn_2_50_invariant ../results/deepsea/conv_filters/cnn_4_25_invariant.meme ../data/JASPAR_CORE_2016_vertebrates.meme

tomtom -evalue -thresh 0.1 -o ../results/deepsea/conv_filters/scnn_25_4 ../results/deepsea/conv_filters/scnn_25_4.meme ../data/JASPAR_CORE_2016_vertebrates.meme
tomtom -evalue -thresh 0.1 -o ../results/deepsea/conv_filters/scnn_4_25 ../results/deepsea/conv_filters/scnn_4_25.meme ../data/JASPAR_CORE_2016_vertebrates.meme

