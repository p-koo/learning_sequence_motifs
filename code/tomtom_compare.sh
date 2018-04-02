#!/bin/bash


tomtom -evalue -thresh 0.1 -o ../results/synthetic/conv_filters/cnn_2 ../results/synthetic/conv_filters/cnn_2.meme ../data/JASPAR_CORE_2016_vertebrates.meme
tomtom -evalue -thresh 0.1 -o ../results/synthetic/conv_filters/cnn_4 ../results/synthetic/conv_filters/cnn_4.meme ../data/JASPAR_CORE_2016_vertebrates.meme
tomtom -evalue -thresh 0.1 -o ../results/synthetic/conv_filters/cnn_10 ../results/synthetic/conv_filters/cnn_100.meme ../data/JASPAR_CORE_2016_vertebrates.meme
tomtom -evalue -thresh 0.1 -o ../results/synthetic/conv_filters/cnn_25 ../results/synthetic/conv_filters/cnn_25.meme ../data/JASPAR_CORE_2016_vertebrates.meme
tomtom -evalue -thresh 0.1 -o ../results/synthetic/conv_filters/cnn_50 ../results/synthetic/conv_filters/cnn_50.meme ../data/JASPAR_CORE_2016_vertebrates.meme
tomtom -evalue -thresh 0.1 -o ../results/synthetic/conv_filters/cnn_100 ../results/synthetic/conv_filters/cnn_100.meme ../data/JASPAR_CORE_2016_vertebrates.meme

tomtom -evalue -thresh 0.1 -o ../results/synthetic/conv_filters/cnn9_25 ../results/synthetic/conv_filters/cnn9_25.meme ../data/JASPAR_CORE_2016_vertebrates.meme
tomtom -evalue -thresh 0.1 -o ../results/synthetic/conv_filters/cnn9_4 ../results/synthetic/conv_filters/cnn9_4.meme ../data/JASPAR_CORE_2016_vertebrates.meme

tomtom -evalue -thresh 0.1 -o ../results/synthetic/conv_filters/cnn_50_2 ../results/synthetic/conv_filters/cnn_50_2.meme ../data/JASPAR_CORE_2016_vertebrates.meme


tomtom -evalue -thresh 0.1 -o ../results/invivo/conv_filters/cnn_2 ../results/invivo/conv_filters/cnn_2.meme ../data/JASPAR_CORE_2016_vertebrates.meme
tomtom -evalue -thresh 0.1 -o ../results/invivo/conv_filters/cnn_4 ../results/invivo/conv_filters/cnn_4.meme ../data/JASPAR_CORE_2016_vertebrates.meme
tomtom -evalue -thresh 0.1 -o ../results/invivo/conv_filters/cnn_10 ../results/invivo/conv_filters/cnn_10.meme ../data/JASPAR_CORE_2016_vertebrates.meme
tomtom -evalue -thresh 0.1 -o ../results/invivo/conv_filters/cnn_25 ../results/invivo/conv_filters/cnn_25.meme ../data/JASPAR_CORE_2016_vertebrates.meme
tomtom -evalue -thresh 0.1 -o ../results/invivo/conv_filters/cnn_50 ../results/invivo/conv_filters/cnn_50.meme ../data/JASPAR_CORE_2016_vertebrates.meme
tomtom -evalue -thresh 0.1 -o ../results/invivo/conv_filters/cnn_100 ../results/invivo/conv_filters/cnn_100.meme ../data/JASPAR_CORE_2016_vertebrates.meme
