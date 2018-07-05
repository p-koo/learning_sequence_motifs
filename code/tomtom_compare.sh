#!/bin/bash

# Synthetic TomTom analysis (run after plot_conv_filters_synthetic.py)
tomtom -evalue -thresh 0.1 -o ../results/synthetic/conv_filters/cnn_2 ../results/synthetic/conv_filters/cnn_2_clip.meme ../data/JASPAR_CORE_2016_vertebrates.meme
tomtom -evalue -thresh 0.1 -o ../results/synthetic/conv_filters/cnn_4 ../results/synthetic/conv_filters/cnn_4_clip.meme ../data/JASPAR_CORE_2016_vertebrates.meme
tomtom -evalue -thresh 0.1 -o ../results/synthetic/conv_filters/cnn_10 ../results/synthetic/conv_filters/cnn_10_clip.meme ../data/JASPAR_CORE_2016_vertebrates.meme
tomtom -evalue -thresh 0.1 -o ../results/synthetic/conv_filters/cnn_25 ../results/synthetic/conv_filters/cnn_25_clip.meme ../data/JASPAR_CORE_2016_vertebrates.meme
tomtom -evalue -thresh 0.1 -o ../results/synthetic/conv_filters/cnn_50 ../results/synthetic/conv_filters/cnn_50_clip.meme ../data/JASPAR_CORE_2016_vertebrates.meme
tomtom -evalue -thresh 0.1 -o ../results/synthetic/conv_filters/cnn_100_2 ../results/synthetic/conv_filters/cnn_100_clip.meme ../data/JASPAR_CORE_2016_vertebrates.meme
tomtom -evalue -thresh 0.1 -o ../results/synthetic/conv_filters/cnn9_25 ../results/synthetic/conv_filters/cnn9_25_clip.meme ../data/JASPAR_CORE_2016_vertebrates.meme
tomtom -evalue -thresh 0.1 -o ../results/synthetic/conv_filters/cnn9_4 ../results/synthetic/conv_filters/cnn9_4_clip.meme ../data/JASPAR_CORE_2016_vertebrates.meme
tomtom -evalue -thresh 0.1 -o ../results/synthetic/conv_filters/cnn3_2 ../results/synthetic/conv_filters/cnn3_2_clip.meme ../data/JASPAR_CORE_2016_vertebrates.meme
tomtom -evalue -thresh 0.1 -o ../results/synthetic/conv_filters/cnn3_50 ../results/synthetic/conv_filters/cnn3_50_clip.meme ../data/JASPAR_CORE_2016_vertebrates.meme
tomtom -evalue -thresh 0.1 -o ../results/synthetic/conv_filters/cnn_50_2 ../results/synthetic/conv_filters/cnn_50_2_clip.meme ../data/JASPAR_CORE_2016_vertebrates.meme

# In vivo TomTom analysis (run after plot_conv_filters_invivo.py)
tomtom -evalue -thresh 0.1 -o ../results/invivo/conv_filters/cnn_2 ../results/invivo/conv_filters/cnn_2_clip.meme ../data/JASPAR_CORE_2016_vertebrates.meme
tomtom -evalue -thresh 0.1 -o ../results/invivo/conv_filters/cnn_4 ../results/invivo/conv_filters/cnn_4_clip.meme ../data/JASPAR_CORE_2016_vertebrates.meme
tomtom -evalue -thresh 0.1 -o ../results/invivo/conv_filters/cnn_10 ../results/invivo/conv_filters/cnn_10_clip.meme ../data/JASPAR_CORE_2016_vertebrates.meme
tomtom -evalue -thresh 0.1 -o ../results/invivo/conv_filters/cnn_25 ../results/invivo/conv_filters/cnn_25_clip.meme ../data/JASPAR_CORE_2016_vertebrates.meme
tomtom -evalue -thresh 0.1 -o ../results/invivo/conv_filters/cnn_50 ../results/invivo/conv_filters/cnn_50_clip.meme ../data/JASPAR_CORE_2016_vertebrates.meme
tomtom -evalue -thresh 0.1 -o ../results/invivo/conv_filters/cnn_100 ../results/invivo/conv_filters/cnn_100_clip.meme ../data/JASPAR_CORE_2016_vertebrates.meme
tomtom -evalue -thresh 0.1 -o ../results/invivo/conv_filters/cnn_50_2 ../results/invivo/conv_filters/cnn_50_2_clip.meme ../data/JASPAR_CORE_2016_vertebrates.meme
tomtom -evalue -thresh 0.1 -o ../results/invivo/conv_filters/cnn9_4 ../results/invivo/conv_filters/cnn9_4_clip.meme ../data/JASPAR_CORE_2016_vertebrates.meme
tomtom -evalue -thresh 0.1 -o ../results/invivo/conv_filters/cnn9_25 ../results/invivo/conv_filters/cnn9_25_clip.meme ../data/JASPAR_CORE_2016_vertebrates.meme
