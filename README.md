# Learning Sequence Motifs

This is a repository that contains datasets, scripts, and results of "Representation Learning of Genomic Sequence Motifs with Convolutional Neural Networks" by Peter K. Koo and Sean R. Eddy.

The code here depends on Deepomics, a custom-written, high-level APIs written on top of Tensorflow to seamlessly build, train, test, and evaluate neural network models.  WARNING: Deepomics is a required sub-repository.  To properly clone this repository, please use: 

$ git clone --recursive https://github.com/p-koo/learning_sequence_motifs.git

#### Dependencies
* Tensorflow r1.0 or greater (preferably r1.4 or r1.5)
* Python dependencies: PIL, matplotlib, numpy, scipy, sklearn


## Overview of the code

To generate datasets:
* code/Generate_synthetic_datasets.ipynb
* code/Generate_invivo_datasets.ipynb

To train the models on the synthetic dataset and the in vivo dataset: 
* code/train_synthetic_data.py 
* code/train_invivo_data.py 

These scripts loop through all models described in the manuscript.  Each model can be found in /code/models/

To evaluate the performance of each model on the test set: 
* code/print_performance_table_synthetic.py 
* code/print_performance_table_invivo.py 

To visualize and save 1st convolutional layer filters and also save a .meme file for the Tomtom search comparison tool: 
* code/plot_conv_filters_synthetic.py
* code/plot_conv_filters_invivo.py

To perform the Tomtom search comparison tool :
* code/tomtom_compare.sh  

Requires Tomtom installation as well as command-line abilities from the current directory.

To visualize guided-backprop saliency maps:
* code/Saliency_comparison.ipynb

## Overview of data

* Due to size restrictions, the dataset is not included in the repository.  Each dataset can be easily created by running the python notebooks: Generate_synthetic_datasets.ipynb and Generate_invivo_datasets.ipynb
* JASPAR_CORE_2016_vertebrates.meme contains a database of PWMS which is used for the Tomtom comparison search
* pfm_vertebrates.txt also contrains JASPAR motifs. This is the file that is used as ground truth for the synthetic dataset.

## Overview of results

* All results for each CNN model and dataset are saved in a respective directory (synthetic or invivo). 
* Trained model parameters are saved in results/synthetic/model_params.  
* visualization for convolution filters and results from Tomtom are saved in results/synthetic/conv_filters
* A reported performance table is saved in results/synthetic/performance_summary.tsv (automatically outputted from print_performance_table_synthetic.py)


