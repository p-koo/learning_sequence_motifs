# Learning Sequence Motifs

This is a repository that contains datasets and scripts to reproduce the results of "Representation Learning of Genomic Sequence Motifs with Convolutional Neural Networks" by Peter K. Koo and Sean R. Eddy, which canbe found: https://www.biorxiv.org/content/10.1101/362756v2

The code here depends on Deepomics, a custom-written, high-level APIs written on top of Tensorflow to seamlessly build, train, test, and evaluate neural network models.  WARNING: Deepomics is a required sub-repository.  To properly clone this repository, please use: 

$ git clone --recursive \url{https://github.com/p-koo/learning_sequence_motifs.git}

#### Dependencies
* Tensorflow r1.0 or greater (preferably r1.14 or r1.15)
* Python dependencies: PIL, matplotlib, numpy, scipy (version 1.1.0), sklearn
* meme suite (5.1.0)

## Overview of the code

To generate datasets:
* code/0_Generate_synthetic_datasets.ipynb
* code/0_Generate_invivo_datasets.ipynb

To train the models on the synthetic dataset and the in vivo dataset: 
* code/1_train_synthetic.py 
* code/1_train_invivo.py 

This script trains each model, evaluates the generalization performance of each model on the test set, and plots visualizations of 1st convolutional layer filters and saves a .meme file for the Tomtom search comparison tool. Each model can be found in /code/models/


To perform the Tomtom search comparison tool:
* code/2_tomtom_synthetic.sh  
* code/2_tomtom_invivo.sh  

Requires Tomtom installation as well as command-line abilities from the current directory.


To calculate statistics across different initialization trials for each model, this script aggregates the matches to ground truth motifs:
* code/3_match_statistics_synthetic.sh  
* code/3_match_statistics_invivo.sh  


To plot 2nd layer filters for CNN-1 and 2nd and 3rd layer filters for CNN-1-3:
* code/4_2ndlayer_filter_analysis_synthetic.sh  
* code/4_2ndlayer_filter_analysis_invivo.sh  


To perform the Tomtom search comparison tool on the deeper layer filters:
* code/5_2ndlayer_tomtom_synthetic.sh  
* code/5_2ndlayer_tomtom_invivo.sh  


To calculate statistics across different initialization trials for each model, this script aggregates the matches to ground truth motifs for 2nd layer filters for CNN-1 and CNN-1-3:
* code/6_2ndlayer_match_statistics_synthetic.sh  
* code/6_2ndlayer_match_statistics_invivo.sh  


## Overview of data

* Due to size restrictions, the dataset is not included in the repository.  Each dataset can be easily created by running the python notebooks: Generate_synthetic_datasets.ipynb and Generate_invivo_datasets.ipynb
* JASPAR_CORE_2016_vertebrates.meme contains a database of PWMs which is used for the Tomtom comparison search
* pfm_vertebrates.txt also contrains JASPAR motifs. This is the file that is used as ground truth for the synthetic dataset.

## Overview of results

* All results for each CNN model and dataset are saved in a respective directory (synthetic or invivo). 
* Trained model parameters are saved in results/synthetic/model_params.  
* visualization for convolution filters and results from Tomtom are saved in results/synthetic/conv_filters
* A reported performance table is saved in results/synthetic/performance_summary.tsv (automatically outputted from print_performance_table_synthetic.py)


