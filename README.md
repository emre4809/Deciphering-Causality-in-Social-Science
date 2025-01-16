# Exploring Clustering of Causes and Effects in  Unstructured Social Science Text Using HDBSCAN and BERTopic

This branch represents the technical work done for my (Emre) Bachelor Thesis. It combines the preprocessing code that is in main with my work done on clustering causes and effects. Below are instructions on how the files should be executed.

### main_preprocessing and definitions_preprocessing
These two files are taken from main and are for the preprocessing section. definitions_preprocessing includes all of the functions done for preprocessing. main_preprocessing runs these functions on the provided data and as output, gets the folder "processed_csv_causal" which includes the causal sentences extracted from the data in csv format. Each csv file includes all the causal sentences extracted from their corresponding pdf in the data. This file should be run first to obtain this folder with causal sentences.

### cause-effect_extract
This file uses the "unicausal-tok-baseline" model by Fiona Anting Tan and others, taken from Huggingface. The csv files including causal sentences are fed into the model and the cause and effect in these causal sentences are outputted. The output is a folder named "extracted_causes_effects" which includes csv files again with causes and effects instead of causal sentences. More detail on this can be found in the thesis. This should be run after main_preprocessing.

### bertopic, hdbscan_pre_trained_miniLM, and hdbscan_word2vec
These three files represent the three methods used for clustering the causes and effects. THESE FILES HAVE BEEN DONE IN GOOGLE COLAB. Due to issues with my laptop, Google Colab has been used to run the clustering methods. The data obtained after cause-effect_extract should be uploaded to Google Drive and these files can then be run in Google Colab.

### tests
This folder contains all my previous test files for my project. These files are ideas that were discontinued and did not get used in the thesis.
