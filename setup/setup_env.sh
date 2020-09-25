#!/bin/bash

ENV_NAME='amr'

conda create -n ${ENV_NAME} python=3.7
conda install pandas -n ${ENV_NAME} 
conda install scikit-learn -n ${ENV_NAME} 

# This is for the notebook
conda install jupyter -n ${ENV_NAME} 

# This is for plotting
conda install r -n ${ENV_NAME}

echo "Done with creation of python environment, you will still need to install R packages for plotting"

