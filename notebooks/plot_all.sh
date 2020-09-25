#!/bin/bash

if [ ! -d "fig_data" ]; then
  echo "./fig_data does not exist: Please run the figures_and_tables.ipynb notebook first"
  exit 1
fi

echo "Making directory ./figures to store output"
mkdir figures

if [ -f "fig_data/figure_2_threshold_sensitivity.csv" ]; then
  echo "Plotting Figure 2"
  Rscript plot_figure_2.R 
else
  echo "Error: Data for Figure 2 not available (this is expected if you have not run the end-to-end analysis), skipping this figure"
fi

if [ -f "fig_data/figure_3_fpr_fnr.csv" ]; then
  echo "Plotting Figure 3"
  Rscript plot_figure_3.R 
else
  echo "Error: Data for Figure 3 not available.  Did you run the notebook?"
  exit 1
fi

if [ -f "fig_data/figure_4_error_analysis.csv" ]; then
  echo "Plotting Figure 4"
  Rscript plot_figure_4.R 
else
  echo "Error: Data for Figure 4 not available.  Did you run the notebook?"
  exit 1
fi

if [ -f "fig_data/figure_5.csv" ]; then
  echo "Plotting Figure 5"
  Rscript plot_figure_5.R 
else
  echo "Error: Data for Figure 5 not available.  Did you run the notebook?"
  exit 1
fi

if [ -f "fig_data/figure_s2.csv" ]; then
  echo "Plotting Figure S-2"
  Rscript plot_figure_s2.R 
else
  echo "Error: Data for Figure S-2 not available.  Did you run the notebook?"
  exit 1
fi

if [ -f "fig_data/figure_s3.csv" ]; then
  echo "Plotting Figure S-3"
  Rscript plot_figure_s3.R 
else
  echo "Error: Data for Figure S-3 not available.  Did you run the notebook?"
  exit 1
fi
