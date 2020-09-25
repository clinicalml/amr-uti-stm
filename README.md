# Code for replication of "A decision algorithm to promote outpatient antimicrobial stewardship for uncomplicated urinary tract infection"

This code is meant to be used in conjunction with the [AMR-UTI dataset](http://www.clinicalml.org/data/amr-dataset) for replication of the main analyses in the paper. 

## Preface: Important Note on Replication

The uncomplicated specimens included in this dataset release (in both the train and test sets) are identical to those used for our analyses.

Nonetheless, there are minor differences that will arise when replicating our analyses with the released dataset.  There are two broad reasons for this:
1. Slight differences in features, primarily due to de-identification efforts.
2. The absence of patient-level identifiers, which were used in our original analysis to construct all train/validate splits.

Regarding (1), the differences are as follows:
* Any binary feature with fewer than 20 positive observations was dropped from the dataset.
* All colonization pressure features were rounded to the nearest 0.01
* Age was censored, so that any patient with an age > 89 has their age set to 90.
* Basic laboratory values (WBC, lymphocytes, and neutrophil counts) are excluded from the dataset release, due to inconsistencies in reporting of laboratory values. These features did not have a noticeable impact on our results.

Regarding (2), our analysis used patient identifiers to ensure that there were no patients with specimens in both the train and validate sets.  Because there are no patient identifiers included in this dataset release, the splits performed by our replication code are done without knowledge of patient identifiers.  As a result, they are necessarily different from the ones we used in our work.

For this reason, we provide utilities to run our scripts both in an end-to-end fashion (replicating the approach taken in the paper, but with different train/validate splits and therefore different selection of hyperparameters), as well as directly using the hyperparameters and thresholds chosen by our original analysis.

# Replication Instructions

## Setup 

First, you need to load in the data from Physionet.  See the [project website](http://www.clinicalml.org/data/amr-dataset) for more information on how to access this data. Place the Physionet data in a folder **outside** this repository to avoid accidentally uploading any files to Github.

Second, you need to **edit the paths in `setup/paths.sh`** 
* `DATA_PATH` in `setup/paths.sh` should reflect the absolute path to the files from Physionet. The files should be all be accessible at `${DATA_PATH}/<file_name>.csv`.
* `REPO_PATH` in `setup/paths.sh` should reflect the absolute path to this directory.  That is, this file should be located at `${REPO_PATH}/README.md`.

Third, you need to set up your `python`, `R`, and bash environment.
* Run `bash setup/setup_env.sh` to create a python environment `amr` that will contain the necessary packages. 
* Run `conda activate amr` to activate the environment.
* Run `Rscript setup/install_r_packages.R` to install relevant `R` packages into the environment (for plotting purposes), which can take some time. 
* Run `source setup/paths.sh` (note: use `source`, not `bash` here) to populate the necessary bash variables that define paths.
* *Going forward (e.g., in subsequent terminal sessions), you will need to run `conda activate amr` and `source setup/paths.sh` before running the experiment scripts*

Finally, you need to run `python setup/load_data.py` to split the data release into train/test splits that the remaining code expects.  This will create additional `.csv` files in `${DATA_PATH}`

## Running the experiment scripts

We present two options for replication:
1. Using our original hyperparameters and thresholds: Run `bash run_all_rep.sh`
2. Running the analysis end-to-end: Run `bash run_all.sh`

Note that Option 1 is much faster, as it skips all hyperparameter tuning, while Option 2 will take approximatly 2-3 hours to run, driven primarily by threshold selection (see details below).

After either of these scripts have been run, see the "Replicating Figures and Tables" section below for instructions on running the analysis notebook + plotting code.  If you have run with Option 1, then you will want to ensure that the `USE_REP_HP` flag in the notebook is set to `True`, and vice versa if you are using Option 2.

## (Optional) Manually Running the Scripts

Alternatively, you can manually run the relevant experiment scripts that are called by `run_all.sh` and `run_all_rep.sh` respectively.  These details are given below.

Before manually running the experiments in this section, run `cd ${REPO_PATH}/experiments` to go to the right directory.  All the below assumes you are in that directory, and there will be errors if you try to run from a different directory.

When you run these scripts for the first time, they will create (if it does not already exist) a folder `${REPO_PATH}/experiments/experiment_results/<exp_type>/<exp_name>/` that contains `results` to store the artifacts, and `logs` where you can watch progress by examining the log files.  In this context, `<exp_type>` might be `train_outcome_models` and `<exp_name>` might be `train_outcome_models_validation`.  

**NOTE**: The scripts (and paths in `setup/paths.sh`) assume the experiment names that are already given in these scripts, so **do not change them**

### Option 1: Running the analysis with the original hyperparameters

This will skip the validation scripts, because those are used to choose hyperparameters, which in this analysis is not necessary.

To run these analyses, you can either of the following equivalent options:
* Run `bash run_all_rep.sh` from the base directory `${REPO_PATH}`
* Move into the `${REPO_PATH}/experiments` directory and run the following script
```
bash scripts/eval_test_scripts/train_outcome_models_eval_test_replication.sh
```
This script will train models with the original hyperparameters on the entire training set, and then evaluate on the test set.  The original thresholds are then manually applied in the notebook that replicates tables and figures (see next section).

### Option 2: Running our analysis end-to-end

This will re-run the entire analysis, including the automatic selection of thresholds and hyperparameters using train/validate splits.  As noted above, this will result in slightly different choices than in our published analysis, in part because the splits will differ.

To run these analyses, you can either of the following equivalent options:
* Run `bash run_all.sh` from the base directory `${REPO_PATH}`
* Move into the `${REPO_PATH}/experiments` directory and run the following scripts (in order)

First, run the following to train outcome models using the train/validation set, using the hyperparameter grid defined in `../models/hyperparameter_grids.py`. This also generates train/validation set predictions using the chosen hyperparameters.  **NOTE**: In our experiments we observed that logistic regression and random forests performed comparably, and by default the script will only investigate hyperparameters for logistic regression models.  To change this, you will need to change the code in `experiment_train_outcome_model.py:117, 127` where we have commented out random forests.
```
bash scripts/validation_scripts/train_outcome_models_validation.sh
``` 

Second, run the following to choose thresholds based on the train/validation set.  Note that this script will take a significant amount of time (about 2 hours) to run, becuase it investigates a large set of thresholding combinations.
```
bash scripts/validation_scripts/thresholding_validation.sh
```

Third, run the following to re-train outcome models using the chosen hyperparameters, and then evaluate these models on the test set.
```
bash scripts/eval_test_scripts/train_outcome_models_eval_test.sh
```

Finally, run the following script to perform the final thresholding experiment on the test set.
```
bash scripts/eval_test_scripts/thresholding_test.sh
```

## Replicating tables and figures

Within this repository, `notebooks/` contains a jupyter notebook that can be used to replicate figures from the paper and examine results.

To replicate figures, you will need to do the following:
* First, navigate to the folder `notebooks/` and open the Jupyter notebook `figures_and_tables.ipynb`
* Set the flag `USE_REP_HP` in this notebook based on whether or not you wish to compute results using the original hyperparameters, or the results of the end-to-end analysis applied to the dataset release.  Note that either of these options assumes you have already run the relevant code above.
* In either case, run the entire notebook end-to-end: The notebook generates the data used for plotting, as well as main tables in the paper.
* To generate figures (AFTER running the notebook code), run the script `plot_all.sh` which will sequentially call the various `R` scripts to generate the plots.
