# BrainPower

## Table of Contents

  - [About the Dataset](#about-the-dataset)
    - [Overview](#overview)
    - [Artifacts](#artifacts)
      - [Scripts](#scripts)
      - [Files](#files)
  - [Examples](#examples)
    - [Run Tests](#run-tests)
    - [Import](#import)

# About the Dataset
This GitHub page includes all of the code related to the development of a machine learning model for identifying biomarkers for Alzheimer’s and Parkinson’s. The data comes from the MacCoss Lab and it is a proteomics dataset collected on cerebral spinal fluid (CSF) from patients with four condition groups: Alzheimer's (AD), Parkinson's (PD), Lewy Body Dementia (LBD) and PD w/ Mild Cognitive Impairment (MCI), and Healthy. 
Alzheimer’s (AD) and Parkinson’s (PD) are the most prevalent neurodegenerative diseases. Since there are limitations on the current biomarkers used, our goal is to identify new biomarkers that will be able to diagnose patients without having to physically see the brains post-mortem.

### Overview

This directory stores the scripts used to download, train, and test the dataset with machine learning models. 

![flow_chart](https://github.com/BrainPowerChemE/BrainPower/assets/121738843/db6cd4cd-9438-4ffc-b0ee-268ef90adc1b)

### Artifacts

### Scripts

#### Python Scripts
[`automation.py`](automation.py) trains and tests the dataset on a machine learning model and outputs two resulting figures: ROC Curve plot and confusion matrix.
[`test_find_false_positive_patients.py`](brainpower/find_false_positive_patients.py) tests find_false_positive_patients.py

[`test_make_roc_curves.py`](brainpower/make_roc_curves.py) outputs one-vs-rest ROC Curves

[`test_handle_scale_and_nan.py`](brainpower/handle_scale_and_nan.py) replaces NaNs with the lowest value in the dataframe and applies a standard scaler.

[`test_make_confusion_mtrx.py`](brainpower/make_confusion_mtrx.py) outputs a confusion matrix based on the machine learning model results

[`test_over_under.py`](brainpower/over_under.py) 


#### Test Scripts
[`test_find_false_positive_patients.py`](brainpower/tests/test_find_false_positive_patients.py) tests find_false_positive_patients.py

[`test_make_roc_curves.py`](brainpower/tests/test_make_roc_curves.py) tests make_roc_curves.py

[`test_handle_scale_and_nan.py`](brainpower/tests/test_handle_scale_and_nan.py) tests handle_scale_and_nan.py

[`test_make_confusion_mtrx.py`](brainpower/tests/test_make_confusion_mtrx.py) tests make_confusion_mtrx.py

[`test_over_under.py`](brainpower/tests/test_over_under.py) tests over_under.py

### Files

[`data_with_biomarkers.csv`](data/final_dataset/data_with_biomarkers.csv) 


## Examples

### Run Tests

1. To test the code, run:

```
/BrainPower/
python -m unittest
```
