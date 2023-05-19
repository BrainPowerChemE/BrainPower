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

![flow_chart (1)](https://github.com/BrainPowerChemE/BrainPower/assets/121738843/d4390625-87b5-4112-8b6c-5ccc9e0a6df9)

### Artifacts

### Scripts

#### Python Scripts
[`automation.py`](automation.py) trains and tests the dataset on a machine learning model and outputs two resulting figures: ROC Curve plot and confusion matrix.


#### Test Scripts
[`test_automation.py`](test_automation.py) tests automation.py


### Files

[`data_with_biomarkers.csv`](test-data/data_with_biomarkers.csv) 

[`dev_with_biomarkers.csv`](test-data/dev_with_biomarkers.csv) 

[`test_with_biomarkers.csv`](test-data/test_with_biomarkers.csv) 


## Examples

### Run Tests

1. To test [`automation.py`](automation.py), run:

```
/BrainPower/
python -m unittest
```

### Import

2. Run the script to output  ROC curve and confusion matrix figures.
```
python automation.py
```
