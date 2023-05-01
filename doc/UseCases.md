**Components:**

+ handle_scale_and_nan.py
	+ Function: scale of numeric features and handle of missing values in the input
	+ Input: raw data csv file with Patient’s ID number and biomakers
	+ Output: single formatted DataFrame with metadata

+ split_cats_by_tolerance.py
 	+ Function: split the input into two DataFrame,"df_dev" and "df_test", such that the percentage of each category is within a specified tolerance	+ Input: single formatted DataFrame with metadata
	+ Output: two formatted DataFrame called "data_dev" and "data_test"

+ x.py
	+ Function: Create anonymized label for data, separate metadata from data, append both to the respective tables in the database
	+ Input: Formatted DF with metadata
	+ Output: Labeled data to extended anonymous database, associated metadata to extended patient database

+ find_biomarkers.py
	+ Function: Assesses which proteins are good predictors of Alzeihmer’s or Parkinson’s 
	+ Input: Proteomics dataset 
	+ Output: Pandas dataset of potential biomarkers and their p-values

+ cluster_protein.py
	+ Function: Sort out the biomarkers
	+ Input: csv of protein list
	+ Output: Visualize Plots

+ compare_patient_groups.py
	+ Function: It is an interactive visualization that takes the results from the machine learning algorithm and displays potential biomarkers
	+ Input: ML algorithm results
	+ Output: Volcano plot of various proteins to assess potential biomarkers for Alzeihmer’s or Parkinson’s as a PNG

+ interaction_biomarkers.py
	+ Function: Sort out the properties of biomarkers
	+ Input: Pandas dataset of potential biomarkers and their p-values
	+ Output: Cluster Pandas dataset of identified biomarkers & visualization plots


