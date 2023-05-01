**Components:**

+ handle_scale_and_nan.py
	+ Function: scale of numeric features and handle of missing values in the input
	+ Input: raw data csv file with Patient’s ID number and biomakers
	+ Output: single formatted DataFrame with metadata

+ split_cats_by_tolerance.py
 	+ Function: split the input into development set and testing set, such that the percentage of each category is within a specified tolerance	
 	+ Input: single formatted DataFrame with metadata
	+ Output: formatted development DataFrame and testing DataFrame

+ over_under.py
	+ Function: perform oversampling and undersampling techniques to balance the distribution of a categorical variable in a dataset 
	+ Input: formatted development DataFrame
	+ Output: balanced dataset as a dataframe

+ make_confusion_mtrx.py
	+ Function: Assesses which proteins are good predictors of Alzeihmer’s or Parkinson’s 
	+ Input: Proteomics dataset 
	+ Output: Pands

+ select_features.py
	+ Function: Sort out the biomarkers
	+ Input: csv of protein list
	+ Output: Visualize Plots

+ roc_curves_one_vs_rest.py
	+ Function: It is an interactive visualization that takes the results from the machine learning algorithm and displays potential biomarkers
	+ Input: ML algorithm results
	+ Output: Volcano plot of various proteins to assess potential biomarkers for Alzeihmer’s or Parkinson’s as a PNG

+ x.py
	+ Function: Sort out the properties of biomarkers
	+ Input: Pandas dataset of potential biomarkers and their p-values
	+ Output: Cluster Pandas dataset of identified biomarkers & visualization plots


