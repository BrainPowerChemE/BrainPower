**Components:**

+ handle_scale_and_nan.py
	+ Function: scale of numeric features and handle of missing values in the input
	+ Input: raw research_data.csv file with Patient’s ID number and biomakers
	+ Output: single formatted DataFrame (scaled_data_full.df)

+ over_under.py
	+ Function: perform oversampling and undersampling techniques to balance the distribution of a categorical variable in a dataset 
	+ Input: formatted development DataFrame (data_dev.df)
	+ Output: balanced dvelopment Dataframe (data_dev.df)

+ select_features.py
	+ Function: selecte the top N features using the mRMR (minimum redundancy maximum relevance) algorithm
	+ Input: balanced dvelopment Dataframe (data_dev.df) 
	+ Output: a JSON file with selected N features (selected_features.json)

+ apply_ml_model.py
	+ Function: evaluate the performance of various machine learning classifiers with cross-validation
	+ Input: balanced dvelopment Dataframe (data_dev.df), testing DataFrame (data_test.df) and selected features list (selected_features.json)
	+ Output: Dataframe (stats.df) store the evaluation results including the number of folds, scores, average score, standard deviation, model used and method

+ make_confusion_mtrx.py
	+ Function: generate a confusion matrix for a random forest classifier
	+ Input: formatted balanced development DataFrame(data_dev.df) and testing DataFrame (data_test.df)
	+ Output: confusion matrix visualization with balanced accuracy score and dataframe (ml_results.df) with predicted condition vs. actual condition

+ find_false_positive_patients.py 
	+ Function: identifies the "healthy" patients that were identified as diseased by the ML model
	+ Input: dataframe (ml_results.df) with predicted condition vs. actual condition and Metadata.csv contains patient information 
	+ Output: DataFrame (false_positives_info.df) of the "healthy" patients that were identified as diseased by the ML model
    
+ roc_curves_one_vs_rest.py
	+ Function: generates and plots Receiver Operating Characteristic (ROC) curves for a one-vs-rest multiclass classification
	+ Input: formatted development DataFrame(data_dev) and testing DataFrame (data_test)
	+ Output: plot of ROC curves with calculated AUC (Area Under the Curve) as a PNG file
