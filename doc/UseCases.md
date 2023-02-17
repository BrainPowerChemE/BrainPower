**Components:**

+ add_data.py
	+ Function: Upload new MSMS data to the database
	+ Input: Patient’s ID number, date of test, raw data csv file, additional notes
	+ Output: Formatted DF with metadata

+ clean_data.py
 	+ Function: Correct data type, remove white space, remove punctuation, format data in dataframe, append metadata
	+ Input: Raw data in csv format and metadata in dataframe format, GUI to edit
	+ Output: Formatted DF with metadata
+ check_data.py
	+ Function: Check formatted DF & metadata for any errors
	+ Input: Formatted DF with metadata
	+ Output: Formatted DF with metadata
+ append_trainingdata.py
	+ Function: Create anonymized label for data, separate metadata from data, append both to the respective tables in the database
	+ Input: Clean DF with metadata
	+ Output: Labeled data to extended anonymous database, associated metadata to extended patient database

+ find_biomarkers.py
	+ Function: Assesses which proteins are good predictors of Alzeihmer’s or Parkinson’s 
	+ Input: Proteomics dataset 
	+ Output: Pandas dataset of potential biomarkers and their p-values

+ compare_patient_groups.py
	+ Function: It is an interactive visualization that takes the results from the machine learning algorithm and displays potential biomarkers
	+ Input: ML algorithm results
	+ Output: Volcano plot of various proteins to assess potential biomarkers for Alzeihmer’s or Parkinson’s as a PNG

+ interaction_biomarkers.py
	+ Function: Sort out the properties of biomarkers
	+ Input: Pandas dataset of potential biomarkers and their p-values
	+ Output: Cluster Pandas dataset of identified biomarkers & visualization plots

+ dashboard_user_interface.py
	+ Function: Formats the model for user interaction
	+ Input: Create HTML components for interactive components from users
	+ Output: Readily interpretive user interface for inputting data and receiving predictions

+ patient_visualizer.py
	+ Function: Generate visualization of patient data / risk assessment against extended anonymous data
	+ Input: Visual model, risk model, extended data, patient data
	+ Output: Image file or plot

+ statrisk.py
	+ Function: Generate statistical risk assessment for patient based on patient data and selected model(s)
	+ Input: Risk model, patient data
	+ Output: Statistical risk assessment csv
+ report.py
	+ Function: Combine visualizations and statistical risk assessment into report for clinician
	+ Input: Stat risk csv, visualization image or plot
	+ Output: Report PDF
+ cluster_protein.py
	+ Function: Sort out the biomarkers
	+ Input: csv of protein list
	+ Output: Visualize Plots
