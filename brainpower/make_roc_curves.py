import imblearn
from itertools import combinations, cycle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import seaborn as sns
import sklearn.ensemble
import sklearn.linear_model
from sklearn.metrics import RocCurveDisplay
import sklearn.neighbors
import sklearn.preprocessing
import statsmodels
import statistics
from numpy import absolute, mean, sqrt
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import ConfusionMatrixDisplay, auc, roc_curve
from sklearn.model_selection import LeaveOneOut, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelBinarizer
from statsmodels.sandbox.stats.multicomp import multipletests
from xgboost import XGBClassifier

import sys

from make_confusion_mtrx import make_confusion_mtrx
from select_features import select_features
from false_positives import find_false_positive_patients



def roc_curves_one_vs_rest(data_dev, data_test, metadata, ml_results, feature_list=None):
	
    """
    Function: generates and plots Receiver Operating Characteristic (ROC) curves for a one-vs-rest multiclass classification
    
    Input: formatted development DataFrame(data_dev) and testing DataFrame (data_test)
    
    Output: plot of ROC curves with calculated AUC (Area Under the Curve) as a PNG file
    """
    if feature_list is None:
        feature_list = selected_features
        
    classes_of_interest=['Healthy', 'PD_MCI_LBD', 'PD', 'AD_MCI']
    
    X_test = data_test[feature_list]
    y_test = data_test['group']

    X_train = dev[feature_list]
    y_train = dev['group'] 


    classifier = sklearn.ensemble.RandomForestClassifier()
    y_score = classifier.fit(X_train, y_train).predict_proba(X_test)

    label_binarizer = LabelBinarizer().fit(y_train)
    y_onehot_test = label_binarizer.transform(y_test)

    fig, ax = plt.subplots(figsize=(6, 6))
    
    # store the fpr, tpr, and roc_auc for all averaging strategies
    fpr, tpr, roc_auc = dict(), dict(), dict()
    
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_onehot_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    plt.plot(
        fpr["micro"],
        tpr["micro"],
        label=f"micro-average ROC curve (AUC = {roc_auc['micro']:.2f})",
        color="deeppink",
        linestyle=":",
        linewidth=4,
    )

    colors = cycle(["aqua", "darkorange", "cornflowerblue", 'red'])
    for class_id, color in zip(range(4), colors):
        RocCurveDisplay.from_predictions(
            y_onehot_test[:, class_id],
            y_score[:, class_id],
            name=f"ROC curve for {classes_of_interest[class_id]}",
            color=color,
            ax=ax,
        )

    plt.plot([0, 1], [0, 1], "k--", label="ROC curve for chance level (AUC = 0.5)")
    plt.axis("square")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"One-vs-Rest ROC curves, using {len(feature_list)} features")
    plt.legend()
    plt.savefig('roc_curves.png')
    plt.show()
    return find_false_positive_patients(metadata, ml_results)
    
def main(): 
    PATH_DEV_DATA = sys.argv[1]
    PATH_TEST_DATA = sys.argv[2]
    PATH_METADATA = sys.argv[3]
    data_dev = pd.read_csv(PATH_DEV_DATA)
    data_test = pd.read_csv(PATH_TEST_DATA)
    metadata = pd.read_csv(PATH_METADATA)
    feature_list = select_features(data_dev, 18)
    results = make_confusion_mtrx(data_dev, data_test, feature_list)
    false_pos_df=roc_curves_one_vs_rest(data_dev=data_dev, data_test=data_test, 
        metadata=metadata, ml_results=results, feature_list=feature_list)

if __name__ == '__main__':
	main()
