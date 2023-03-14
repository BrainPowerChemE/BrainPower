import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import scipy
import statistics

import sklearn.linear_model
import sklearn.metrics
import sklearn.naive_bayes
import sklearn.preprocessing
import xgboost

import imblearn
import mrmr
import ast
import altair as alt


def handle_scale_and_nan(frame, nandecision="drop", scale="MinMax"):
    features = list(frame.select_dtypes(include="float64"))
    cat = list(frame.select_dtypes(include="object"))

    if scale == "MinMax":
        scaler = sklearn.preprocessing.MinMaxScaler().fit(frame[features])
    elif scale == "Standard":
        scaler = sklearn.preprocessing.StandardScaler().fit(frame[features])

    df_cont = pd.DataFrame(data=scaler.transform(frame[features]), columns=features)
    df_cat = pd.DataFrame(data=frame[cat], columns=cat)

    frame = pd.concat([df_cat, df_cont], axis=1)

    if nandecision == "mean":
        for feature in features:
            frame[feature].fillna((frame[feature].mean()), inplace=True)
    elif nandecision == "drop":
        frame = frame.dropna(axis=1)

    return frame


def split_cats_by_tolerance(
    frame,
    tolerance,
    silent=False,
    randomstate=None,
    split=0.15,
    step=1,
    target="group",
    categories=["Healthy", "AD_MCI", "PD", "PD_MCI_LBD"],
):
    tolerable_list = []
    if randomstate == None:
        randomstate = np.random.randint(0, 2**31)
    elif type(randomstate) == int:
        pass
    while sum(tolerable_list) != 4:
        df_dev, df_test = sklearn.model_selection.train_test_split(
            frame, test_size=split, random_state=randomstate
        )

        dev_dict = dict(df_dev[target].value_counts())
        test_dict = dict(df_test[target].value_counts())

        tolerable_list = []
        stats_dict = {}
        for i in range(0, len(categories)):
            try:
                percents = [
                    (dev_dict[categories[i]] / len(df_dev)),
                    (test_dict[categories[i]] / len(df_test)),
                ]
            except:
                break
            standdev = np.std(percents)
            if standdev <= tolerance:
                tolerable_list.append(1)
                stats_dict[str(categories[i])] = [[*percents], standdev]
            else:
                tolerable_list.append(0)

        randomstate += step

    if sum(tolerable_list) == 4:
        if silent == False:
            print(dev_dict)
            print(test_dict)
            print("Randstate:", randomstate - 1)
            for i in range(0, len(categories)):
                print(
                    "\nPercent",
                    categories[i],
                    "in dev, test:",
                    stats_dict[categories[i]][0],
                    "\nStandard deviation of these values:",
                    stats_dict[categories[i]][1],
                    "\n",
                )
        elif silent == True:
            pass

    return df_dev, df_test


def over_under(
    df_train,
    cat_in_excess="Healthy",
    target="group",
    randomstate=None,
    silent=False,
    replacement=False,
):
    """
    Takes dataframe(s) with only the target value and float64 features
    This function is to balance the samples in an imbalanced training dataset that has one category in excess, with additional categories more near each other
    The categories below the category in excess will be oversampled to equality, then the category in excess will be undersampled to equality
    ---Parameters---
    df_train: the training dataframe
    df_val: the validation dataframe
    cat_in_excess: the category which is present in excess, far above the other categories
    target: target column in the dataframe
    randomstate: if chosen, this will the random state for the sampling. Default: None, numpy random integer method between 0 and 4294967295, the range of the sampling module used
    randomstate_sampler: the number of loops to run to compare random states starting from
    replacement: (defalut=False) whether or not to perform the undersampling step with replacement
    """

    if randomstate == None:
        randomstate = np.random.randint(0, 4294967295)
    elif type(randomstate) == int:
        pass
    else:
        raise TypeError("Select None or integer value for randomstate")

    # Drop the excessive category and oversample minority to the intermediate category
    df_train_no_excess = df_train[df_train[target] != cat_in_excess]
    over_sampler = imblearn.over_sampling.RandomOverSampler(random_state=randomstate)
    X_train = df_train_no_excess.drop(columns=target)
    y_train = df_train_no_excess[target]
    X_train_over, y_train_over = over_sampler.fit_resample(X_train, y_train)
    df_train_over = pd.concat([y_train_over, X_train_over], axis=1)

    # Re-introduce the excessive category and undersample the majority to the minority
    df_train_excess = pd.concat(
        [df_train_over, df_train[df_train[target] == cat_in_excess]]
    )
    under_sampler = imblearn.under_sampling.RandomUnderSampler(
        random_state=randomstate, replacement=replacement
    )
    X_train = df_train_excess.drop(columns=target)
    y_train = df_train_excess[target]
    X_train_under, y_train_under = under_sampler.fit_resample(X_train, y_train)
    df_train_eq = pd.concat([y_train_under, X_train_under], axis=1)

    if silent == False:
        print(randomstate)
    elif silent == True:
        pass

    return df_train_eq


def mrmr_shorthand(X_train, y_train, X_val, y_val, K, model, score):
    feature_performance = []

    for k in K:
        reduced_features = mrmr.mrmr_classif(X_train, y_train, K=k)
        X_train_reduced = X_train[reduced_features]
        X_val_reduced = X_val[reduced_features]
        model.fit(X_train_reduced, y_train)
        feature_performance.append(
            [k, score(y_val, model.predict(X_val_reduced)), reduced_features]
        )
    return feature_performance


def mrmr_feature_selection(
    data_dev,
    split,
    min_features,
    max_features,
    step_features,
    folds,
    ci=0.68,
    balancer="over_under without replacement",
    tolerance=0.01,
    target="group",
    model=sklearn.linear_model.RidgeClassifier(),
    score=sklearn.metrics.balanced_accuracy_score,
    featureselector=mrmr.mrmr_classif,
):

    """
    data_dev: development data in a pandas dataframe
    split: split proportion of training and val data each time a fold is executed
    min_features: minimum number of features to return
    max_features: maximum number of features to return
    step_features: step between evaluation of features
    folds: number of repeat evaluations of one feature space size. The training and validation data will be resampled from the dev data each time. It is important that test data is not fed to this function.
    balancer: (default: over_under without replacement) what data balancer to use on the training data each time the dev data is split.
        supports:
        'over_under without replacement': bp_preprocessing.over_under without performing replacement on the undersampling step
        'over_under with replacement': bp_preprocessing.over_under with performing replacement on the undersampling step
        'over_sampling': imblearn.over_sampling.RandomOverSampler over samples every category except the majority class
        'under_sampling without replacement': imblearn.under_sampling.RandomOverSampler under samples every category except the minority class without performing replacement
        'under_sampling with replacement': imblearn.under_sampling.RandomOverSampler under samples every category except the minority class with performing replacement
    tolerance: (Default=0.01) acceptable level of standard deviation between categories among the training and validation data. Percent categories are calculated first, then their stdev is calculated and compared to the stated tolerance
    target: target column in the dataframe
    model: ML architchture to train to evaluate the best K features via mrmr feature selection
        supports any machine learning model which behaves in the following way:
        model.fit(X_train,y_train)
        sklearn.metrics.<chosen sklearn metric>(y_val,model.predict(X_val))
    score: scoring metric to score the model
        supports sklearn.metrics scoring methods
    featureselector: feature selector algorithm to perform feature selection.
        Currently only supports mrmr


    """
    t0 = time.time()
    ##### Encode the target column #####
    # Creating an instance of label Encoder.
    le = sklearn.preprocessing.LabelEncoder()
    # Create list of encoded labels for dataframe and create "categories" and "cat_in_excess" parameters based on this list. (cat in excess only applies during over_under balancing)
    label = le.fit_transform(data_dev[target])
    categories = np.unique(label)
    cat_in_excess = statistics.mode(label)
    # Add the encoded label column to the dataframe and drop the unencoded label column
    data_dev["encoded_label"] = label
    data_dev = data_dev.drop(columns=target)
    # Create K variable for the list of feature spaces to be tested
    K = list(range(min_features, max_features + 1, step_features))

    ##### Create the balancer parameters #####
    balancer_dict = {
        None: {"method": "NaN", "replacement": "NaN"},
        "over_under with replacement": {"method": "over_under", "replacement": True},
        "over_under without replacement": {
            "method": "over_under",
            "replacement": False,
        },
        "under_sampling with replacement": {"method": "under", "replacement": True},
        "under_sampling without replacement": {"method": "under", "replacement": False},
        "over_sampling": {"method": "over", "replacement": "NaN"},
    }
    balancer_params = balancer_dict.get(balancer)
    replacement = balancer_params["replacement"]
    assert balancer_params != None, "Select a valid balancer option"

    ##### Initiate the results list and begin the folds loop #####
    sum_folded_features = []
    i = 0
    while i < folds:
        # Split dev data
        data_train, data_val = split_cats_by_tolerance(
            data_dev,
            tolerance,
            silent=True,
            split=split,
            target="encoded_label",
            categories=categories,
        )
        X_train = data_train.drop(columns="encoded_label")
        y_train = data_train["encoded_label"]
        X_val = data_val.drop(columns="encoded_label")
        y_val = data_val["encoded_label"]

        # Balance the training data
        if balancer_params["method"] == "NaN":
            pass
        elif balancer_params["method"] == "over_under":
            data_train = over_under(
                data_train,
                cat_in_excess=cat_in_excess,
                target="encoded_label",
                silent=True,
                replacement=replacement,
            )
            X_train = data_train.drop(columns="encoded_label")
            y_train = data_train["encoded_label"]
            X_val = data_val.drop(columns="encoded_label")
            y_val = pd.DataFrame(data_val["encoded_label"], columns=["encoded_label"])
        elif balancer_params["method"] == "under":
            under_sampler = imblearn.under_sampling.RandomUnderSampler(
                replacement=replacement
            )
            X_train, y_train = under_sampler.fit_resample(X_train, y_train)
        elif balancer_params["method"] == "over":
            over_sampler = imblearn.over_sampling.RandomOverSampler()
            X_train, y_train = over_sampler.fit_resample(X_train, y_train)

        # Generate the chosen features from this fold along with the score according to the model selected
        single_fold_features = mrmr_shorthand(
            X_train, y_train, X_val, y_val, K, model, score
        )
        sum_folded_features.append(single_fold_features)
        i += 1

    # Create dataframe from list of features discovered in loop of folds
    sum_folded_features = pd.DataFrame(data=sum_folded_features)

    # Calculate the statistical performance of the features across all the folds
    performance_mean = []
    for j in range(0, len(sum_folded_features.columns)):
        performance_values = []
        for i in range(0, len(sum_folded_features)):
            folded_list = sum_folded_features[j].iloc[i]
            featnum = folded_list[0]
            scores = folded_list[1]
            feats = folded_list[2]
            performance_values.append([featnum, scores, feats])

        # List scores from all folds and take their mean
        scores = []
        for i in range(0, len(performance_values)):
            scores.append(performance_values[i][1])
        mean_score = np.mean(scores)

        # List features from all folds
        features = []
        for i in range(0, len(performance_values)):
            features.append(performance_values[i][2])
        uniquefeatures = list(np.unique(features))

        # Calculate statistical information
        std = np.std(scores)
        onesigma = scipy.stats.norm.interval(0.68, loc=mean_score, scale=std)
        twosigma = scipy.stats.norm.interval(0.95, loc=mean_score, scale=std)
        threesigma = scipy.stats.norm.interval(0.99, loc=mean_score, scale=std)
        yerrone = float(np.diff(onesigma)) / 2
        yerrtwo = float(np.diff(twosigma)) / 2
        yerrthree = float(np.diff(threesigma)) / 2

        performance_mean.append(
            [
                featnum,
                scores,
                mean_score,
                std,
                onesigma,
                twosigma,
                threesigma,
                yerrone,
                yerrtwo,
                yerrthree,
                features,
                uniquefeatures,
            ]
        )
        (
            number,
            scores,
            mean_score,
            std,
            onesigma,
            twosigma,
            threesigma,
            yerrone,
            yerrtwo,
            yerrthree,
            features,
            uniquefeatures,
        ) = zip(*performance_mean)

    performance_mean = pd.DataFrame(
        data=performance_mean,
        columns=[
            "feature_num",
            "ind_scores",
            "avg_score",
            "stdev",
            "onesigma",
            "twosigma",
            "threesigma",
            "yerrone",
            "yerrtwo",
            "yerrthree",
            "features",
            "uniquefeatures",
        ],
    )

    t1 = time.time()
    total_time = t1 - t0
    print("Time elapsed:", total_time)
    return performance_mean


def get_feature_dict(frame, uniqueness):
    """
    Generate a feature dictionary from the performance df output by mrmr_feature_selection
    uniqueness: 'unique' or 'duplicate'. generate dictionary with unique feature names across the folds, or from duplicate feature names across the folds
    """
    count_dict = {}
    for i in range(0, len(frame)):
        # Convert the string into a list of lists
        list_of_lists = frame["features"].iloc[i]
        if type(uniqueness) == int:
            lst, count = np.unique(list_of_lists, return_counts=True)
            lst = lst[count > uniqueness - 1]
        elif uniqueness == "all":
            lst, count = np.unique(list_of_lists, return_counts=True)
            lst = lst[count == len(list_of_lists)]
            # print('features in duplicate list from feature_num',n,':',len(lst))
        else:
            return "Invalid value for parameter 'uniqueness'. Use 'all' or an integer."
        count_dict[frame["feature_num"].iloc[i]] = [len(lst), lst]
    return count_dict


def features_ina_string(frame, uniqueness):
    """
    Reutrn a list of lists for each row of the performance df output by mrmr_feature_selection.
    Assumes features are in 'feautres' column.
    uniqueness: 'unique' or 'duplicate' whether the strings contain the unique or duplicate values from each fold in the features column
    generally only called within altair_feautre_selection_chart visualizer
    """

    def unique_features_to_string(X, uniqueness):
        if type(X) == str:
            string = X
        elif type(X) == list:
            string = str(X)
        # remove the outer quotation marks and brackets
        string = string[2:-2]

        # split the string into a list of lists of values
        lst = [s.strip().split(", ") for s in string.split("], [")]
        lst = np.array(lst).flatten()

        if type(uniqueness) == int:
            lst, count = np.unique(lst, return_counts=True)
            lst = lst[count > uniqueness - 1]
        elif uniqueness == "all":
            lst, count = np.unique(lst, return_counts=True)
            lst = lst[count == len(lst)]
            # print('features in duplicate list from feature_num',n,':',len(lst))
        else:
            raise Exception(
                "Invalid value for parameter 'uniqueness'. Use 'all' or an integer."
            )
        hyphen_items = []
        for i, item in enumerate(lst):
            # if it's an odd index, add a hyphen before the item
            if i != 1 and i % 20 == 1:
                hyphen_items.append("- " + item)
            else:
                hyphen_items.append(item)
        stringy = ", ".join(hyphen_items)
        stringy = stringy.replace("'", "")
        return stringy

    features_string = []
    count_dict = {}
    for i in range(0, len(frame)):
        feature = unique_features_to_string(
            frame["features"].iloc[i], uniqueness=uniqueness
        )
        features_string.append(feature)

    return features_string


def altair_feature_selection_chart(frame, uniqueness):
    """
    Assumes independent scores list is in ind_scores
    Assumes feature number is in feature_num

    """
    df = frame

    # features_ina_string assumes features are in 'features' column
    df["uniqueX_features_ina_string"] = features_ina_string(df, uniqueness=uniqueness)
    # def feature_selection_vis(frame,quantitative,ordinal)
    selection = alt.selection_single(empty="none")

    # Use the explode method to expand the dataframe for the error bars calculation
    df_exploded = df.explode("ind_scores").reset_index(drop=True)
    df_exploded.head(10)
    error_bars = (
        alt.Chart(df_exploded)
        .mark_boxplot()
        .encode(
            y=alt.Y("ind_scores:Q"),
            x=alt.X("feature_num:O"),
        )
        .add_selection(selection)
    )

    points = (
        alt.Chart(df)
        .mark_point(filled=True, size=100,color="black")
        .encode(
            y=alt.Y("avg_score:Q", title="score"),
            x=alt.X("feature_num:O", title="feature_num"),
        )
        .add_selection(selection)
    )

    # Create a list to split into multiple lines in the chart
    df["featuresX"] = (
        df["uniqueX_features_ina_string"].astype(str).str.split("-")
    )  # Create a list to split into multiple lines in the chart
    df["zero"] = 0
    text = (
        alt.Chart(df)
        .mark_text(baseline="bottom", dy=-75)
        .encode(y="zero", text="featuresX")
        .transform_filter(selection)
    )

    chart = (
        (error_bars + points + text)
        .properties(title="plot", height=600, width=1000)
        .interactive()
    )

    return chart
