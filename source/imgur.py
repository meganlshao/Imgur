# imgur.py, created by Megan Shao (mshao@hmc.edu) and Vincent Fiorentini
# (vfiorentini@hmc.edu).
# Runs baseline majority classifier and ordinal regression model for predicting
# popularity of Imgur comments.

import csv
from string import punctuation
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import KFold
from scipy.stats import ttest_rel
import numpy as np
from math import sqrt
from collections import Counter

from datetime import datetime

# from https://github.com/fabianp/mord
from mord.threshold_based import LogisticAT


CROSS_VALIDATE = False
PLOT_RESULTS = False

CSV_FILE_PATH = '../data/commentFeaturesList_1448649764.csv' # ~62k samples
NUM_TOTAL_SAMPLES = 10000 # assumes above file contains at least this number
NUM_TRAIN_SAMPLES = int(NUM_TOTAL_SAMPLES * 0.9)
CLASS_UPPER_BOUNDS = [1] # 2-class 
# CLASS_UPPER_BOUNDS = [0, 1, 5] # 4-class
# CLASS_UPPER_BOUNDS = [-4, 0, 1, 5, 10] # 6-class
NUM_FOLDS = 10
ALPHA = 1.0
MAX_ITER = 10000


def read_csv(filepath):
    """
    Parameters
    --------------------
        filepath         -- path to csv file of samples, expected format:
                            0. 'text', 1. 'isReply', 2. 'isAuthorOP', 
                            3. 'datetimeDelta', 4. 'upvotes', 5. 'downvotes', 
                            6. 'netUpvotes', 7. 'postTitle', 8. 'postTime', 
                            9. 'postDescription', 10. 'postCategory',
                            11. 'postNetUpvotes'

    Returns
    --------------------
        raw_features     -- list of comments as raw features (0. 'text', 
                            1. 'isReply', 2. 'isAuthorOP', 3. 'datetimeDelta',
                            4. 'postTitle', 5. 'postTime', 6. 'postDescription',
                            7. 'postCategory', 8. 'postNetUpvotes') from csv
        raw_labels       -- list of labels (netUpvotes)
    """
    comment_features = []
    with open(filepath, 'rb') as f:
        reader = csv.reader(f)
        for i in range(NUM_TOTAL_SAMPLES + 1):
            comment_features.append(next(reader))
    # remove headers
    comment_features = comment_features[1:]
    np.random.seed(1234)
    np.random.shuffle(comment_features)
    # separate raw data into features and labels
    raw_features = [(comment[0:4] + comment[7:]) for comment in comment_features]
    raw_labels = [comment[6] for comment in comment_features]
    return raw_features, raw_labels

def extract_feature_vectors(raw_features):
    """
    Parameters
    --------------------
        raw_features     -- list of comments as raw features (0. 'text', 
                            1. 'isReply', 2. 'isAuthorOP', 3. 'datetimeDelta',
                            4. 'postTitle', 5. 'postTime', 6. 'postDescription',
                            7. 'postCategory', 8. 'postNetUpvotes') from csv

    Returns
    --------------------
        feature_matrix   -- array of feature vectors
    """
    # remove postDescription and postCategory features
    raw_features = [(comment[0:6] + [comment[8]]) for comment in raw_features]
    # raw_features:
    # 0. 'text', 1. 'isReply', 2. 'isAuthorOP', 3. 'datetimeDelta',
    # 4. 'postTitle', 5. 'postTime', 6. 'postNetUpvotes'

    num_lines = len(raw_features)
    num_features = 8 + 7 + 24 # 8 features plus 7 days of week plus 24 hours in day
    feature_matrix = np.zeros((num_lines, num_features))

    # load AFINN (http://www2.imm.dtu.dk/pubdb/views/publication_details.php?id=6010)
    afinn_path = "AFINN/AFINN-111.txt"
    afinn = dict(map(lambda (k,v): (k,int(v)), [ line.split('\t') for line in open(afinn_path) ]))

    for i in range(num_lines):
        raw_feature = raw_features[i]
        # set isReply feature
        if raw_feature[1] == 'True':
            feature_matrix[i][0] = 1
        else:
            feature_matrix[i][0] = 0
        # set isAuthorOP feature
        if raw_feature[2] == 'True':
            feature_matrix[i][1] = 1
        else:
            feature_matrix[i][1] = 0
        # set datetimeDelta feature - convert seconds to hours
        feature_matrix[i][2] = int(raw_feature[3]) / 3600

        # set postNetUpvotes feature
        feature_matrix[i][3] = int(raw_feature[-1])

        # set comment percent capitalized
        feature_matrix[i][4] = get_percent_capitalized(raw_feature[0])
        # set comment length
        feature_matrix[i][5] = len(raw_feature[0])
        # set number of links
        feature_matrix[i][6] = raw_feature[0].count("http://")
        # set AFINN sentiment score
        sentiment = sum(map(lambda word: afinn.get(word, 0), raw_feature[0].lower().split()))
        feature_matrix[i][7] = sentiment

        # set post day of week
        dt = datetime.fromtimestamp(int(raw_feature[-2]))
        weekday = dt.weekday() # in range [0, 6]
        feature_matrix[i][8 + weekday] = 1

        # set post hour of day
        dt = datetime.fromtimestamp(int(raw_feature[-2]))
        hour = dt.hour # in range [0, 23]
        feature_matrix[i][8 + 7 + hour] = 1

    return feature_matrix

def get_percent_capitalized(line):
    """
    Parameters
    --------------------
        line      -- string

    Returns
    --------------------
                  -- float, proportion of letters that are capitalized
    """
    num_cap_letters = 0
    num_letters = 0
    for c in line:
        if c.isalpha():
            num_letters += 1
        if c.isupper():
            num_cap_letters += 1
    if num_letters == 0:
        return 0
    else:
        return float(num_cap_letters) / num_letters

def extract_X_y(filename):
    """
    Parameters
    --------------------
        filename  -- name of csv file (string)

    Returns
    --------------------
        X         -- feature vectors
        y         -- labels
    """
    raw_features, raw_labels = read_csv(filename)
    X = extract_feature_vectors(raw_features)
    print("Extracted %d feature vectors" % len(X))
    y = np.asarray([int(label) for label in raw_labels])
    return X, y

def get_multiclass_labels(y):
    """
    Parameters
    --------------------
        y -- labels as net upvotes

    Returns
    --------------------
        y -- labels for multiclass
             2-class labels are:
               0. unpopular/neutral (less than or equal to +1 net upvote)
               1. popular (at least +2)
             4-class labels are:
               0. unpopular (less than +1 net upvote)
               1. neutral (+1)
               2. popular (+2 to +5)
               3. very popular (at least +6)
             6-class labels are:
               0. very unpopular (less than or equal to -4 net upvotes)
               1. slightly unpopular (-3 to 0)
               2. neutral (+1)
               3. slightly popular (+2 to +5)
               4. popular (+6 to +10)
               5. most popular (at least +11)
    """
    for index, label in enumerate(y):
        y[index] = len(CLASS_UPPER_BOUNDS)
        for class_index, bound in enumerate(CLASS_UPPER_BOUNDS):
            if label <= bound:
                y[index] = class_index
                break
    return y


def run_majority_classifier(X_train, y_train, X_test, y_test):
    print("Running majority classifier...")
    counter = Counter()
    for net in y_train:
        counter[net] += 1
    majority = counter.most_common()[0][0]

    baseline_num_correct = 0
    for expected in y_train:
        if (expected == majority):
            baseline_num_correct += 1
    print("%.4f = Training accuracy for majority classifier" % 
            (float(baseline_num_correct) / len(y_train)))

    baseline_num_correct = 0
    for expected in y_test:
        if (expected == majority):
            baseline_num_correct += 1
    print("%.4f = Test accuracy for majority classifier" % 
            (float(baseline_num_correct) / len(y_test)))

    return float(baseline_num_correct) / len(y_test)

def run_ordinal_regression(X_train, y_train, X_test, y_test, ordinal_regression_model):
    print("Running ordinal regression with multiclass labels...")
    ordinal_regression_clf = ordinal_regression_model(alpha=ALPHA, max_iter=MAX_ITER)
    ordinal_regression_clf.fit(X_train, y_train)

    y_pred = ordinal_regression_clf.predict(X_train)
    training_err = metrics.zero_one_loss(y_train, y_pred, normalize=False)
    print("%.4f = Training accuracy for ordinal regression with multiclass labels" % 
            (float(len(y_train) - training_err) / len(y_train)))

    y_pred = ordinal_regression_clf.predict(X_test)
    test_err = metrics.zero_one_loss(y_test, y_pred, normalize=False)
    print("%.4f = Test accuracy for ordinal regression with multiclass labels" % 
            (float(len(y_test) - test_err) / len(y_test)))

    return float(len(y_test) - test_err) / len(y_test)

def run_cross_validation(X, y, ordinal_regression_model):
    kf = KFold(len(X), n_folds=NUM_FOLDS)
    majority_scores = []
    ordinal_scores = []
    for train_indices, test_indices in kf:
        X_train = np.asarray([X[index] for index in train_indices])
        y_train = np.asarray([y[index] for index in train_indices])
        X_test = np.asarray([X[index] for index in test_indices])
        y_test = np.asarray([y[index] for index in test_indices])
        
        majority_scores.append(run_majority_classifier(X_train, y_train, X_test, y_test))
        ordinal_scores.append(run_ordinal_regression(X_train, y_train, X_test, y_test, ordinal_regression_model))

    print "The majority scores are ", majority_scores
    print "The ordinal regression scores are ", ordinal_scores

    t, prob = ttest_rel(majority_scores, ordinal_scores)
    print("p-value for majority and ordinal regression is %f" % prob)


def main():
    print("==============================")
    print("Extracting features...")
    X, y = extract_X_y(CSV_FILE_PATH)
    # convert net upvote labels into multiclass labels
    y = get_multiclass_labels(y)


    if CROSS_VALIDATE:
        print("==============================")
        print("----- Running cross validation -----")
        run_cross_validation(X, y, LogisticAT)


    print("==============================")
    print("----- Comparing classifiers using accuracy -----")

    X_train = X[0:NUM_TRAIN_SAMPLES]
    y_train = y[0:NUM_TRAIN_SAMPLES]
    X_test = X[NUM_TRAIN_SAMPLES:]
    y_test = y[NUM_TRAIN_SAMPLES:]
    run_majority_classifier(X_train, y_train, X_test, y_test)
    run_ordinal_regression(X_train, y_train, X_test, y_test, LogisticAT)


if __name__ == "__main__" :
    main()
