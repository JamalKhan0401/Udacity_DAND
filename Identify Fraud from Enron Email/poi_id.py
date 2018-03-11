#!/usr/bin/python

import sys
import pickle
import numpy as np
import pprint
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
import matplotlib.pyplot as plt


from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import GaussianNB

FS_OPTIONS = ['SelectKBest', 'tree']
FEATURE_SELECTOR = FS_OPTIONS[0]

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary', 'expenses', 'total_stock_value', 'bonus', 
                 'from_poi_to_this_person', 'shared_receipt_with_poi'] 
# You will need to use more features


features_financial = ['salary', 'deferral_payments', 'total_payments', 
                      'loan_advances', 'bonus', 'restricted_stock_deferred', 
                      'deferred_income', 'total_stock_value', 'expenses', 
                      'exercised_stock_options', 'other', 
                      'long_term_incentive', 'restricted_stock', 
                      'director_fees']

features_email = ['to_messages', 'email_address', 'from_poi_to_this_person', 
                  'from_messages', 'from_this_person_to_poi', 
                  'shared_receipt_with_poi']

POI_label = ['poi']
total_features = POI_label + features_financial + features_email

### Load the dictionary containing the dataset
with open("../final_project/final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

print 'Total number of data points = ', len(data_dict)
# allocation across classes (POI/non-POI)
poi_count = 0
for employee_data in data_dict:
    if data_dict[employee_data]['poi'] == True:
        poi_count += 1
print 'number of POI = ', poi_count
print 'number of non-POI = ', len(data_dict) - poi_count

# number of features used
print 'total number of available features for every employee = '
print len(total_features), 'which are: ', total_features
print 'number of features used = ', len(features_list), 'which are: '
print features_list

# are there features with many missing values? etc.
missing_data = {}
for feat in total_features:
    missing_data[feat] = 0

for emp in data_dict:
    for f in data_dict[emp]:
        if data_dict[emp][f] == 'NaN':
            missing_data[f] += 1
            # fill NaN values
            # data_dict[emp][f] = 0

print 'missing values: ', missing_data

### Task 2: Remove outliers
def draw_scatter_plot(dataset, feature1, feature2):
    """ given two features feature1 (x) and feature2 (y),
    this function creates a 2D scatter plot showing
    both x and y
    """
    data = featureFormat(dataset, [feature1, feature2])
    for p in data:
        x = p[0]
        y = p[1]
        plt.scatter(x, y)

    plt.xlabel(feature1)
    plt.ylabel(feature2)
    plt.show()

# identify outliers
draw_scatter_plot(data_dict, "salary", "bonus")

data_dict.pop( "TOTAL", 0 )
draw_scatter_plot(data_dict, "salary", "bonus")

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

# create new features
def calculateFraction(poi_messages, all_messages):
    """ given a number messages to/from POI (numerator) 
    and number of all messages to/from a person (denominator),
    return the fraction of messages to/from that person
    that are from/to a POI
    """
    fraction = 0
    if poi_messages != 'NaN' and all_messages != 'NaN':
        fraction = poi_messages/float(all_messages)

    return fraction

def takeSecond(elem):
    """ take second element for sort
    """
    return elem[1]

for employee in my_dataset:
    from_poi_to_this_person = my_dataset[employee]['from_poi_to_this_person']
    to_messages = my_dataset[employee]['to_messages']
    fraction_from_poi = calculateFraction(from_poi_to_this_person, to_messages)
    #print "Fraction - ", fraction_from_poi
    my_dataset[employee]['fraction_from_poi'] = fraction_from_poi

    from_this_person_to_poi = my_dataset[employee]['from_this_person_to_poi']
    from_messages = my_dataset[employee]['from_messages']
    fraction_to_poi = calculateFraction(from_this_person_to_poi, from_messages)
    my_dataset[employee]['fraction_to_poi'] = fraction_to_poi

features_list_n = total_features
features_list_n.remove('email_address')
features_list_n =  features_list_n + ['fraction_from_poi', 'fraction_to_poi']
print "Feature List-", features_list_n


### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list_n, sort_keys = True)
labels, features = targetFeatureSplit(data)

# intelligently select features (univariate feature selection)
from sklearn.feature_selection import SelectKBest, f_classif
selector = SelectKBest(f_classif, k = 8)
selector.fit(features, labels)
scores = zip(features_list_n[1:], selector.scores_)
sorted_scores = sorted(scores, key = takeSecond, reverse = True)
print 'SelectKBest scores: ', sorted_scores

kBest_features = POI_label + [(i[0]) for i in sorted_scores[0:8]]
print 'KBest', kBest_features

for emp in data_dict:
    for f in data_dict[emp]:
        if data_dict[emp][f] == 'NaN':
            # fill NaN values
            data_dict[emp][f] = 0

my_dataset = data_dict

kBest_features.remove('fraction_to_poi')

##Added New Stuff Start
def test_classifier(clf, features, labels):
    cv = StratifiedShuffleSplit(labels, 1000, random_state=42)
    true_negatives = 0
    false_negatives = 0
    true_positives = 0
    false_positives = 0
    for train_idx, test_idx in cv:
        features_train = []
        features_test = []
        labels_train = []
        labels_test = []
        for ii in train_idx:
            features_train.append(features[ii])
            labels_train.append(labels[ii])
        for jj in test_idx:
            features_test.append(features[jj])
            labels_test.append(labels[jj])

        # fit the classifier using training set, and test on test set
        clf.fit(features_train, labels_train)
        predictions = clf.predict(features_test)
        for prediction, truth in zip(predictions, labels_test):
            if prediction == 0 and truth == 0:
                true_negatives += 1
            elif prediction == 0 and truth == 1:
                false_negatives += 1
            elif prediction == 1 and truth == 0:
                false_positives += 1
            elif prediction == 1 and truth == 1:
                true_positives += 1
            else:
                print "Warning: Found a predicted label not == 0 or 1."
                print "All predictions should take value 0 or 1."
                print "Evaluating performance for processed predictions:"
                break
    try:
        total_predictions = np.sum([true_negatives, false_negatives,
                                    false_positives, true_positives])
        accuracy = 1.0*(true_positives + true_negatives)/total_predictions
        precision = 1.0*true_positives/(true_positives+false_positives)
        recall = 1.0*true_positives/(true_positives+false_negatives)
        f1 = 2.0 * true_positives/(2*true_positives +
                                   false_positives+false_negatives)
        f2 = (1+2.0*2.0) * precision*recall/(4*precision + recall)
        return total_predictions, accuracy, precision, recall,\
            true_positives, false_positives, true_negatives, \
            false_negatives, f1, f2
    except:
        print "Got a divide by zero when trying out:", clf
        print "Precision or recall may be undefined due to a lack of \
        true positive predicitons."

def getClassifiers():
    """Returns tuned classifiers
    """
    gnb = GaussianNB()
    # decision tree after tuning with gridsearch
    tree = DecisionTreeClassifier(max_features=9, min_samples_split=4,
                                  criterion='entropy', max_depth=10,
                                  min_samples_leaf=2)

    # kneighbour after tuning with gridsearch
    kneighbour = KNeighborsClassifier(n_neighbors=5, weights='uniform',
                                      leaf_size=1, algorithm='auto', p=1)
    return gnb, tree, kneighbour


def testClassifers(classifiers):
    """Calls the test_classifier from tester.py for each classifier
    args:
        classifiers: list of classifiers to test
    """
    PERF_FORMAT_STRING = "\
    \tAccuracy: {:>0.{display_precision}f}\t\
    Precision: {:>0.{display_precision}f}\t\
    Recall: {:>0.{display_precision}f}\t"
    RESULTS_FORMAT_STRING = "\tTotal predictions: {:4d}\t\
    True positives: {:4d}\tFalse positives: {:4d}\
    \tFalse negatives: {:4d}\tTrue negatives: {:4d}"
    print "Classifier Test Results"
    print "==================================="
    for clf in classifiers:
        total_predictions, accuracy, precision, recall, true_positives, \
            false_positives, true_negatives, false_negatives, f1, f2 = \
            test_classifier(clf, features, labels)
        print clf
        print PERF_FORMAT_STRING.format(accuracy, precision, recall,
                                        display_precision=5)
        print RESULTS_FORMAT_STRING.format(total_predictions, true_positives,
                                           false_positives, false_negatives,
                                           true_negatives)
        
def getTrainingTestSets(labels, features):
    """ Creates training and test sets based on the StratifiedShuffleSplit
    args:
        labels: list of labels from the data
        features: list of features in the data
    """
    cv = StratifiedShuffleSplit(labels, 1000, random_state=42)
    for train_idx, test_idx in cv:
        features_train = []
        features_test = []
        labels_train = []
        labels_test = []
        for ii in train_idx:
            features_train.append(features[ii])
            labels_train.append(labels[ii])
        for jj in test_idx:
            features_test.append(features[jj])
            labels_test.append(labels[jj])
    return features_train, features_test, labels_train, labels_test

def getBestFeatures(features, labels, num_features, showResults=False, num_max_features=0):
    """ Returns the best features based on the Feature Selection Options
    The features are selected based on the highest score / importance
    args:
        labels: list of labels from the data
        features: list of features in the data
        showResults: boolean set to true to print list of features and scores
    """
    features_train, features_test, labels_train, labels_test = \
        getTrainingTestSets(labels, features)
    revised_feature_list = ['poi']
    if FEATURE_SELECTOR == "tree":
        clf = DecisionTreeClassifier()
        clf = clf.fit(features_train, labels_train)
        importance = clf.feature_importances_
    else:
        if num_features == num_max_features:
            k_best = SelectKBest(k='all')
        else:
            k_best = SelectKBest(k=num_features)
        k_best.fit(features_train, labels_train)
        importance = k_best.scores_

    feature_scores = sorted(zip(features_list[1:], importance),
                            key=lambda l: l[1], reverse=True)
    for feature, importance in feature_scores[:num_features]:
        revised_feature_list.append(feature)
    if showResults:
        print "Top features and scores:"
        print "==================================="
        pprint.pprint(feature_scores[:num_features])
    return revised_feature_list

def scoreNumFeatures(test_feature_list, test_data_set):
    """ function for determining the best number of features to use
    """
    scaler = MinMaxScaler()
    accuracy_scores = []
    recall_scores = []
    precision_scores = []
    feature_count = []
    f1_scores = []
    PERF_FORMAT_STRING = "\
    Features: {:>0.{display_precision}f}\t\
    Accuracy: {:>0.{display_precision}f}\t\
    Precision: {:>0.{display_precision}f}\t\
    Recall: {:>0.{display_precision}f}\t\
    F1: {:>0.{display_precision}f}\t\
    "

    gnb, tree, kneighbour = getClassifiers()
    clf = kneighbour
    for x in range(1, len(test_feature_list)+1):
        test_data = featureFormat(test_data_set, test_feature_list,
                                  sort_keys=True)
        test_labels, test_features = targetFeatureSplit(test_data)
        test_features = scaler.fit_transform(test_features)
        best_features = getBestFeatures(test_features, test_labels, x, False,len(test_feature_list))
        # Resplit data using best feature list
        test_data = featureFormat(test_data_set, best_features,
                                  sort_keys=True)
        test_labels, test_features = targetFeatureSplit(test_data)
        test_features = scaler.fit_transform(test_features)
        total_predictions, accuracy, precision, recall, true_positives, \
            false_positives, true_negatives, false_negatives, f1, f2 = \
            test_classifier(clf, test_features, test_labels)
        print PERF_FORMAT_STRING.format(x, accuracy, precision, recall, f1,
                                        display_precision=5)
        accuracy_scores.append(accuracy)
        recall_scores.append(recall)
        precision_scores.append(precision)
        f1_scores.append(f1)
        feature_count.append(x)

    plt.plot(feature_count, accuracy_scores, marker='o', label="Accuracy")
    plt.plot(feature_count, precision_scores, marker='o', label="Precision")
    plt.plot(feature_count, recall_scores, marker='o', label="Recall")  
    #plt.plot(feature_count, f1_scores, marker='o', label="F1")
    plt.legend()
    plt.title("Accuracy, Precision, Recall vs Number of K-Best features")
    plt.xlabel('Number of K Best Features')
    plt.ylabel('Score')
    plt.show()
 
scoreNumFeatures(kBest_features, my_dataset)
##Added New Stuff End
# dataset without new features
from sklearn import preprocessing
data = featureFormat(my_dataset, kBest_features, sort_keys = True)
labels, features = targetFeatureSplit(data)
scaler = preprocessing.MinMaxScaler()
features = scaler.fit_transform(features)

# dataset with new features
kBest_new_features = kBest_features + ['fraction_from_poi', 'fraction_to_poi']
data = featureFormat(my_dataset, kBest_new_features, sort_keys = True)
new_labels, new_features = targetFeatureSplit(data)
new_features = scaler.fit_transform(new_features)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# split 30% of the data for testing
from sklearn.cross_validation import train_test_split
from sklearn import cross_validation
features_train, features_test, labels_train, labels_test = \
    cross_validation.train_test_split(features, labels, test_size=0.3, 
                                      random_state=42)

from sklearn.metrics import accuracy_score, precision_score, recall_score

# # Provided to give you a starting point. Try a variety of classifiers.
#from sklearn.naive_bayes import GaussianNB
#from time import time
clf_gNB = GaussianNB()
clf_gNB.fit(features_train, labels_train)
pred = clf_gNB.predict(features_test)
nb_score = clf_gNB.score(features_test, labels_test)

gnb_acc = accuracy_score(labels_test, pred)
gnb_pre = precision_score(labels_test, pred)
gnb_rec = recall_score(labels_test, pred)
print "GNB accuracy: ", gnb_acc
print "GNB precision: ", gnb_pre
print "GNB recall: ", gnb_rec

from sklearn import tree
clf_DTC = tree.DecisionTreeClassifier()
clf_DTC.fit(features_train, labels_train)
pred = clf_DTC.predict(features_test)
dt_score = clf_DTC.score(features_test, labels_test)

dtc_acc = accuracy_score(labels_test, pred)
dtc_pre = precision_score(labels_test, pred)
dtc_rec = recall_score(labels_test, pred)
print "DTC accuracy: ", dtc_acc
print "DTC precision: ", dtc_pre
print "DTC recall: ", dtc_rec

from sklearn.ensemble import RandomForestClassifier
clf_RFC = RandomForestClassifier(n_estimators=10)
clf_RFC.fit(features_train, labels_train)
pred = clf_RFC.predict(features_test)
rf_score = clf_RFC.score(features_test, labels_test)

rfc_acc = accuracy_score(labels_test, pred)
rfc_pre = precision_score(labels_test, pred)
rfc_rec = recall_score(labels_test, pred)
print "RFC accuracy: ", rfc_acc
print "RFC precision: ", rfc_pre
print "RFC recall: ", rfc_rec

from sklearn.linear_model import LogisticRegression
clf_LR = LogisticRegression(C=1e5)
clf_LR.fit(features_train, labels_train)
pred = clf_LR.predict(features_test)
lr_acc = accuracy_score(labels_test, pred)
lr_pre = precision_score(labels_test, pred)
lr_rec = recall_score(labels_test, pred)
print "LR accuracy: ", lr_acc
print "LR precision: ", lr_pre
print "LR recall: ", lr_rec

#from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score

def tune_parameters(grid_search, features, labels, params, iters = 80):
    """ given a grid_search and parameters list (if exist) for a specific model,
    along with features and labels list,
    it tunes the algorithm using grid search and prints out the average 
    evaluation metrics     results (accuracy, percision, recall) after 
    performing the tuning for iter times,
    and the best hyperparameters for the model
    """
    accuracy = []
    precision = []
    recall = []
    for i in range(iters):
        
        #features_train, features_test, labels_train, labels_test = \
        #cross_validation.train_test_split(features, labels, test_size=0.3, 
        #                              random_state=42)
        
        features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size = 0.3, random_state = i)
        grid_search.fit(features_train, labels_train)
        predicts = grid_search.predict(features_test)

        accuracy = accuracy + [accuracy_score(labels_test, predicts)] 
        precision = precision + [precision_score(labels_test, predicts)]
        recall = recall + [recall_score(labels_test, predicts)]
    print "accuracy: {}".format(np.mean(accuracy))
    print "precision: {}".format(np.mean(precision))
    print "recall: {}".format(np.mean(recall))

    best_params = grid_search.best_estimator_.get_params()
    for param_name in params.keys():
        print("%s = %r, " % (param_name, best_params[param_name]))


from sklearn.model_selection import GridSearchCV

# 1. Naive Bayes
from sklearn.naive_bayes import GaussianNB
gnb_clf = GaussianNB()
gnb_param = {}
gnb_grid_search = GridSearchCV(estimator = gnb_clf, param_grid = gnb_param)
print("Naive Bayes model evaluation")
print("Without New features")
tune_parameters(gnb_grid_search, features, labels, gnb_param)
print("With New features")
tune_parameters(gnb_grid_search, new_features, new_labels, gnb_param)


# 2. Support Vector Machines
from sklearn import svm
svm_clf = svm.SVC()
svm_param = {'kernel':('linear', 'rbf', 'sigmoid'),
'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
'C': [0.1, 1, 10, 100, 1000]}
svm_grid_search = GridSearchCV(estimator = svm_clf, param_grid = svm_param)

print("SVM model evaluation")
print("Without New features")
tune_parameters(svm_grid_search, features, labels, svm_param)
print("With New features")
tune_parameters(svm_grid_search, new_features, new_labels, svm_param)


# 3. Decision Tree
from sklearn import tree
dtc_clf = tree.DecisionTreeClassifier()
dtc_param = {'criterion':('gini', 'entropy'),
'splitter':('best','random')}
dtc_grid_search = GridSearchCV(estimator = dtc_clf, param_grid = dtc_param)

print("Decision Tree model evaluation")
print("Without New features")
tune_parameters(dtc_grid_search, features, labels, dtc_param)
print("With New features")
tune_parameters(dtc_grid_search, new_features, new_labels, dtc_param)

# 4. Random Forest
from sklearn.ensemble import RandomForestClassifier
rfc_clf = RandomForestClassifier(n_estimators=10)
rfc_param = {}
rfc_grid_search = GridSearchCV(estimator = rfc_clf, param_grid = rfc_param)

print("Random Forest model evaluation")
print("Without New features")
tune_parameters(rfc_grid_search, features, labels, rfc_param)
print("With New features")
tune_parameters(rfc_grid_search, new_features, new_labels, rfc_param)

# 5. Logistic Regression
from sklearn.linear_model import LogisticRegression
lr_clf = LogisticRegression()
lr_param = {'tol': [1, 0.1, 0.01, 0.001, 0.0001],
'C': [0.1, 0.01, 0.001, 0.0001]}
lr_grid_search = GridSearchCV(estimator = lr_clf, param_grid = lr_param)

print("Logistic Regression model evaluation")
print("Without New features")
tune_parameters(lr_grid_search, features, labels, lr_param)
print("With New features")
tune_parameters(lr_grid_search, new_features, new_labels, lr_param)

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.
### StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
#from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

clf = clf_gNB
features_list = kBest_features


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.
dump_classifier_and_data(clf, my_dataset, features_list)