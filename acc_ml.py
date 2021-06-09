'''
Peiyu(Ian) Lu
This program is a class for setting up and running machine learning
decision tree model for car severity prediction from different
input data
'''
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
sns.set()


class Ml_Model:

    def __init__(self, df, mask, features, labels):
        '''
        this is the initializer of the class which takes
        in a panda dataframe df, filtering mask, features, and
        labels for setting up the decision tree. Other variables
        includes statistical fields for ML model.
        The default value of statistical
        fields is -1
        '''
        self._mask = mask
        self._features = features
        self._df = df
        self._labels = labels
        self._cv_acc = -1
        self._cv_mean = -1
        self._cv_sd = -1
        self._split_train = -1
        self._split_test = -1

    def run_model(self):
        '''
        this function will finalize the features and labels
        been passed in the ML models
        '''
        if self._mask != '':
            self._df = self._df[eval(self._mask)]
        cols = self._features + self._labels
        self._df = self._df[cols].dropna()
        self._features = self._df[self._features]
        self._features = pd.get_dummies(self._features)
        self._labels = self._df[self._labels]
        self.set_up_model()

    def set_up_model(self):
        '''
        This function uses the features and label to
        set up a cross-validation model from cross_model
        function and update the statistical filed in the class.
        Additionally, the function also get the ideal depth of the
        decision tree for the train_test model which will be set up
        by calling the predict function.
        '''
        # try to get the best depth of the decision tree
        sc_mean, sc_std, acc_sc = self.cross_model(
            20, 6)  # maybe try bigger number
        # get the index of the max element in a numpy
        id_of_max = sc_mean.argmax()
        ideal_depth = id_of_max + 1
        self._cv_mean = sc_mean[id_of_max]
        self._cv_sd = sc_std[id_of_max]
        self._cv_acc = acc_sc[id_of_max]
        print('Cross_validation model results: ')
        print('''ideal depth: {}   \nmean accuracy : {}
            standard deviation : {}  \nacc_score : {}'''.format(
            ideal_depth, round(self._cv_mean, 6),
            round(self._cv_sd, 6), round(self._cv_acc, 5)))
        # set up the model with the best depth
        model = DecisionTreeClassifier(max_depth=ideal_depth)
        self.predict(model)

    def predict(self, model):
        '''
        This function takes in the model passed in and
        perform a train_test split on the model with a ratio
        of train set being 80 percent, and test set being 20 percent
        Data will be printed in the console
        '''
        features_train, features_test, labels_train, labels_test = \
            train_test_split(self._features, self._labels, test_size=0.2)
        model.fit(features_train, labels_train)
        self._split_train = model.predict(features_train)
        print('train_test_split model results: ')
        print('Train Accuracy:', round(
            accuracy_score(labels_train, self._split_train), 6))
        self._split_test = model.predict(features_test)
        print('Test  Accuracy:', round(
            accuracy_score(labels_test, self._split_test), 6))

    def cross_model(self, max_depth, level):
        '''
        this function takes in max_depth, and
        level(folds) of the cross_validation and run a cross validation
        on the given data and print out the results.
        '''
        standard_d = []
        mean = []
        accuracy_sc = []
        for depth in range(1, max_depth):
            tree_model = DecisionTreeClassifier(max_depth=depth)
            cv_scores = cross_val_score(  # cv_scores is an arrayof the scores
                tree_model, self._features,
                self._labels, cv=level, scoring='accuracy')
            # estimates the expected accuracy of your model training data
            mean.append(cv_scores.mean())
            standard_d.append(cv_scores.std())
            acc_score = tree_model.fit(
                self._features,
                self._labels).score(self._features, self._labels)
            accuracy_sc.append(acc_score)

        # convert them to numpy
        mean_np = np.array(mean)
        std_np = np.array(standard_d)
        acc_np = np.array(accuracy_sc)
        return (mean_np, std_np, acc_np)

    def get_data(self):
        return (self._cv_acc,
                self._cv_mean,
                self._cv_sd,
                self._split_train,
                self._split_test)
