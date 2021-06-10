'''
Sean Gombart, Ian Lu, and Michael Christensen
This program is a class for setting up and running machine learning
models for car severity prediction from different
input data
'''
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.inspection import plot_partial_dependence
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

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
        the features passed in should be numeric
        '''
        self._mask = mask
        self._features = features
        self._df = df
        self._labels = labels
        self._features_list = features
        self._labels_list = labels
        self._cv_acc = -1
        self._cv_mean = -1
        self._cv_sd = -1
        self._split_train = -1
        self._split_test = -1
        self._ideal_depth = -1

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
            15, 10)
        # get the index of the max element in a numpy
        id_of_max = sc_mean.argmax()
        ideal_depth = id_of_max + 1
        self._cv_mean = sc_mean[id_of_max]
        self._cv_sd = sc_std[id_of_max]
        self._cv_acc = acc_sc[id_of_max]
        self._ideal_depth = ideal_depth
        model = DecisionTreeClassifier(max_depth=ideal_depth)
        self.predict(model)

    def predict(self, model):
        '''
        This function takes in the model passed in and
        perform a train_test split on the model with a ratio
        of train set being 80 percent, and test set being 20 percent
        the function will also call the plot_partial_dep with
        the first feature in the features list as the plot
        is only taking one feature
        '''
        features_train, features_test, labels_train, labels_test = \
            train_test_split(self._features, self._labels, test_size=0.2)
        model.fit(features_train, labels_train)
        split_train = model.predict(features_train)
        self._split_train = round(
            accuracy_score(labels_train, split_train), 10)
        split_test = model.predict(features_test)
        self._split_test = round(
            accuracy_score(labels_test, split_test), 10)
        plt_col = []
        plt_col.append(self._features_list[0])
        self.plot_partial_dep(model, self._features, plt_col)

    def cross_model(self, max_depth, level):
        '''
        this function takes in max_depth, and
        level(folds) of the cross_validation and run a cross validation
        on the given data. The default number of
        folds for the cross validation process will be set to 6.
        '''
        standard_d = []
        mean = []
        accuracy_sc = []
        for depth in range(1, max_depth):
            tree_model = DecisionTreeClassifier(max_depth=depth)
            cv_scores = cross_val_score(  # cv_scores is an array of the scores
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
        '''
        this function returns the statistical data of the object
        in the form of dictionary.
        '''
        model_stat = {}
        model_stat['ideal depth:  '] = round(self._ideal_depth, 10)
        model_stat['cross val mean accuracy:  '] = round(self._cv_mean, 10)
        model_stat['cross val standard deviation:  '] = round(self._cv_sd, 10)
        model_stat['cross val accuracy score:  '] = round(self._cv_acc, 10)
        model_stat['split train accuracy:  '] = round(self._split_train, 10)
        model_stat['split test accuracy:  '] = round(self._split_test, 10)
        return model_stat

    def plot_partial_dep(self, model, features, name_list):
        '''
        this function takes in a model, features, and labels
        and produce a plot_partial_dep.
        '''
        plot_partial_dependence(model,
                                features,
                                name_list,
                                target=1)
        plt.autoscale()
        plt.savefig('partial_dep.png', bbox_inches='tight')

    def random_forest_plot(self):
        '''
        this function establish a RandomForestClassifier and
        produce a plot that represents the importance of features.
        Make sure the features are all numeric.
        '''
        features_train, _, labels_train, _ = \
            train_test_split(self._features, self._labels,
                             test_size=0.2, random_state=0)
        model = RandomForestClassifier(
            n_estimators=100, n_jobs=-1, random_state=1)
        model.fit(features_train, labels_train)
        features = self._features_list
        importances = model.feature_importances_
        indices = np.argsort(importances)
        plt.figure(figsize=(10, 15))
        plt.title('Feature Importances')
        plt.barh(range(len(indices)),
                 importances[indices], color='r', align='center')
        plt.yticks(range(len(indices)), [features[i] for i in indices])
        plt.xlabel('Relative Importance Plot')
        plt.savefig('importance.png', bbox_inches='tight')
