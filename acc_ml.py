import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
sns.set()
'''
['ID', 'Severity', 'Start_Time', 'End_Time', 'Start_Lat', 'Start_Lng',
       'End_Lat', 'End_Lng', 'Distance(mi)', 'Description', 'Number', 'Street',
       'Side', 'City', 'County', 'State', 'Zipcode', 'Country', 'Timezone',
       'Airport_Code', 'Weather_Timestamp', 'Temperature(F)', 'Wind_Chill(F)',
       'Humidity(%)', 'Pressure(in)', 'Visibility(mi)', 'Wind_Direction',
       'Wind_Speed(mph)', 'Precipitation(in)', 'Weather_Condition', 'Amenity',
       'Bump', 'Crossing', 'Give_Way', 'Junction', 'No_Exit', 'Railway',
       'Roundabout', 'Station', 'Stop', 'Traffic_Calming', 'Traffic_Signal',
       'Turning_Loop', 'Sunrise_Sunset', 'Civil_Twilight', 'Nautical_Twilight',
       'Astronomical_Twilight']
'''
# purpose: find which factor contributes the most to the level of severity
# simple: n features run model with exclusing one at a time -> find accuracy
# shortage: many features have correlations, so removing one of the many will
# not lead to a big change to the accruacy of the model


class Ml_Model:

    def __init__(self, df, mask, features, labels):
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
        This function takes the features and label for target data
        can set up a cross-validation model and a train_test_split
        model. 
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
        This function takes in the features, labels, and ml
        model using a train-test split method on the processing
        the data. 
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
        this function takes in features, label, max_depth, and
        level of the cross_validation and run a cross validation
        on the given data and print out the results.
        '''
        standard_d = []
        mean = []
        accuracy_sc = []
        for depth in range(1, max_depth):
            tree_model = DecisionTreeClassifier(max_depth=depth)
            cv_scores = cross_val_score(  # cv_scores is an array consists of the scores
                tree_model, self._features, self._labels, cv=level, scoring='accuracy')  # estimates the expected accuracy of your model on out-of-training data
            mean.append(cv_scores.mean())
            standard_d.append(cv_scores.std())
            acc_score = tree_model.fit(
                self._features, self._labels).score(self._features, self._labels)
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
