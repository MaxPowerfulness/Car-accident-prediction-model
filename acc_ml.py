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


class Ml_Model:

    def __init__(self, df, mask, features, labels):
        self._mask = mask
        self._features = features
        self._df = df
        self._labels = labels

    def run_model(self):
        if self._mask != '':
            self._df = self._df[eval(self._mask)]
        cols = self._features + self._labels
        self._df = self._df[cols]
        self._df = self._df.dropna()
        features = self._df[self._features]
        features = pd.get_dummies(features)
        labels = self._df[self._labels]
        self.set_up_model(features, labels)

    def set_up_model(self, features, label):
        '''
        This function takes the features and label for target data
        can set up a cross-validation model and a train_test_split
        model. 
        '''
        # try to get the best depth of the decision tree
        sc_mean, sc_std, acc_sc = self.cross_model(
            features, label, 20, 6)
        # get the index of the max element in a numpy
        id_of_max = sc_mean.argmax()
        ideal_depth = id_of_max + 1
        score = sc_mean[id_of_max]
        score_std = sc_std[id_of_max]
        accuracy = acc_sc[id_of_max]
        print('Cross_validation model results: ')
        print('''ideal depth: {}   \nmean accuracy : {}
            standard deviation : {}  \nacc_score : {}'''.format(
            ideal_depth, round(score, 6), round(score_std, 6), round(accuracy, 5)))
        # set up the model with the best depth
        model = DecisionTreeClassifier(max_depth=ideal_depth)
        self.predict(features, label, model)

    def predict(self, features, label, model):
        '''
        This function takes in the features, labels, and ml
        model using a train-test split method on the processing
        the data. 
        '''
        features_train, features_test, labels_train, labels_test = \
            train_test_split(features, label, test_size=0.2)
        model.fit(features_train, labels_train)
        train_pred = model.predict(features_train)
        print('train_test_split model results: ')
        print('Train Accuracy:', round(
            accuracy_score(labels_train, train_pred), 6))
        test_pred = model.predict(features_test)
        print('Test  Accuracy:', round(accuracy_score(labels_test, test_pred), 6))

    def cross_model(self, features, label, max_depth, level):
        '''
        this function takes in features, label, max_depth, and
        level of the cross_validation and run a cross validation
        on the given data and print out the results.
        '''
        accuracy = []
        standard_d = []
        mean = []
        accuracy_sc = []
        for depth in range(1, max_depth):
            tree_model = DecisionTreeClassifier(max_depth=depth)
            cv_scores = cross_val_score(
                tree_model, features, label, cv=level, scoring='accuracy')
            accuracy.append(cv_scores)
            mean.append(cv_scores.mean())
            standard_d.append(cv_scores.std())
            acc_score = tree_model.fit(features, label).score(features, label)
            accuracy_sc.append(acc_score)

        # convert them to numpy
        mean_np = np.array(mean)
        std_np = np.array(standard_d)
        acc_np = np.array(accuracy_sc)
        return (mean_np, std_np, acc_np)
