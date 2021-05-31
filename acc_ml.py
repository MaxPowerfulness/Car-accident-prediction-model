import geopandas as gpd
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from shapely.geometry import Point
from sklearn.tree.tree import DecisionTreeRegressor
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

# ask how to treat precipitation column, because it contains mostly 'nan' values


def weather_model(df):
    '''
    this function takes in a panda dataframe and use sklearn
    to prodict the severity of the car accidents based on natural
    features includes Temperature, Wind Chill, Humidity, Pressure, Visibility,
    Wind direction, Wind speed, weather condition
    '''
    df = df[['Severity', 'Temperature(F)', 'Wind_Chill(F)',
             'Humidity(%)', 'Pressure(in)', 'Visibility(mi)']]
    df = df.dropna()
    features = df[['Temperature(F)', 'Wind_Chill(F)',
                   'Humidity(%)', 'Pressure(in)', 'Visibility(mi)']]
    features = pd.get_dummies(features)
    labels = df[['Severity']]

    # try to get the best depth of the decision tree
    scores_mean, sm_cv_scores_std, accuracy_sc = run_cross_validation_on_trees(
        features, labels, 20, 6)
    # get the index of the max element in a numpy
    id_of_max = scores_mean.argmax()
    ideal_depth = sm_tree_depths[id_of_max]
    score = scores_mean[id_of_max]
    score_std = sm_cv_scores_std[id_of_max]
    accuracy = accuracy_sc[id_of_max]
    print('''depth: {}   \n mean accuracy : {}  \n standard deviation : {}  \n  acc_score : {}'''.format(
        ideal_depth, round(score, 6), round(score_std, 6), round(accuracy, 5)))
    # set up the model with the best depth
    model = DecisionTreeClassifier(max_depth=ideal_depth)
    # predict(features, labels, model)


def predict(features, labels, model):
    features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=0.2)
    model.fit(features_train, labels_train)
    train_pred = model.predict(features_train)
    print('Train Accuracy:', accuracy_score(labels_train, train_pred))
    test_pred = model.predict(features_test)
    print('Test  Accuracy:', accuracy_score(labels_test, test_pred))


def run_cross_validation_on_trees(X, y, max_depth, level):
    accuracy = []
    cv_scores_std = []
    cv_scores_mean = []
    accuracy_scores = []
    for depth in range(1, max_depth):
        tree_model = DecisionTreeClassifier(max_depth=depth)
        cv_scores = cross_val_score(
            tree_model, X, y, cv=level, scoring='accuracy')
        accuracy.append(cv_scores)
        cv_scores_mean.append(cv_scores.mean())
        cv_scores_std.append(cv_scores.std())
        accuracy_scores.append(tree_model.fit(X, y).score(X, y))
    cv_scores_mean = np.array(cv_scores_mean)
    cv_scores_std = np.array(cv_scores_std)
    accuracy_scores = np.array(accuracy_scores)
    return cv_scores_mean, cv_scores_std, accuracy_scores
