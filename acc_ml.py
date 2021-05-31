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
from sklearn.externals import joblib
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
    sm_tree_depths = range(1, 25)
    sm_cv_scores_mean, sm_cv_scores_std, sm_accuracy_scores = run_cross_validation_on_trees(
        features, labels, sm_tree_depths)

    idx_max = sm_cv_scores_mean.argmax()

    sm_best_tree_depth = sm_tree_depths[idx_max]
    sm_best_tree_cv_score = sm_cv_scores_mean[idx_max]
    sm_best_tree_cv_score_std = sm_cv_scores_std[idx_max]
    print('The depth-{} tree achieves the best mean cross-validation accuracy {} +/- {}% on training dataset'.format(
        sm_best_tree_depth, round(sm_best_tree_cv_score*100, 5), round(sm_best_tree_cv_score_std*100, 5)))
    # set up the model with the best depth
    model = DecisionTreeClassifier(max_depth=sm_best_tree_depth)
    # predict(features, labels, model)
    # best_depth_graphing(features, labels)
    
    
def predict(features, labels, model):
    features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=0.2)
    model.fit(features_train, labels_train)
    # joblib.dump(model, 'car_accident.joblib') save the model into a file
    train_pred = model.predict(features_train)
    print('Train Accuracy:', accuracy_score(labels_train, train_pred))
    test_pred = model.predict(features_test)
    print('Test  Accuracy:', accuracy_score(labels_test, test_pred))
    
def best_depth_graphing(features, labels):
    '''
    this method uses the code from the ML pipeline
    file on ED
    '''
    features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=0.2, random_state=2)

    accuracies = []
    for i in range(1, 30):
        model = DecisionTreeClassifier(max_depth=i, random_state=1)
        model.fit(features_train, labels_train)
        pred_train = model.predict(features_train)
        train_acc = accuracy_score(labels_train, pred_train)
        pred_test = model.predict(features_test)
        test_acc = accuracy_score(labels_test, pred_test)
        accuracies.append({'max depth': i, 'train accuracy': train_acc,
                           'test accuracy': test_acc})
    accuracies = pd.DataFrame(accuracies)

def run_cross_validation_on_trees(X, y, tree_depths, cv=5, scoring='accuracy'):
    cv_scores_list = []
    cv_scores_std = []
    cv_scores_mean = []
    accuracy_scores = []
    for depth in tree_depths:
        tree_model = DecisionTreeClassifier(max_depth=depth)
        cv_scores = cross_val_score(tree_model, X, y, cv=cv, scoring=scoring)
        cv_scores_list.append(cv_scores)
        cv_scores_mean.append(cv_scores.mean())
        cv_scores_std.append(cv_scores.std())
        accuracy_scores.append(tree_model.fit(X, y).score(X, y))
    cv_scores_mean = np.array(cv_scores_mean)
    cv_scores_std = np.array(cv_scores_std)
    accuracy_scores = np.array(accuracy_scores)
    return cv_scores_mean, cv_scores_std, accuracy_scores
