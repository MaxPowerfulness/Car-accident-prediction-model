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
# purpose: find which factor contributes the most to the level of severity


def road_condition_model(df):
    '''
    this function takes in a panda dataframe and use sklearn to
    predict the severity of the car accidents based on the timing factors
    include(maybe not all) Amenity,Bump,Crossing,Give_Way,Junction,No_Exit,Railway,
    Roundabout,Station,Stop,Traffic_Calming,Traffic_Signal,Turning_Loop
    '''
    df = df[['Severity', 'Bump', 'Crossing']]
    df = df.dropna()
    features = df[['Bump', 'Crossing']]
    features = pd.get_dummies(features)
    label = df[['Severity']]
    set_up_model(features, label)


def timing_model(df):
    '''
    this function takes in a panda dataframe and use sklearn to
    predict the severity of the car accidents based on the timing factors
    include Sunrise_Sunset, Civil_Twilight, Nautical_Twilight, Astronomical_Twilight
    '''
    df = df[['Severity', 'Sunrise_Sunset', 'Civil_Twilight',
             'Nautical_Twilight', 'Astronomical_Twilight']]
    df = df.dropna()
    features = df[['Sunrise_Sunset', 'Civil_Twilight',
                   'Nautical_Twilight', 'Astronomical_Twilight']]
    features = pd.get_dummies(features)
    label = df[['Severity']]

    set_up_model(features, label)


def weather_model(df):
    '''
    this function takes in a panda dataframe and use sklearn
    to predict the severity of the car accidents based on natural
    features includes Temperature, Wind Chill, Humidity, Pressure, Visibility,
    Wind direction, Wind speed, weather condition
    '''
    df = df[['Severity', 'Temperature(F)', 'Wind_Chill(F)',
             'Humidity(%)', 'Pressure(in)', 'Visibility(mi)']]
    df = df.dropna()
    features = df[['Temperature(F)', 'Wind_Chill(F)',
                   'Humidity(%)', 'Pressure(in)', 'Visibility(mi)']]
    features = pd.get_dummies(features)
    label = df[['Severity']]

    set_up_model(features, label)


def set_up_model(features, label):
    '''
    This function takes the features and label for target data
    can set up a cross-validation model and a train_test_split
    model. 
    '''
    # try to get the best depth of the decision tree
    scores_mean, sm_cv_scores_std, accuracy_sc = cross_model(
        features, label, 20, 6)
    # get the index of the max element in a numpy
    id_of_max = scores_mean.argmax()
    ideal_depth = id_of_max + 1
    score = scores_mean[id_of_max]
    score_std = sm_cv_scores_std[id_of_max]
    accuracy = accuracy_sc[id_of_max]
    print('''depth: {}   \n mean accuracy : {}  \n standard deviation : {}  \n  acc_score : {}'''.format(
        ideal_depth, round(score, 6), round(score_std, 6), round(accuracy, 5)))
    # set up the model with the best depth
    model = DecisionTreeClassifier(max_depth=ideal_depth)
    predict(features, label, model)


def predict(features, label, model):
    '''
    This function takes in the features, labels, and ml
    model using a train-test split method on the processing
    the data. 
    '''

    features_train, features_test, labels_train, labels_test = \
        train_test_split(features, label, test_size=0.2)
    model.fit(features_train, labels_train)
    train_pred = model.predict(features_train)
    print('Train Accuracy:', accuracy_score(labels_train, train_pred))
    test_pred = model.predict(features_test)
    print('Test  Accuracy:', accuracy_score(labels_test, test_pred))


def cross_model(features, label, max_depth, level):
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
