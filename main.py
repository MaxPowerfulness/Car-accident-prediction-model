"""
Sean Gombart, Ian Lu, and Michael Christensen
CSE 163 Accident Analyzer
main.py includes one method that plots all of the accidents on a U.S. map while
the rest graph additional visuals that show some general trends when accidents
occur.
"""
import geopandas as gpd
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
from shapely.geometry import Point
from acc_ml import Ml_Model

ACCIDENT_FILE = 'Datasets/US_Accidents_Dec20_Updated.csv'


sns.set()


def us_accidents(us_map_file, accidents):
    """
    Takes in a US map file and an accident CSV file and plots the occurrences
    of all accidents on a map of the United States.
    Creates a new .png of plotted accidents called "US_Accidents.png".
    :param us_map_file: the .json that outlines the geometry shape of the U.S.
    :param accidents: the pandas df of U.S. accidents
    """
    fig, axis = plt.subplots(1, figsize=(10, 10))

    # Reading in US map
    states = gpd.read_file(us_map_file)
    states = states[(states['NAME'] != 'Alaska') &
                    (states['NAME'] != 'Hawaii')]

    # Reading in accident CSV
    coordinates = zip(accidents['Start_Lng'], accidents['Start_Lat'])
    accidents['coordinates'] = [Point(lat, lon) for lat, lon in coordinates]
    accidents = gpd.GeoDataFrame(accidents, geometry='coordinates')

    # Plot map of US and accidents
    states.plot(ax=axis)
    accidents.plot(ax=axis, data=accidents, column='Severity',
                   markersize=1, legend=True)
    plt.title('U.S. Accidents between 2016-2020 (By Severity)')

    # Save figure
    plt.savefig('Images/US_Accidents.png', bbox_inches='tight')
    plt.clf()


def total_severity_for_all_accidents(accident_df):
    """
    Takes in a pandas dataframe of U.S. accidents and creates a pie chart
    detailing the percentage of the severity of all accidents
    """
    # Grouping by Severity and determing counts for each level of severity
    accident_df = accident_df.groupby('Severity').size()

    # Plotting
    fig, axis = plt.subplots(1)
    accident_df.plot.pie(title='Percentage of Accident Severity',
                         autopct='%1.1f%%')
    plt.ylabel('Severity')
    plt.savefig('Images/Accident_Severity_Percentage_All.png')


def precipitation_severity_correlation(accident_df):
    """
    Takes in a pandas dataframe of U.S. accidents and creates a stacked
    barchart showing the correlation between precipitation levels in inches
    and accident severity
    """
    # Getting rid of all NA values present in the Severity of Precipitation(in)
    # columns
    accident_df = accident_df.loc[:, ['Severity', 'Precipitation(in)']]
    accident_df = accident_df.dropna()
    # Generalizing data
    accident_df['Precipitation'] = (np.where(accident_df['Precipitation(in)']
                                    <= 0.5, '0 - 0.5',
                                    accident_df['Precipitation(in)']))
    accident_df['Precipitation'] = (np.where(accident_df['Precipitation(in)']
                                    > 0.5, '0.51 - 1',
                                    accident_df['Precipitation']))
    accident_df['Precipitation'] = (np.where(accident_df['Precipitation(in)']
                                    > 1, '1.1 - 2',
                                    accident_df['Precipitation']))
    accident_df['Precipitation'] = (np.where(accident_df['Precipitation(in)']
                                    > 2, '2+', accident_df['Precipitation']))
    # Grouping data and converting it into a dataframe
    accident_df['Counter'] = 1
    precipitation = (accident_df.groupby(['Precipitation', 'Severity'])
                     ['Counter'].sum())
    precipitation = precipitation.to_frame()
    df = precipitation.reset_index(level=['Precipitation', 'Severity'])
    # Creating percentages of car accident severity for each precipitation group
    zero_to_point5 = df.loc[df['Precipitation'] == '0 - 0.5', :]
    point5_to_one = df.loc[df['Precipitation'] == '0.51 - 1', :]
    one_to_two = df.loc[df['Precipitation'] == '1.1 - 2', :]
    two_above = df.loc[df['Precipitation'] == '2+', :]
    zero_to_point5['percentage'] = ((zero_to_point5['Counter'] / zero_to_point5
                                    ['Counter'].sum()) * 100)
    point5_to_one['percentage'] = ((point5_to_one['Counter'] / point5_to_one
                                   ['Counter'].sum()) * 100)
    one_to_two['percentage'] = ((one_to_two['Counter'] /
                                 one_to_two['Counter'].sum()) * 100)
    two_above['percentage'] = ((two_above['Counter'] /
                                two_above['Counter'].sum()) * 100)
    frames = [zero_to_point5, point5_to_one, one_to_two, two_above]
    result = pd.concat(frames)
    # Plotting
    fig, axs = plt.subplots(1)
    sns.histplot(x='Precipitation', weights='percentage', hue='Severity',
                 multiple='stack', data=result, shrink=.8)
    plt.title('Percentage of Accident Severity for Precipitation Levels')
    plt.ylabel('Percentage')
    plt.xlabel('Precipitation Level (in)')
    plt.savefig('Images/severity_precipitation.png')


def visibility_severity_correlation(accident_df):
    """
    Takes in a pandas dataframe of U.S. accidents and creates a stacked
    barchart showing the correlation between visibility levels in miles and
    accident severity
    """
    # Generalizing data
    accident_df = accident_df.loc[accident_df['Visibility(mi)'] <= 10, :]
    accident_df['Visibility'] = (np.where(accident_df['Visibility(mi)'] <= 2,
                                 '0 - 2', accident_df['Visibility(mi)']))
    accident_df['Visibility'] = (np.where(accident_df['Visibility(mi)'] > 2,
                                 '2.1 - 4', accident_df['Visibility']))
    accident_df['Visibility'] = (np.where(accident_df['Visibility(mi)'] > 4,
                                 '4.1 - 6', accident_df['Visibility']))
    accident_df['Visibility'] = (np.where(accident_df['Visibility(mi)'] > 6,
                                 '6.1 - 8', accident_df['Visibility']))
    accident_df['Visibility'] = (np.where(accident_df['Visibility(mi)'] > 8,
                                 '8.1 - 10', accident_df['Visibility']))
    # Grouping data and converting it into a dataframe
    accident_df['Counter'] = 1
    visibility = (accident_df.groupby(['Visibility', 'Severity'])
                  ['Counter'].sum())
    visibility = visibility.to_frame()
    visibility_df = visibility.reset_index(level=['Visibility', 'Severity'])
    # Creating percentages of car accident severity for each visibility group
    zero_two = visibility_df.loc[visibility_df['Visibility'] == '0 - 2', :]
    two_four = visibility_df.loc[visibility_df['Visibility'] == '2.1 - 4', :]
    four_six = visibility_df.loc[visibility_df['Visibility'] == '4.1 - 6', :]
    six_eight = visibility_df.loc[visibility_df['Visibility'] == '6.1 - 8', :]
    eight_ten = visibility_df.loc[visibility_df['Visibility'] == '8.1 - 10', :]
    zero_two['percentage'] = ((zero_two['Counter'] / zero_two['Counter'].sum())
                              * 100)
    two_four['percentage'] = ((two_four['Counter'] / two_four['Counter'].sum())
                              * 100)
    four_six['percentage'] = ((four_six['Counter'] / four_six['Counter'].sum())
                              * 100)
    six_eight['percentage'] = ((six_eight['Counter'] /
                                six_eight['Counter'].sum()) * 100)
    eight_ten['percentage'] = ((eight_ten['Counter'] /
                                eight_ten['Counter'].sum()) * 100)
    frames = [zero_two, two_four, four_six, six_eight, eight_ten]
    result = pd.concat(frames)
    # Plotting
    fig, axs = plt.subplots(1)
    sns.histplot(x='Visibility', weights='percentage', hue='Severity',
                 multiple='stack', data=result, shrink=.8)
    plt.title('Percentage of Accident Severity for Visibility Levels')
    plt.ylabel('Percentage')
    plt.xlabel('Visibility (mi)')
    plt.savefig('Images/severity_visibility.png')


def weather_condition_severity_correlation(accident_df):
    """
    Takes in a pandas dataframe of U.S. accidents and creates a stacked
    barchart showing the correlation between weather conditions and accident
    severity
    """
    # Cleaning and filtering data
    accident_df = (accident_df.loc[:, ['Severity',
                   'Weather_Condition']].dropna())

    # Generalizing weather conditions
    (accident_df.loc[accident_df['Weather_Condition'].str.contains('Cloudy',
     flags=re.IGNORECASE), 'normalized_wc']) = 'Cloudy'
    (accident_df.loc[accident_df['Weather_Condition'].str.contains(
     'Fog|Haze|Smoke', flags=re.IGNORECASE),
     'normalized_wc']) = 'Reduced Visibility'
    (accident_df.loc[accident_df['Weather_Condition'].str.contains(
     'Rain|Drizzle|Thunder', flags=re.IGNORECASE),
     'normalized_wc']) = 'Rain Conditions'
    (accident_df.loc[accident_df['Weather_Condition'].str.contains(
     'Snow|Ice|Wintry Mix', flags=re.IGNORECASE),
     'normalized_wc']) = 'Snow Conditions'
    (accident_df.loc[accident_df['Weather_Condition'].str.contains('Fair',
     flags=re.IGNORECASE), 'normalized_wc']) = 'Fair'
    (accident_df.loc[accident_df['Weather_Condition'].str.contains('Clear',
     flags=re.IGNORECASE), 'normalized_wc']) = 'Clear'
    (accident_df.loc[accident_df['Weather_Condition'].str.contains('Overcast',
     flags=re.IGNORECASE), 'normalized_wc']) = 'Overcast'
    generalized_df = accident_df.dropna()
    # Grouping data and converting it into a dataframe
    generalized_df['Counter'] = 1
    weather_conditions = (generalized_df.groupby(['normalized_wc', 'Severity'])
                          ['Counter'].sum())
    weather_conditions = weather_conditions.to_frame()
    df = weather_conditions.reset_index(level=['normalized_wc', 'Severity'])
    # Creating percentages of car accident severity for each weather condition
    # group
    cloudy = df.loc[df['normalized_wc'] == 'Cloudy', :]
    reduced_visibility = df.loc[df['normalized_wc'] == 'Reduced Visibility', :]
    rain = df.loc[df['normalized_wc'] == 'Rain Conditions', :]
    snow = df.loc[df['normalized_wc'] == 'Snow Conditions', :]
    fair = df.loc[df['normalized_wc'] == 'Fair', :]
    clear = df.loc[df['normalized_wc'] == 'Clear', :]
    overcast = df.loc[df['normalized_wc'] == 'Overcast', :]
    cloudy['percentage'] = (cloudy['Counter'] / cloudy['Counter'].sum()) * 100
    reduced_visibility['percentage'] = ((reduced_visibility['Counter'] /
                                         reduced_visibility['Counter'].sum())
                                        * 100)
    rain['percentage'] = (rain['Counter'] / rain['Counter'].sum()) * 100
    snow['percentage'] = (snow['Counter'] / snow['Counter'].sum()) * 100
    fair['percentage'] = (fair['Counter'] / fair['Counter'].sum()) * 100
    clear['percentage'] = (clear['Counter'] / clear['Counter'].sum()) * 100
    overcast['percentage'] = ((overcast['Counter'] / overcast['Counter'].sum())
                              * 100)
    frames = [fair, clear, overcast, cloudy, rain, snow, reduced_visibility]
    result = pd.concat(frames)
    # Plotting
    fig, axs = plt.subplots(1, figsize=(10, 10))
    sns.histplot(x='normalized_wc', weights='percentage', hue='Severity',
                 multiple='stack', data=result, shrink=.8)
    plt.xticks(rotation=-45)
    plt.title('Percentage of Accident Severity for Weather Conditions')
    plt.ylabel('Percentage')
    plt.xlabel('Weather Condition')
    plt.savefig('Images/severity_weather_condition.png')


def graph_accident_poi(accidents):
    """
    Takes in a pandas df of U.S. accidents and creates a bar graph of the number
    of accidents reported in each Point of Interest (POI).
    Saves the bar graph as "US_Accidents.png".
    :param accidents: the read-in pandas df of the U.S. accident CSV.
    """
    counts = accidents.loc[:, 'Amenity':'Turning_Loop'].sum() \
        .sort_values(ascending=True)
    x_labels = counts.index.str.replace('_', ' ')  # Remove underscores
    sns.barplot(x=x_labels, y=counts.values)
    plt.xticks(rotation=90)
    plt.yticks(np.arange(0, 500000, 50000))
    plt.xlabel('POI')
    plt.ylabel('Accident Count')
    plt.title('Number of U.S. Accidents Between 2016 and 2020'
              'That Occurred in Certain Point of Interests (POI)')
    plt.savefig('Images/accident_POI.png', bbox_inches='tight')
    plt.clf()


def graph_by_hour(accidents):
    """
    Takes in a pandas df of U.S. accidents and creates a histogram of the
    cumulative number of accidents reported during each hour of the day.
    Saves the histogram as "accidents_by_hour.png".
    :param accidents: the read-in pandas df of the U.S. accident CSV.
    """
    accidents = accidents.loc[:, 'Start_Time']
    accidents = accidents.dt.hour
    sns.histplot(data=accidents, x=accidents.values, discrete=True, bins=24)
    plt.xticks(range(0, 24, 4))
    plt.xlabel('Hour of Day')
    plt.ylabel('Accident Count')
    plt.title('Cumulative Number of Accidents That Occurred During Each Hour')
    plt.savefig('Images/accidents_by_hour.png', bbox_inches='tight')
    plt.clf()


def graph_by_week(accidents):
    """
     Takes in a pandas df of U.S. accidents and creates a histogram of the
     cumulative number of accidents reported during each week of the year.
     Saves the histogram as "accidents_by_week.png".
    :param accidents: the read-in pandas df of the U.S. accident CSV.
    """
    accidents = accidents.loc[:, 'Start_Time']
    accidents = accidents.dt.weekofyear
    sns.histplot(data=accidents, x=accidents.values,
                 discrete=True, bins=52, kde=True)
    plt.xticks(range(1, 53, 4))
    plt.xlabel('Week')
    plt.ylabel('Accident Count')
    plt.title('Cumulative Number of Accidents That Occurred During Each Week')
    plt.savefig('Images/accidents_by_week.png', bbox_inches='tight')
    plt.clf()


def graph_by_month(accidents):
    """
     Takes in a pandas df of U.S. accidents and creates a histogram of the
     cumulative number of accidents reported by month of the year.
     Saves the histogram as "accidents_by_month.png".
    :param accidents: the read-in pandas df of the U.S. accident CSV.
    """
    accidents = accidents.loc[:, 'Start_Time']
    accidents = accidents.dt.month
    sns.histplot(data=accidents, x=accidents.values, discrete=True, bins=12)
    plt.xticks(range(1, 13))
    plt.xlabel('Month')
    plt.ylabel('Accident Count')
    plt.title('Cumulative Number of Accidents That Occurred During Each Month')
    plt.savefig('Images/accidents_by_month.png', bbox_inches='tight')
    plt.clf()


def graph_by_year(accidents):
    """
     Takes in a pandas df of U.S. accidents and creates a histogram of the
     cumulative number of accidents reported by month of the year.
     Saves the histogram as "accidents_by_year.png".
    :param accidents: the read-in pandas df of the U.S. accident CSV.
    """
    accidents = accidents.loc[:, 'Start_Time']
    accidents = accidents.dt.year
    sns.histplot(data=accidents, x=accidents.values, discrete=True)
    plt.xlabel('Year')
    plt.ylabel('Accident Count')
    plt.title('Cumulative Number of Accidents That Occurred During Each Year')
    plt.savefig('Images/accidents_by_year.png', bbox_inches='tight')
    plt.clf()

    
def print_result(dict):
    '''
    this is a helper function that takes in a dictionary
    and print out the statistical data of ml model
    '''
    print()
    for x, y in dict.items():
        print(x, y)
    print()

def main():
    accident_data = pd.read_csv(ACCIDENT_FILE)
    # Change Start_Times to datetime objects
    accident_data['Start_Time'] = pd.to_datetime(accident_data['Start_Time'])
    # Create each figure/model
    us_accidents('Maps/USMap.json', accident_data)
    total_severity_for_all_accidents(accident_data)
    precipitation_severity_correlation(accident_data)
    visibility_severity_correlation(accident_data)
    weather_condition_severity_correlation(accident_data)
    graph_accident_poi(accident_data)
    graph_by_hour(accident_data)
    graph_by_week(accident_data)
    graph_by_month(accident_data)
    current_model = Ml_Model(accident_data, '',
                             ['Bump', 'Crossing', 'Stop'], ['Severity'])
    current_model.run_model()


if __name__ == '__main__':
    main()
