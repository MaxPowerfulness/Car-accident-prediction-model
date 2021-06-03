import geopandas as gpd
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from shapely.geometry import Point
from acc_ml import Ml_Model

ACCIDENT_FILE = 'Datasets/US_Accidents_Dec20_Updated.csv'


sns.set()


def us_accidents(us_map_file, accidents):
    """
    Takes in a US map file and an accident CSV file and plots the occurances of all accidents on a map of the United States.
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
    accidents.plot(ax=axis, color='red', markersize=1)

    # Save figure
    plt.savefig('US_Accidents.png', bbox_inches='tight')


def graph_accident_poi(accidents):
    """
    Takes in a pandas df of U.S. accidents and creates a bar graph of the number of accidents reported in each
    Point of Interest (POI). Saves the bar graph as "US_Accidents.png".

    :param accidents: the read-in pandas df of the U.S. accident CSV.
    """
    counts = accidents.loc[:, 'Amenity':'Turning_Loop'].sum().sort_values(ascending=True)
    x_labels = counts.index.str.replace('_', ' ')  # Remove underscores in column names
    sns.barplot(x=x_labels, y=counts.values)
    plt.xticks(rotation=90)
    plt.yticks(np.arange(0, 500000, 50000))
    plt.xlabel('POI')
    plt.ylabel('Accident Count')
    plt.title('Number of U.S. Accidents Between 2016 and 2020 That Occurred in Certain Point of Interests (POI)')
    plt.savefig('accident_POI.png', bbox_inches='tight')
    plt.clf()


def graph_by_hour(accidents):
    """
    Takes in a pandas df of U.S. accidents and creates a histogram of the cumulative number of accidents reported
    during each hour of the day. Saves the histogram as "accidents_by_hour.png".

    :param accidents: the read-in pandas df of the U.S. accident CSV.
    """
    accidents = accidents.loc[:, 'Start_Time']
    accidents = accidents.dt.hour
    sns.histplot(data=accidents, x=accidents.values, discrete=True, bins=25, kde=True)
    plt.xticks(range(0, 25, 4))
    plt.xlabel('Hour of Day')
    plt.ylabel('Accident Count')
    plt.title('Cumulative Number of Accidents That Occurred During Each Hour')
    plt.savefig('accidents_by_hour.png', bbox_inches='tight')
    plt.clf()


def graph_by_week(accidents):
    """
     Takes in a pandas df of U.S. accidents and creates a histogram of the cumulative number of accidents reported
     during each week of the year. Saves the histogram as "accidents_by_week.png".

    :param accidents: the read-in pandas df of the U.S. accident CSV.
    """
    accidents = accidents.loc[:, 'Start_Time']
    accidents = accidents.dt.weekofyear
    sns.histplot(data=accidents, x=accidents.values, discrete=True, bins=52, kde=True)
    plt.xticks(range(0, 53, 4))
    plt.xlabel('Week')
    plt.ylabel('Accident Count')
    plt.title('Cumulative Number of Accidents That Occurred During Each Week')
    plt.savefig('accidents_by_week.png', bbox_inches='tight')
    plt.clf()

def graph_by_month(accidents):
    """
     Takes in a pandas df of U.S. accidents and creates a histogram of the cumulative number of accidents reported
     by month of the year. Saves the histogram as "accidents_by_month.png".

    :param accidents: the read-in pandas df of the U.S. accident CSV.
    """
    accidents = accidents.loc[:, 'Start_Time']
    accidents = accidents.dt.month
    sns.histplot(data=accidents, x=accidents.values, discrete=True, bins=12, kde=True)
    plt.xticks(range(13))
    plt.xlabel('Month')
    plt.ylabel('Accident Count')
    plt.title('Cumulative Number of Accidents That Occurred During Each Month')
    plt.savefig('accidents_by_month.png', bbox_inches='tight')
    plt.clf()


def main():
    accident_data = pd.read_csv(ACCIDENT_FILE)
    # Change Start_Times to datetime objects
    accident_data['Start_Time'] = pd.to_datetime(accident_data['Start_Time'])
    us_accidents('Maps/USMap.json', accident_data)
    graph_accident_poi(accident_data)
    graph_by_hour(accident_data)
    graph_by_week(accident_data)
    graph_by_month(accident_data)
    current_model = Ml_Model(
        accident_data, '', ['Bump', 'Crossing', 'Stop'], ['Severity'])
    current_model.run_model()
    print('yeet')


if __name__ == '__main__':
    main()
