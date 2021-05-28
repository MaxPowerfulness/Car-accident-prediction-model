import geopandas as gpd
import numpy as np
import pandas as pd
import requests
import seaborn as sns
import sklearn
import skimage
import matplotlib.pyplot as plt
from shapely.geometry import Point


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
    '''
    Takes in a pandas df of U.S. accidents and creates a bar graph of the number of accidents reported in each
    Point of Interest (POI). Saves the bar graph as "US_Accidents.png".

    :param accidents: the read-in pandas df of the U.S. accident CSV.
    '''
    counts = accidents.loc[:, 'Amenity':'Turning_Loop'].sum().sort_values(ascending=True)
    x_labels = counts.index.str.replace('_', ' ')  # Remove underscores in column names
    sns.barplot(x=x_labels, y=counts.values)
    plt.xticks(rotation=90)
    plt.yticks(np.arange(0, 500000, 50000))
    plt.xlabel('POI')
    plt.ylabel('Accident Count')
    plt.title('Number of U.S. Accidents between 2016 and 2020 that occurred in certain Points of Interest (POI)')
    plt.savefig('accident_POI.png', bbox_inches='tight')


def main():
    accident_data = pd.read_csv(ACCIDENT_FILE)
    us_accidents('Maps/USMap.json', accident_data)
    graph_accident_poi(accident_data)
    print('yeet')


if __name__ == '__main__':
    main()
