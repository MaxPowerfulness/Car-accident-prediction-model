import geopandas as gpd
import numpy
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
    counts = accidents.loc[:, 'Amenity':'Turning_Loop'].sum()
    print(counts)
    sns.catplot(data=counts, kind='bar')  # TODO: NOT FIXED; NEED TO FIGURE OUT WAY TO BAR PLOT A PD SERIES
    plt.xlabel('Points of Interest')
    plt.ylabel('Accident Count')
    plt.title('Number of U.S. Accidents between 2016 and 2020 that occurred in certain Points of Interest')
    plt.savefig('accident_POI.png', bbox_inches='tight')


def main():
    accident_data = pd.read_csv(ACCIDENT_FILE)
    us_accidents('Maps/USMap.json', accident_data)
    graph_accident_poi(accident_data)
    print('yeet')


if __name__ == '__main__':
    main()
