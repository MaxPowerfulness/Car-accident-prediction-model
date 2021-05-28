import geopandas as gpd
import numpy
import pandas as pd
import requests
import seaborn
import sklearn
import skimage
import matplotlib.pyplot as plt
from shapely.geometry import Point
ACCIDENT_FILE = 'Datasets/US_Accidents_Dec20_Updated.csv'

def us_accidents(us_map_file, accidents):
    """
    Takes in a US map file and an accident CSV file and plots the occurances of all accidents on a map of the United States.
    Creates a new .png of plotted accidents called "US_Accidents.png".

    :param us_map_file: the .json that outlines the geometry shape of the U.S.
    :param accident_file: the CSV file of U.S. accidents to plot.
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


def main():
    accident_data = pd.read_csv(ACCIDENT_FILE)
    us_accidents('Maps/USMap.json', accident_data)
    print('yeet')
   

if __name__ == '__main__':
    main()
