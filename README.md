# Accident Analyzer
By Michael Christensen, Ian Lu, Sean Gombart

## Description
Accident Analyzer is a project that takes in a U.S. car accident dataset and compiles figures that highlight noticiable trends.
Main.py includes all of the visual graphs and plots for displaying these patterns.
The folder "/Datasets" is where the .csv files for the U.S. accidents and randomly truncated .csv belong.
The folder "/Maps" is the geometry outline that is used for plotting the U.S. map
The folder "/Images" contains all of the .pngs that are produced from the code.
The project also incorporates a Machine Learning algorithm that takes in the characteristics of the environment and tries to predict
the possible severity if an accident is to occur.

## "US-Accidents: A Countrywide Traffic Accident Dataset" Description
This dataset that is used contains U.S. car accidents spanning between 2016 to December of 2020.
In total, there are 2906610 accidents inside the dataset, each with descriptive data of when, where, and how the accident happened.
Each accident is classified with a severity level bewteen 1 to 4. This is analytically based on the impact that the accident had on traffic.
For mild accidents where traffic was not impacted, this is denoted as 1, while severe accidents that cause major delay are denoted as 4.

This .csv is too large to contain in the repo, so to download it into "/Datasets", you must go to https://www.kaggle.com/sobhanmoosavi/us-accidents. For testing purposes, the 
randomly truncated dataset "xaa.csv" is used instead of the entire U.S. dataset.

To read more about the dataset, go to https://smoosavi.org/datasets/us_accidents

## Python Installation
To set up the code, follow the CSE 163 software tutorial to download the exact modules with Anaconda:
https://courses.cs.washington.edu/courses/cse163/20wi/software.html
The following packages need to be installed into the project environment with Python 3.7 (listed as environment.yaml in tutorial):
 - descartes=1.1.0
 - flake8=3.7.9
 - geopandas=0.6.1
 - matplotlib=3.1.1
 - mock=3.0.5
 - numpy=1.17.4
 - pandas=0.25.3
 - python=3.7.5
 - requests=2.22.0
 - scikit-image=0.15.0
 - scikit-learn=0.21.3
 - scipy=1.3.1
 - seaborn=0.9.0

Next 

## Usage

### Machine Learning Portion:

Run the main function of the main.py
an object of Ml_Model will be instantiated.
The initializer will take four parameters accordingly
(dataset needed, filtering mask for the dataset, list of features needed to set up the models, list of labels needed to set up the models)
All the ML related program uses [randomly truncated dataset](https://raw.githubusercontent.com/MaxPowerfulness/Car-accident-prediction-model/main/Datasets/xaa.csv)

- For this ML project, the label should be 'Severity'
- For this ML project, the features should be one or many of the features below
- Temperature(F)', 'Wind_Chill(F)',
  'Humidity(%)', 'Pressure(in)', 'Visibility(mi)',
  'Wind_Direction', 'Wind_Speed(mph)', 'Precipitation(in)'

Uncomment the line below which will plot the importance of the features.
only run the line once with all the features above.

```python
current_model.random_forest_plot()
```

then comment this line out again to prevent later modification.

For clarification:

- The flow of function calls when you call `current_model.run_model()`should be: run_model() -> set_up_model() -> [cross_model() and then predict()] -> plot_partial_dep()
- The importance plot represents the expected importance of each feature regarding its impact on 'Severity', the label.
- Random_forest_plot can only take in numeric features
- Because the partial_dep only takes one feature, so when multiple features are passed in, only the first one will be used to create the partial_dep plot.
- The partial_dep plot represents the impact of different values of the one feature on the Severity of the car accidents.

