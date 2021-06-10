# Accident Analyzer
## Description

## Car Accident Dataset Description

## Installation Instructions
To set up the code, follow the CSE 163 software tutorial to download the exact modules with Anaconda:
https://courses.cs.washington.edu/courses/cse163/20wi/software.html
please refer to this [website](https://courses.cs.washington.edu/courses/cse163/20wi/software.html) for basic installation details
Use the package manager [pip](https://pip.pypa.io/en/stable/) to install
the most recent [sklearn](https://scikit-learn.org/stable/install.html)

## Usage

1. Download Anaconda
2. Create new environment (In Python 3.7) with the following modules included in the environment.yaml:
Here is the list of packages that need to be installed:
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
3. 
### Machine learning Portion:

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

