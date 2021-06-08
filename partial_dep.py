from numpy.lib.nanfunctions import nanmax
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble.partial_dependence import partial_dependence, plot_partial_dependence
from sklearn.preprocessing import Imputer
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier


class Partial_Dep:

    def __init__(self, df, mask, features, labels):
        self._mask = mask
        self._features = features
        self._df = df
        self._labels = labels
        self._graph_labels = features

    def set_up_model(self):
        if self._mask != '':
            self._df = self._df[eval(self._mask)]
        cols = self._features + self._labels
        self._df = self._df[cols].dropna()
        self._features = self._df[self._features]
        self._features = pd.get_dummies(self._features)
        my_im = SimpleImputer()
        self._features = my_im.fit_transform(self._features)
        self._labels = self._df[self._labels]
        self.run_model()

    def run_model(self):
        model = GradientBoostingClassifier()
        model.fit(self._features, self._labels)
        # print(model.classes_)
        plot = plot_partial_dependence(model,
                                       features=[0],
                                       feature_names=['Temperature(F)', ],
                                       label=1,
                                       X=self._features)
        print("the score of the partial model is: ",
              model.score(self._features, self._labels))
        plt.savefig('partial_dep.png', bbox_inches='tight')
