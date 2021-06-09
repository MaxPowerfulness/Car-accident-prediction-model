'''
Peiyu(Ian) Lu
this program is a Partial_Dep class for plotting partial
dependency plots using plot_partial_dependence(). The model and
plot should take in one feature max to product 2d graph.
'''
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble.partial_dependence import plot_partial_dependence
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt


class Partial_Dep:

    def __init__(self, df, mask, features, labels):
        '''
        this function is the initializer of the Partial_Dep class,
        which takes in a pandas dataframe, a filter for the dataframe
        and features and labels to consider. The default value of statistical
        fields is -1
        '''
        self._mask = mask
        self._features = features
        self._df = df
        self._labels = labels
        self._graph_labels = features
        self._partial_score = -1

    def set_up_model(self):
        '''
        this function finds the final features and labels to train
        the data.
        '''
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
        '''
        this function creates a GradientBoostingClassifier
        and plot a partial dependent. After careful test trials,
        the model depth will be 10. Data will be printed in the console
        '''
        model = GradientBoostingClassifier(max_depth=10)
        model.fit(self._features, self._labels)
        plot = plot_partial_dependence(model,  # plot will be saved by savefig
                                       features=[0],
                                       feature_names=self._features,
                                       label=1,
                                       X=self._features)
        score = model.score(self._features, self._labels)
        print('Gradient Boosting Classifier score:   ', score)
        plt.savefig('partial_dep.png', bbox_inches='tight')

    def get_data(self):
        '''
        this function return the statistical data of the ml model
        '''
        return (self._partial_score)
