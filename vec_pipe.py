# Author: Olivier Grisel <olivier.grisel@ensta.org>
#         Peter Prettenhofer <peter.prettenhofer@gmail.com>
#         Mathieu Blondel <mathieu@mblondel.org>
# License: BSD 3 clause
from pprint import pprint
from time import time
import logging
import warnings


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import pandas as pd
from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning


print(__doc__)

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

from warnings import filterwarnings
filterwarnings('ignore')

# #############################################################################
class data():
    def __init__(self):
        self.Data = pd.DataFrame()
        self.Target = pd.DataFrame()
books = [
    'data/The Elephant Vanishes_df.pkl']

for idx, book in enumerate(books):
    df = pd.read_pickle(book)

data = data()

data.Data, data.Target = df['text'], df['translator']


# #############################################################################
# Define a pipeline combining a text feature extractor with a simple
# classifier
pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', SGDClassifier(tol=1e-3)),
])

# uncommenting more parameters will give better exploring power but will
# increase processing time in a combinatorial way
parameters = {
    'vect__max_df': (0.1, 0.25, 0.5, 0.75, 1.0),
    'vect__max_features': (None, 10000),
    'vect__ngram_range': ((1, 2), (1,3)),  # unigrams or bigrams
    'tfidf__use_idf': (True, False),
    'tfidf__norm': ('l1','l2'),
    'clf__alpha': (0.000001, 0.0000001),
    'clf__penalty': ('l2','elasticnet'),
    'clf__max_iter': (80,100)
}

if __name__ == '__main__':
    # multiprocessing requires the fork to happen in a __main__ protected
    # block

    # find the best parameters for both the feature extraction and the
    # classifier
    grid_search = GridSearchCV(pipeline, parameters, cv=5,
                               n_jobs=-1, verbose=1)

    print("Performing grid search...")
    print("pipeline:", [name for name, _ in pipeline.steps])
    print("parameters:")
    pprint(parameters)
    t0 = time()
    # run block of code and catch warnings

    # execute code that will generate warnings
    grid_search.fit(data.Data, data.Target)
    print("done in %0.3fs" % (time() - t0))
    print()

    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))

