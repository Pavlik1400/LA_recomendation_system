import itertools as it
from surprise.prediction_algorithms.matrix_factorization import SVD, SVDpp

from funkSVD import BaisedFunkSVD

MODELS = {BaisedFunkSVD.__name__: BaisedFunkSVD,
          SVD.__name__: SVD,
          SVDpp.__name__: SVDpp}


baseline_params = {
    'method': ['als', 'sgd'],  # Alternating Least Squares, Stochastic Gradient Descent
    'n_epochs': [5, 10],
    'reg_u': [10, 15],
    'reg_i': [5, 10],
}

allNames = sorted(baseline_params)
combinations = it.product(*(baseline_params[Name] for Name in allNames))
baseline_comb = list(combinations)

baseline_params_comb = {'bsl_options': []}
for comb in baseline_comb:
    baseline_params_comb['bsl_options'].append({})
    baseline_params_comb['bsl_options'][-1]['method'] = comb[0]
    baseline_params_comb['bsl_options'][-1]['n_epochs'] = comb[1]
    baseline_params_comb['bsl_options'][-1]['reg_u'] = comb[2]
    baseline_params_comb['bsl_options'][-1]['reg_i'] = comb[3]
baseline_params_comb['verbose'] = [True]


MODEL_PARAMS = {
    'BaselineOnly': baseline_params_comb,
    'SVD': {
        'lr_all': [0.005, 0.01, 0.02],
        'n_epochs': [5, 10],
        'n_factors': [10, 15, 20],
        'reg_all': [0.01, 0.02],
        'verbose': [True]
    },
    'SVDpp': {
        'lr_all': [0.005, 0.01, 0.03],
        'n_epochs': [5, 10],
        'n_factors': [10, 15],
        'reg_all': [0.01],
        'verbose': [True]
    },
    'BaisedFunkSVD': {
        'lr_all': [0.01],
        'n_epochs': [10],
        'n_factors': [10],
        'reg_all': [0.01],
        'verbose': [True]
    }
}

DATA_PARAMS = {
    'MoviesRecommendation.csv': {
        'labels': ['Cust_Id', 'Movie_Id', 'Rating'],
        'nth': 10,
    },
    'BooksRecommendation.csv': {
        'labels': ["ISBN", "User-ID", "Book-Rating"],
        'delimiter': ';'
    }

}