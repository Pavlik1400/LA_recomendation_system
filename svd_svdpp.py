from argparse import ArgumentParser
import numpy as np
import pandas as pd
from surprise import SVD, SVDpp, BaselineOnly, Dataset, Reader, accuracy
from surprise.model_selection import cross_validate, GridSearchCV, train_test_split
import json
import time
import os
from pprint import pprint
import itertools as it


ALGOS = [SVD, SVDpp]

# shit shit shit shit shit
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
# print(baseline_params_comb)

PARAMS = {
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
    }
}


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def run_and_save_grid(algo, params, data, algo_name, verbose=True):
    RES_FOLDER = "results"
    gs = GridSearchCV(algo, params, measures=['rmse'], cv=2,
                      refit=True, return_train_measures=True)
    gs.fit(data)
    results = gs.cv_results

    if verbose:
        pprint(f"Results: {results}")
    if not os.path.exists(RES_FOLDER):
        os.mkdir(RES_FOLDER)

    with open(RES_FOLDER + "/" + algo_name + ".json", "w") as result_file:
        json.dump(results, result_file, indent=4, cls=NumpyEncoder)


def main(args, algos=None, verbose=True, nth=30):
    if algos is None:
        algos = [SVD]
    before_read = time.time()
    if verbose:
        print("Reading data...")
    df = pd.read_csv(args.data, header=0)
    df = df[df.index % nth == 0]
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df[['Cust_Id', 'Movie_Id', 'Rating']], reader)
    after_read = time.time()
    if verbose:
        print(f"Read time: {'%.3f' % (after_read - before_read)}s")

    if verbose:
        print("Searching...")
    for algo in algos:
        algo_name = str(algo).split(".")[-1][:-2]  # that pervert
        if algo_name not in PARAMS:
            print(f"Unknown algrorithm: {algo_name}")
            continue
        print(f"    {algo_name}")
        run_and_save_grid(algo, PARAMS[algo_name], data,
                          algo_name=algo_name,
                          verbose=verbose)

    if verbose:
        print(f"Search time: {'%.3f' % (time.time() - after_read)}s")


if __name__ == '__main__':
    begin = time.time()
    parser = ArgumentParser()
    parser.add_argument("-d", "--data",
                        help="csv file with data. If provided folder then will search for comb_prep_i.csv",
                        required=True)

    args = parser.parse_args()
    main(args, algos=ALGOS, verbose=True, nth=10)
#
#
# params = {
#         'BaselineOnly': {
#             'method': 'als',
#             'n_epochs': 5,
#             'reg_u': 12,
#             'reg_i': 5,
#             'verbose': [True]
#         },
#         'SVD': {
#             'lr_all': [0.01],
#             'n_epochs': [20],
#             'n_factors': [10],
#             'reg_all': [0.0005],
#             'verbose': [True]
#         }
#     }
