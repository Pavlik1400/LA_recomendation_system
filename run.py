import json
import os
import time
from argparse import ArgumentParser
from pprint import pprint

import surprise

from configs import MODEL_PARAMS, DATA_PARAMS, MODELS
from utils import NumpyEncoder, load_data


def run_and_save_grid(algorithm, params, data, verbose, n_cores=-1, n_split=2):
    RES_FOLDER = "results"
    gs = surprise.model_selection.GridSearchCV(algorithm,
                                               param_grid=params,
                                               measures=['rmse'],
                                               return_train_measures=True,
                                               cv=surprise.model_selection.KFold(n_splits=n_split, shuffle=True),
                                               n_jobs=n_cores)
    gs.fit(data)
    results = gs.cv_results

    if verbose:
        pprint(f"Results: {results}")
    if not os.path.exists(RES_FOLDER):
        os.mkdir(RES_FOLDER)

    with open(RES_FOLDER + "/" + algorithm.__name__ + ".json", "w") as result_file:
        json.dump(results, result_file, indent=4, cls=NumpyEncoder)


def main(args, verbose=True):
    before_read = time.time()
    if verbose:
        print("Reading data...")

    data = load_data(args.data, **DATA_PARAMS[args.data.replace('\\', '/').split('/')[-1]])
    after_read = time.time()
    if verbose:
        print(f"Read time: {'%.3f' % (after_read - before_read)}s")

    if verbose:
        print("Searching...")

    if args.algorithm not in MODELS:
        print(f"Unknown algrorithm: {args.algorithm}")

    print(f"    {args.algorithm}")
    run_and_save_grid(MODELS[args.algorithm], MODEL_PARAMS[args.algorithm], data, verbose=verbose)

    if verbose:
        print(f"Search time: {'%.3f' % (time.time() - after_read)}s")


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument("-d", "--data",
                        help="csv file with data",
                        required=True)

    parser.add_argument("-a", "--algorithm",
                        help="recommendation algorithm to be run",
                        required=True)
    parser.add_argument("-n", "--n_threads",
                        help="define how many threads to use",
                        required=False)

    args = parser.parse_args()
    main(args, verbose=True)
