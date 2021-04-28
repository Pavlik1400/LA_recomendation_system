
import numpy as np
import pandas as pd
import surprise
import json

from recommendation import FunkSVD, BaisedFunkSVD

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def load_data(filename, nth=1, statistics=True):
    assert int(nth) == nth, "nth has to be integer"
    data = pd.read_csv(filename)
    data.drop_duplicates(inplace=True)
    if nth != 1:
        data = data[data.index % nth == 0]
    min_, max_ = data.Rating.min(), data.Rating.max()
    if statistics:
        print(f"Ratings: {data.shape[0]}")
        print(f"Users: {len(data.Cust_Id.unique())}")
        print(f"Movies: {len(data.Movie_Id.unique())}")
        print(f"Median user/movie {data.Cust_Id.value_counts().median()}")
        print(f"Rating range: {min_, max_}")
    # change the order
    data = data[['Cust_Id', 'Movie_Id', 'Rating']]
    reader = surprise.Reader(rating_scale=(min_, max_))
    return surprise.Dataset.load_from_df(data, reader)

def run_experiment(algorithm, params, data, n_cores=-1, n_split=2):
    gs = surprise.model_selection.GridSearchCV(algorithm,
                                               param_grid=params,
                                               measures=['rmse'],
                                               return_train_measures=True,
                                               cv=surprise.model_selection.KFold(n_splits=n_split, shuffle=True),
                                               n_jobs=n_cores)
    gs.fit(data)
    return gs



if __name__ == '__main__':
    data = load_data("comb_prep_1.csv", 10)


    parameters = {
            "lr_all": [0.01],
            "n_epoch": [20],
            "n_factors": [10],
            "reg_all": [0.02]
        }
    results = run_experiment(BaisedFunkSVD, parameters, data, n_split=2)
    print(results.cv_results)
    with open('result4.json', 'w') as fp:
       json.dump(results.cv_results, fp, indent=4, cls=NumpyEncoder)
