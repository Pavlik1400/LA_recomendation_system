import json
import pandas as pd
import surprise
import numpy as np




class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)



def load_data(filename, labels, nth=1, verbose=True, delimiter=','):
    assert int(nth) == nth, "nth has to be integer"
    data = pd.read_csv(filename, delimiter=delimiter)
    data.drop_duplicates(inplace=True)
    if nth != 1:
        data = data[data.index % nth == 0]
    data = data[labels]
    data[labels[-1]] = data[labels[-1]].astype(float)
    min_, max_ = data[labels[-1]].min(), data[labels[-1]].max()
    if verbose:
        print(f"Ratings: {data.shape[0]}")
        print(f"Users: {len(data[labels[0]].unique())}")
        print(f"Products: {len(data[labels[1]].unique())}")
        print(f"Median user/product {data[labels[0]].value_counts().median()}")
        print(f"Rating range: {min_, max_}")

    reader = surprise.Reader(rating_scale=(min_, max_))
    return surprise.Dataset.load_from_df(data, reader)