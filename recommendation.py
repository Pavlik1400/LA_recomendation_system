import numpy as np
import pandas as pd
import surprise


class FunkSVD(surprise.AlgoBase):
    # Randomly initializes two Matrices, Stochastic Gradient Descent to be able to optimize the best factorization for ratings.
    def __init__(self, lr_all, n_epoch, n_factors, **kwargs):
        # super(surprise.AlgoBase)
        super().__init__(**kwargs)
        self.alpha = lr_all  # learning rate for Stochastic Gradient Descent
        self.num_epochs = n_epoch
        self.num_factors = n_factors

    def fit(self, train):
        # randomly initialize user/item factors from a Gaussian
        P = np.random.normal(0, .1, (train.n_users, self.num_factors))
        Q = np.random.normal(0, .1, (train.n_items, self.num_factors))

        for epoch in range(self.num_epochs):

            for u, i, r_ui in train.all_ratings():
                residual = r_ui - np.dot(P[u], Q[i])
                temp = P[u, :]  # we want to update them at the same time, so we make a temporary variable.
                P[u, :] += self.alpha * residual * Q[i]
                Q[i, :] += self.alpha * residual * temp
        self.P = P
        self.Q = P

        self.trainset = train

    def estimate(self, u, i):

        if self.trainset.knows_user(u) and self.trainset.knows_item(i):
            # return scalar product of P[u] and Q[i]
            nanCheck = np.dot(self.P[u], self.Q[i])

            if np.isnan(nanCheck):
                return self.trainset.global_mean
            else:
                return np.dot(self.P[u, :], self.Q[i, :])
        else:  # if its not known we'll return the general average.
            return self.trainset.global_mean

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
        print("Median user/movie" % data.Cust_Id.value_counts().median())
        print(f"Rating range: {min_, max_}")
    reader = surprise.Reader(rating_scale=(min_, max_))
    return surprise.Dataset.load_from_df(data, reader)

def run_experiment(algorithm, params, data, n_cores=-1, n_split=10):
    gs = surprise.model_selection.GridSearchCV(algorithm,
                                               param_grid=params,
                                               measures=['rmse'],
                                               return_train_measures=True,
                                               cv=n_split,
                                               n_jobs=n_cores)
    gs.fit(data)
    return gs



if __name__ == '__main__':
   data = load_data("", 10)

   parameters = {'lr_all' :[0.005 ,0.01],
                 'n_epochs' :[5 ,10],
                 'n_factors' :[10 ,20]}

   results = run_experiment(FunkSVD, parameters, data)
   print(results)



