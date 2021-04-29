import numpy as np
import surprise

class FunkSVD(surprise.AlgoBase):
    # Randomly initializes two Matrices, Stochastic Gradient Descent to be able to optimize the best factorization for ratings.
    def __init__(self, lr_all, reg_all, n_epoch, n_factors, **kwargs):
        # super(surprise.AlgoBase)
        super().__init__(**kwargs)
        self.alpha = lr_all  # learning rate for Stochastic Gradient Descent
        self.num_epochs = n_epoch
        self.num_factors = n_factors
        self.reg_all = reg_all

    def fit(self, train):
        # randomly initialize user/item factors from a Gaussian
        P = np.random.normal(0, .1, (train.n_users, self.num_factors))
        Q = np.random.normal(0, .1, (train.n_items, self.num_factors))
        self.mu = train.global_mean
        for epoch in range(self.num_epochs):

            for u, i, r_ui in train.all_ratings():
                residual = r_ui - self.mu - P[u] @ Q[i]
                temp = P[u]  # we want to update them at the same time, so we make a temporary variable.
                P[u] += self.alpha * (residual * Q[i] - P[u] * self.reg_all)
                Q[i] += self.alpha * (residual * temp - Q[i] * self.reg_all)
        self.P = P
        self.Q = P

        self.trainset = train

    def estimate(self, u, i):

        if self.trainset.knows_user(u) and self.trainset.knows_item(i):
            # return scalar product of P[u] and Q[i]
            nanCheck = self.P[u] @ self.Q[i] + self.mu

            if np.isnan(nanCheck):
                return self.trainset.global_mean
            else:
                return nanCheck
        else:  # if its not known we'll return the general average.
            return self.trainset.global_mean

class BaisedFunkSVD(surprise.AlgoBase):
    # Randomly initializes two Matrices, Stochastic Gradient Descent to be able to optimize the best factorization for ratings.
    def __init__(self, lr_all, reg_all, n_epoch, n_factors, **kwargs):
        # super(surprise.AlgoBase)
        super().__init__(**kwargs)
        self.lr_all = lr_all  # learning rate for Stochastic Gradient Descent
        self.n_epoch = n_epoch
        self.n_factors = n_factors
        self.reg_all = reg_all

    def fit(self, train):
        # randomly initialize user/item factors from a Gaussian
        P = np.random.normal(0, .1, (train.n_users, self.n_factors))
        Q = np.random.normal(0, .1, (train.n_items, self.n_factors))
        u_bias = np.random.normal(0, .1, train.n_users)
        i_bias = np.random.normal(0, .1, train.n_items)
        self.mu = train.global_mean
        for epoch in range(self.n_epoch):
            for u, i, r_ui in train.all_ratings():
                err = r_ui - (self.mu + P[u] @ Q[i] + u_bias[u] + i_bias[i])
                prevP = P[u].copy()  # we want to update them at the same time, so we make a temporary variable.
                u_bias[u] += self.lr_all * (err - self.reg_all * u_bias[u])
                i_bias[i] += self.lr_all * (err - self.reg_all * i_bias[i])
                P[u] += self.lr_all * (err * Q[i] - self.reg_all * P[u])
                Q[i] += self.lr_all * (err * prevP - self.reg_all * Q[i])
        self.P = P
        self.Q = P
        self.u_bias = u_bias
        self.i_bias = i_bias
        self.trainset = train

    def estimate(self, u, i):

        if self.trainset.knows_user(u) and self.trainset.knows_item(i):
            # return scalar product of P[u] and Q[i]
            prediction = self.mu + self.P[u] @ self.Q[i] + self.u_bias[u] + self.i_bias[i]

            if np.isnan(prediction):
                return self.trainset.global_mean
            else:
                return prediction
        else:  # if its not known we'll return the general average.
            return self.trainset.global_mean



