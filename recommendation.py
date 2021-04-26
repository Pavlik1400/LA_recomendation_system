
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import multiprocessing

import multiprocessing
from joblib import Parallel, delayed

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import os
print(os.listdir("archive"))

import surprise
from surprise import SVDpp

FILE_NAME = ""
def correct_file(filename):
    with open(filename, 'r', encoding="utf-8") as from_:
        with open(f"{filename[:-4]}_corrected.csv", "w", encoding="utf-8") as to:
            new_file = ["book_id,user_id,rating\n"]
            file_id = ''
            for line in from_:
                line = line.strip()
                if line.endswith(":"):
                    file_id = line[:-1]
                    continue
                line = ','.join(line.split(',')[:-1])
                new_file.append(f"{file_id},{line}\n")
            to.writelines(new_file)

def process(args):
    residual = args[0] - np.dot(args[1], args[2])
    return np.array([args[1] + args[3] * residual * args[2], args[2] + args[3] * residual * args[1]])

class ProbabilisticMatrixFactorization(surprise.AlgoBase):
    # Randomly initializes two Matrices, Stochastic Gradient Descent to be able to optimize the best factorization for ratings.
    def __init__(self, learning_rate, num_epochs, num_factors, **kwargs):
        # super(surprise.AlgoBase)
        super().__init__(**kwargs)
        self.alpha = learning_rate  # learning rate for Stochastic Gradient Descent
        self.num_epochs = num_epochs
        self.num_factors = num_factors

    def fit(self, train):
        # randomly initialize user/item factors from a Gaussian
        P = np.random.normal(0, .1, (train.n_users, self.num_factors))
        Q = np.random.normal(0, .1, (train.n_items, self.num_factors))

        # num_cores = multiprocessing.cpu_count()
        # pool = multiprocessing.Pool(processes=num_cores)
        # PQ = None
        for epoch in range(self.num_epochs):
            # PQ = np.array(pool.map(process, ((r_ui, P[u], Q[i], self.alpha) for u, i, r_ui in train.all_ratings())))
            for u, i, r_ui in train.all_ratings():
                residual = r_ui - np.dot(P[u], Q[i])
                temp = P[u, :]  # we want to update them at the same time, so we make a temporary variable.
                P[u, :] += self.alpha * residual * Q[i]
                Q[i, :] += self.alpha * residual * temp

        # self.P = PQ[:, 0]
        # self.Q = PQ[:, 1]
        self.P = P
        self.Q = P

        self.trainset = train

    def estimate(self, u, i):
        # returns estimated rating for user u and item i. Prerequisite: Algorithm must be fit to training set.
        # check to see if u and i are in the train set:
        # print('gahh')

        if self.trainset.knows_user(u) and self.trainset.knows_item(i):
            # print(u,i, '\n','yep:', self.P[u],self.Q[i])
            # return scalar product of P[u] and Q[i]
            nanCheck = np.dot(self.P[u], self.Q[i])

            if np.isnan(nanCheck):
                return self.trainset.global_mean
            else:
                return np.dot(self.P[u, :], self.Q[i, :])
        else:  # if its not known we'll return the general average.
            # print('global mean')
            return self.trainset.global_mean


if __name__ == '__main__':
    FILE_NAME = "combined_data_1_corrected.csv"
    raw=pd.read_csv(FILE_NAME)
    raw.drop_duplicates(inplace=True)
    raw = raw[raw.index % 1000 == 0]
    print('we have' ,raw.shape[0], 'ratings')
    print('the number of unique users we have is:', len(raw.user_id.unique()))
    print('the number of unique books we have is:', len(raw.book_id.unique()))
    print("The median user rated %d books. " %raw.user_id.value_counts().median())
    print('The max rating is: %d ' %raw.rating.max() ,"the min rating is: %d " %raw.rating.min())
    raw.head()

    raw =raw[['user_id' ,'book_id' ,'rating']]
    # when importing from a DF, you only need to specify the scale of the ratings.
    reader = surprise.Reader(rating_scale=(1 ,5))
    # into surprise:
    data = surprise.Dataset.load_from_df(raw ,reader)
    # Alg1 = ProbabilisticMatrixFactorization(learning_rate=0.05 ,num_epochs=4 ,num_factors=10)
    # data1 = data.build_full_trainset()
    # Alg1.fit(data1)
    # print(raw.user_id.iloc[4] ,raw.book_id.iloc[4])
    # print(Alg1.estimate(raw.user_id.iloc[4] ,raw.book_id.iloc[4]))
    #
    # gs = surprise.model_selection.GridSearchCV(ProbabilisticMatrixFactorization, param_grid={'learning_rate' :[0.005 ,0.01],
    #                                                                             'num_epochs' :[5 ,10],
    #                                                                             'num_factors' :[10 ,20]}
    #                                            ,measures=['rmse', 'mae'], cv=2)
    # gs.fit(data)
    #
    # print('rsme: ' ,gs.best_score['rmse'] ,'mae: ' ,gs.best_score['mae'])
    # best_params = gs.best_params['rmse']
    # print('rsme: ' ,gs.best_params['rmse'] ,'mae: ' ,gs.best_params['mae'])



    bestVersion = ProbabilisticMatrixFactorization(learning_rate=0.007
                                                   ,num_epochs=20
                                                   ,num_factors=20)
    # bestVersion = SVDpp(n_factors=10, n_epochs=100, lr_all=0.01)
    # we can use k-fold cross validation to evaluate the best model.
    kSplit = surprise.model_selection.KFold(n_splits=10 ,shuffle=True)
    for train ,test in kSplit.split(data):
        bestVersion.fit(train)
        prediction = bestVersion.test(test)
        surprise.accuracy.rmse(prediction ,verbose=True)



