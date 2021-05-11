import csv

import numpy as np
import pandas as pd
from math import sqrt
from scipy import linalg
from scipy.sparse import csr_matrix, csc_matrix
from sparsesvd import sparsesvd


def svd(user_item_matrix, k):
    # print(user_item_matrix)

    mask = np.isnan(user_item_matrix)
    masked_arr = np.ma.masked_array(user_item_matrix, mask)

    item_distribution = np.mean(masked_arr, axis=0)
    # item_distribution = np.random.normal(3, 1, size=mask.shape)
    # item_distribution = np.where(item_distribution > 5, 5, item_distribution)
    # item_distribution = np.where(item_distribution < 1, 1, item_distribution)
    # print(item_means)
    item_means_tiled = np.tile(item_distribution, (user_item_matrix.shape[0], 1))

    # utility matrix or ratings matrix that can be fed to svd
    utilMat = masked_arr.filled(item_distribution)

    # item_means = np.mean(utilMat, axis=0)
    # item_means_tiled = np.tile(item_means, (utilMat.shape[0], 1))

    # print(utilMat)
    # utilMat = utilMat - item_means_tiled
    # utilMat = utilMat.T / np.sqrt(utilMat.shape[0] - 1)
    # main_mean = np.mean(user_item_matrix)


    # # main_mean = np.mean(np.mean(user_item_matrix))
    # mean_matrix = np.mean(user_item_matrix)
    # print(mean_matrix)
    # mean_matrix = np.tile(mean_matrix, (1, user_item_matrix.shape[1]))
    # print(mean_matrix)
    # # utilMat = np.nan_to_num(user_item_matrix, nan=np.random.normal(3, 2.5))
    # user_item_matrix.update(mean_matrix)
    # utilMat = user_item_matrix
    print(utilMat)

    # Singular Value Decomposition starts
    # k denotes the number of features of each user and item
    # the top matrices are cropped to take the greatest k rows or
    # columns. U, V, s are already sorted descending.

    print("start np svg")
    U, s, V = np.linalg.svd(utilMat, full_matrices=False)
    print("end np svg")

    U = U[:, 0:k]
    V = V[0:k, :]
    s_root = np.diag([sqrt(s[i]) for i in range(0, k)])

    Usk = np.dot(U, s_root)
    skV = np.dot(s_root, V)
    UsV = np.dot(Usk, skV)

    # UsV = UsV + item_means_tiled
    return UsV

def rmse(true, pred):
    # this will be used towards the end
    x = np.array(true) - np.array(pred)
    return sum([xi*xi for xi in x])/len(x)


def perform_svd(util_mat, test: pd.DataFrame):
    svdout = svd(util_mat, k=20)
    svdout = np.array(svdout)
    print("svd done")

    final_matrix = pd.DataFrame(svdout, columns=util_mat.columns, index=util_mat.index)

    pred = []  # to store the predicted ratings
    should = []
    main_mean = np.mean(svdout)

    user_indexes = set(final_matrix.index.tolist())
    item_indexes = set(final_matrix.columns.tolist())

    for _, row in test.iterrows():
        user = row['Cust_Id']
        item = row['Movie_Id']

        if user not in user_indexes:
            pred.append(main_mean)
            should.append(row['Rating'])
        else:
            if item in item_indexes:
                pred_rating = final_matrix[int(item)][int(user)]
            else:
                pred_rating = np.mean(final_matrix.loc[int(user)])
            pred.append(pred_rating)
            should.append(row['Rating'])
    # print(final_matrix)
    print("\ntest done, rmse = ", rmse(should, pred))


def read_processed_data(filename, nth, coeff_size):
    df = pd.read_csv(filename)
    df.drop_duplicates(inplace=True)
    if nth != 1:
        df = df[df.index % nth == 0]

    n = len(df)
    np.random.seed(2021)
    df = df.iloc[np.random.permutation(n)]

    sizee = int(coeff_size * n)
    return df[:sizee], df[sizee:]



if __name__ == '__main__':
    filename = '../data/comb_prep_1.csv'
    train_set, test_set = read_processed_data(filename, nth=1000, coeff_size=0.8)
    print("Read done")

    # print(train_set)

    main_matrix = pd.pivot_table(train_set, values='Rating', index='Cust_Id', columns='Movie_Id')
    # main_matrix = readUrm(train_set)
    # main_matrix = np.reshape(np.nan_to_num(main_matrix_q, nan=0), newshape=main_matrix_q.shape)
    # rows = main_matrix.index
    # cols = main_matrix.columns

    # print(main_matrix)
    # print(main_matrix['Rating'])
    # print(rows)
    # print(cols)

    # main_matrix = csc_matrix((train_set['Rating'], (train_set['Cust_Id'], train_set['Movie_Id'])), dtype=np.int32)
    # print(main_matrix)

    print("Matrix done")

    perform_svd(main_matrix, test_set)

    # filename = '../data/comb_prep_1.csv'
    # df = pd.read_csv(filename)
    #
    # np.random.seed(2021)
    # df = df.iloc[np.random.permutation(len(df))]
    # df = df[:len(df) // 1000]
    # df.to_csv('../small_data.csv')


# k=20, nth=1000, mean of films
# test done, rmse =  1.2131300758331092

# k=20, nth=500, mean of films
# test done, rmse =  1.2072919709854308


# test done, rmse =  1.2131300758331092