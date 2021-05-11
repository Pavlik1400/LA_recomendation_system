import numpy as np
import pandas as pd
from math import sqrt


def svd(user_item_matrix, k):

    mask = np.isnan(user_item_matrix)
    masked_arr = np.ma.masked_array(user_item_matrix, mask)

    item_distribution = np.mean(masked_arr, axis=0)

    item_means_tiled = np.tile(item_distribution, (user_item_matrix.shape[0], 1))

    # utility matrix or ratings matrix that can be fed to svd
    utilMat = masked_arr.filled(item_distribution)

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
    filename = 'datasets/MoviesRecommendation.csv'
    train_set, test_set = read_processed_data(filename, nth=1000, coeff_size=0.8)
    print("Read done")

    main_matrix = pd.pivot_table(train_set, values='Rating', index='Cust_Id', columns='Movie_Id')

    print("Matrix done")

    perform_svd(main_matrix, test_set)

# k=20, nth=1000, mean of films
# test done, rmse =  1.2131300758331092

# k=20, nth=500, mean of films
# test done, rmse =  1.2072919709854308
# test done, rmse =  1.2131300758331092