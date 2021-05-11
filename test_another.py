import csv

import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix, csr_matrix
import math
import csv
from sparsesvd import sparsesvd
from scipy.sparse.linalg import svds
import jax.numpy
# from scipy.sparse.linalg import *
#
# # constants defining the dimensions of our User Rating Matrix (URM)
# MAX_PID = 4499
# MAX_UID = 1000000
#
#
# def readUrm():
#     # urm = np.zeros(shape=(MAX_UID, MAX_PID), dtype=np.float32)
#     main_dict = dict()
#     with open('../data/comb_prep_1.csv', 'rb') as trainFile:
#         urmReader = csv.reader(trainFile, delimiter=',')
#         for row in urmReader:
#             main_dict[int(row[0]), int(row[2])] = float(row[1])
#
#     coeff_size = 0.8
#     n = len(main_dict.items())
#     sizee = int(coeff_size * n)
#     d1 = dict(main_dict.items()[:sizee])
#     d2 = dict(main_dict.items()[sizee:])
#
#     return csr_matrix(d1, dtype=np.float32), d2
#
#
# # def readUsersTest():
# #     uTest = dict()
# #     with open("./testSample.csv", 'rb') as testFile:
# #         testReader = csv.reader(testFile, delimiter=',')
# #         for row in testReader:
# #             uTest[int(row[0])] = list()
# #
# #     return uTest
# #
# #
# # def getMoviesSeen():
# #     moviesSeen = dict()
# #     with open("./trainSample.csv", 'rb') as trainFile:
# #         urmReader = csv.reader(trainFile, delimiter=',')
# #         for row in urmReader:
# #             try:
# #                 moviesSeen[int(row[0])].append(int(row[1]))
# #             except:
# #                 moviesSeen[int(row[0])] = list()
# #                 moviesSeen[int(row[0])].append(int(row[1]))
# #
# #     return moviesSeen
#
#
# def computeSVD(urm, K):
#     U, s, Vt = sparsesvd(urm, K)
#
#     # dim = (len(s), len(s))
#     # S = np.zeros(dim, dtype=np.float32)
#     # for i in range(0, len(s)):
#     #     S[i, i] = mt.sqrt(s[i])
#
#     # U = csr_matrix(np.transpose(U), dtype=np.float32)
#     # S = csr_matrix(S, dtype=np.float32)
#     # Vt = csr_matrix(Vt, dtype=np.float32)
#
#     return U, s, Vt


def read_processed_data(filename, nth, coeff_size):
    df = pd.read_csv(filename)
    df.drop_duplicates(inplace=True, subset=['Movie_Id', 'Cust_Id'])
    if nth != 1:
        df = df[df.index % nth == 0]

    n = len(df)
    np.random.seed(2021)
    df = df.iloc[np.random.permutation(n)]

    sizee = int(coeff_size * n)
    return df[:sizee], df[sizee:]


def computeSVD(train_set, K):
    # print(train_set)

    # urm = csr_matrix((train_set['Rating'], (train_set['Cust_Id'], train_set['Movie_Id'])))
    urm = pd.pivot_table(train_set, values='Rating', index='Cust_Id', columns='Movie_Id')
    urm = np.nan_to_num(urm, nan=0)
    # print(urm)
    # print(urm.shape)
    # P, s, Ut = sparsesvd(urm, K)
    # urm = csr_matrix(urm)
    P, s, Ut = jax.numpy.linalg.svd(urm)

    # print(P)
    # print(P.shape)
    P = pd.DataFrame(P, index=sorted(list(set(train_set['Cust_Id']))), columns=range(len(s)))
    # P = pd.DataFrame(P, index=range(len(s)), columns=train_set['Cust_Id'])
    Ut = pd.DataFrame(Ut, index=range(len(s)), columns=sorted(list(set(train_set['Movie_Id']))))

    print(P)
    # print("Size of P = ", len(P))
    print(s)
    # print("Size of s = ", len(s))
    print(Ut)
    # print("Size of Ut = ", len(Ut))

    return P, s, Ut


def calculate_rating(P, s, Ut, user_ind: int, item_ind: int):
    rating = 0
    for i in range(len(s)):
        p_val = P[i][user_ind]
        s_val = s[i]
        u_val = Ut[item_ind][i]
        rating += p_val * s_val * u_val
    return rating


def rmse(true, pred):
    # this will be used towards the end
    x = np.array(true) - np.array(pred)
    return sum([xi*xi for xi in x])/len(x)


def test_results(P, s, Ut, train_set: pd.DataFrame, test_set: pd.DataFrame):
    pred = []  # to store the predicted ratings
    should = []

    for _, row in test_set.iterrows():
        user = row['Cust_Id']
        item = row['Movie_Id']
        if user in P.index:
            if item in Ut.columns:
                pred.append(calculate_rating(P, s, Ut, user, item))
                should.append(row['Rating'])
                # print("yes")
            else:
                pred.append(3)
                should.append(row['Rating'])
        else:
            pred.append(3)
            should.append(row['Rating'])

    print("\ntest done, rmse = ", rmse(should, pred))


def main():
    filename = '../data/comb_prep_1.csv'
    train_set, test_set = read_processed_data(filename=filename, nth=1000, coeff_size=0.8)
    print('data read')

    P, s, Ut = computeSVD(train_set=train_set, K=20)
    print('svd done')

    test_results(P, s, Ut, train_set, test_set)


if __name__ == '__main__':
    main()

#
# def computeEstimatedRatings(U, S, Vt, uTest):
#     list_pred = []
#     list_should = []
#
#     rightTerm = S * Vt
#
#     estimatedRatings = np.zeros(shape=(MAX_UID, MAX_PID), dtype=np.float16)
#     for userTest in uTest:
#         prod = U[userTest, :] * rightTerm
#
#         # we convert the vector to dense format in order to get the indices of the movies with the best estimated ratings
#         estimatedRatings[userTest, :] = prod.todense()
#         recom = (-estimatedRatings[userTest, :])
#         for r in recom:
#             if r not in moviesSeen[userTest]:
#                 uTest[userTest].append(r)
#
#                 if len(uTest[userTest]) == 5:
#                     break
#
#     return uTest
#
#
# def main():
#     K = 90
#     urm = readUrm()
#     U, S, Vt = computeSVD(urm, K)
#     uTest = readUsersTest()
#     moviesSeen = getMoviesSeen()
#     uTest = computeEstimatedRatings(urm, U, S, Vt, uTest, moviesSeen, K, True)

# import numpy as np
# import pandas as pd
# from scipy.sparse import coo_matrix,csr_matrix
# from scipy.sparse.linalg import svds
# import math
#
#
# def aaaaa(filepath):
#     # u, r, m = np.loadtxt(filepath, delimiter=',', skiprows=1, usecols=(0, 1, 2)).T
#     df = pd.read_csv(filepath)
#     print("data is")
#     # mat = coo_matrix((r, (u - 1, m - 1)), shape=(u.max(), m.max())).tocsr()
#     mat = coo_matrix((df["Rating"], (df["Cust_Id"] - 1, df["Movie_Id"] - 1)), shape=(df["Cust_Id"].max(), df["Movie_Id"].max())).tocsr()
#     return mat
#
#
# def make_svd(mat, k):
#     U, s, V = svds(mat, k=k)
#
#     U = U[:, 0:k]
#     V = V[0:k, :]
#     s_root = np.diag([math.sqrt(s[i]) for i in range(0, k)])
#
#     Usk = np.dot(U, s_root)
#     skV = np.dot(s_root, V)
#     UsV = np.dot(Usk, skV)
#
#     return UsV


# if __name__ == '__main__':
#     data = aaaaa('../data/comb_prep_1.csv')
#     print(data)
#
#     matrix = make_svd(data, 20)
#     print(matrix)