from timeSVD import TimeSVD
from argparse import ArgumentParser
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def main(args, nth=30):
    print("Reading...")
    df = pd.read_csv(args.data)

    print("Preprocessing data...")

    # filter parout part
    df = df[df.index % nth == 0]
    df = df.reset_index()

    # map users to 0-len(users) range
    distinct_users = df["Cust_Id"].unique()
    user_count = distinct_users.shape[0]
    user_mapping = dict(zip(distinct_users, np.arange(user_count)))
    df.loc[:, "Cust_Id"] = df.loc[:, "Cust_Id"].apply(lambda x: user_mapping[x])

    # map items to 0-len(users) range
    distinct_movies = df["Movie_Id"].unique()
    movie_count = distinct_movies.shape[0]
    movie_mapping = dict(zip(distinct_movies, np.arange(movie_count)))
    df.loc[:, "Movie_Id"] = df.loc[:, "Movie_Id"].apply(lambda x: movie_mapping[x])

    df.loc[:, "Index"] = df.index
    df = df[["Index", "Time",  "Cust_Id",  "Movie_Id", "Rating"]]

    if args.verbose:
        print(df.head())
        print(df.tail())
        print(df.info())
        print(df.count() == df.shape[0])
        print(f"Number of Users found: {user_count}")
        print(f"Number of Movies found: {movie_count}")

    # divide into train/test
    mask = np.random.rand(len(df)) < 0.8
    train = df[mask]
    train = train.reset_index()
    train.loc[:, "Index"] = train.index
    train = train[["Index", "Time",  "Cust_Id",  "Movie_Id", "Rating"]]

    test = df[~mask]
    test = test.reset_index()
    test.loc[:, "Index"] = test.index
    test = test[["Index", "Time",  "Cust_Id",  "Movie_Id", "Rating"]]

    train = train.values
    test = test.values

    if args.verbose:
        print(train.shape)
        print(np.max(train[:, 0]))
        print(train[5:, ])

    start_time = df["Time"].min()
    end_time = df["Time"].max()
    algo = TimeSVD(latent_size=40, start_time=start_time, end_time=end_time, data=df.values)

    print("Training...")
    cost_h = algo.train(args.epochs, stochastic_size=120000)

    print("Testing...")
    predictions = algo.predict(test)
    print(f'Predictions: ')
    print(f"RMSE: {algo.RMSE}")
    plt.plot(np.arange(len(cost_h)), cost_h)
    plt.show()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--data", help="Path to file with prepared data", type=str, required=False,
                        default="./data/time_preprocessed1.csv")
    parser.add_argument("--epochs", help="Number of epochs", type=int, required=False,
                        default=20)
    parser.add_argument("--verbose", help="Verbose", action='store_const', const=True,
                        required=False, default=False)

    args = parser.parse_args()
    main(args, 2)
