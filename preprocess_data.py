import pandas as pd
import numpy as np
from argparse import ArgumentParser
from tqdm import trange
import os
import sys


def main(args):
    NUMBER_OF_FILES = 4
    folder = args.path

    preffix = "combined_data_"
    for idx in trange(1, NUMBER_OF_FILES + 1):
        file_name = f"{preffix}{idx}.txt"
        full_path = folder + file_name
        if not os.path.exists(full_path) or not os.path.isfile(full_path):
            print(f"{full_path} does not exists or is not a file")
            sys.exit(1)

        print("Reading data...")
        df = pd.read_csv(full_path, header=None, names=['Cust_Id', 'Rating'], usecols=[0, 1])
        df['Rating'] = df['Rating'].astype(float)
        df.index = np.arange(0, len(df))

        print("Preprocess films id")
        # get film ids
        df_nan = pd.DataFrame(pd.isnull(df.Rating))
        df_nan = df_nan[df_nan['Rating'] == True]
        df_nan = df_nan.reset_index()

        movie_np = []
        movie_id = 1

        for i, j in zip(df_nan['index'][1:], df_nan['index'][:-1]):
            temp = np.full((1, i - j - 1), movie_id)
            movie_np = np.append(movie_np, temp)
            movie_id += 1

        # Account for last record and corresponding length
        last_record = np.full((1, len(df) - df_nan.iloc[-1, 0] - 1), movie_id)
        movie_np = np.append(movie_np, last_record)

        # remove those Movie ID rows
        df = df[pd.notnull(df['Rating'])]

        df['Movie_Id'] = movie_np.astype(int)
        df['Cust_Id'] = df['Cust_Id'].astype(int)

        if args.clear:
            print("Clearing data...")
            f = ['count', 'mean']

            df_movie_summary = df.groupby('Movie_Id')['Rating'].agg(f)
            df_movie_summary.index = df_movie_summary.index.map(int)
            movie_benchmark = round(df_movie_summary['count'].quantile(0.7), 0)
            drop_movie_list = df_movie_summary[df_movie_summary['count'] < movie_benchmark].index

            df_cust_summary = df.groupby('Cust_Id')['Rating'].agg(f)
            df_cust_summary.index = df_cust_summary.index.map(int)
            cust_benchmark = round(df_cust_summary['count'].quantile(0.7), 0)
            drop_cust_list = df_cust_summary[df_cust_summary['count'] < cust_benchmark].index

            df = df[~df['Movie_Id'].isin(drop_movie_list)]
            df = df[~df['Cust_Id'].isin(drop_cust_list)]

        print("Saving data...")
        df.to_csv(folder + f"comb_prep_{idx}.csv", index=False)

        # after reading data if you wanna get matrix user - rating, use this:
        # df_p = pd.pivot_table(df, values='Rating', index='Cust_Id', columns='Movie_Id')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-p", "--path", help="Path to the folder with original data", required=True)
    parser.add_argument("-c", "--clear", help="remove films and users that have mot much reviews",
                        type=bool, required=False, default=False)

    args = parser.parse_args()

    main(args)
