import pandas as pd
import numpy as np
from argparse import ArgumentParser
import time
import datetime


def main(file_name, dest):
    result = "Movie_Id,Cust_Id,Rating,Time"

    counter = 0
    for line in open(file_name):
        line = line.strip()
        if line.endswith(":"):
            movie_id = int(line.replace(":", ""))
            continue
        else:
            date = line.split(",")[-1]
            timestamp = time.mktime(datetime.datetime.strptime(date, "%Y-%m-%d").timetuple())
            line = line.replace(date, "")
            result += f"{movie_id}," + line + f"{int(timestamp)}\n"
        counter += 1
        # if counter == 10000:
        #     break
    with open(dest, 'w') as dest_file:
        dest_file.write(result)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--source", required=True, type=str)
    parser.add_argument("--dest", required=True, type=str)

    args = parser.parse_args()
    main(args.source, args.dest)