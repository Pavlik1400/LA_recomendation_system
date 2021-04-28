import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import os
import json
from math import inf
import pandas as pd


def main(args):
    reses_path = args.input_path
    out_path = args.output_path
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    for result in os.listdir(reses_path):
        if result.endswith(".json"):
            with open(reses_path + "/" + result, 'r') as res_file:
                data = json.load(res_file)
                plt.figure(figsize=(40, 20))
                algo_name = result[:-5]

                # RMSE PART
                # find split with best rmse
                test_with_min_rmse = np.array([inf for _ in data["split0_test_rmse"]])
                min_rmse = inf
                min_rmse_index = ""
                for param in data:
                    if param.startswith("split") and "test" in param:
                        cur_min = min(data[param])
                        if cur_min < min_rmse:
                            min_rmse = cur_min
                            min_rmse_index = param.replace("split", "").replace("_test_rmse", "")
                            test_with_min_rmse = np.array(data[param])
                test_rmses = np.array(data["mean_test_rmse"])
                sorted_indecies = np.argsort(test_rmses)



                plt.subplot(2, 2, 1)
                MIN = 0.8
                plt.grid(axis='y', alpha=0.75)
                plt.yticks(np.arange(MIN, max(test_rmses), max(0.01, (max(test_rmses) - MIN)/30)))
                plt.ylabel("mean RMSE")
                plt.xlabel('Params index')
                plt.title(f'mean RMSE for {algo_name} on test')

                print(f'Best params for {algo_name}: {data["params"][np.where(test_with_min_rmse == np.min(test_with_min_rmse))[0][0]]}')

                for index, value in enumerate(test_rmses[sorted_indecies]):
                    plt.text(index, value+0.01, str(round(value, 4)), rotation=70)
                plt.bar(x=range(len(sorted_indecies)), height=(test_rmses[sorted_indecies]-MIN), tick_label=sorted_indecies,
                        color="red", bottom=MIN)


                # TIME PART
                times = np.array(data["mean_fit_time"])
                plt.subplot(2, 2, 2)
                plt.grid(axis='y', alpha=0.75)
                plt.yticks(np.arange(min(times) - 10, max(times), (max(times) - min(times)) / 15))
                plt.ylabel("Time of fitting (seconds)")
                plt.xlabel('Params index')
                plt.title(f'Time of fitting for {algo_name}')

                for index, value in enumerate(times[sorted_indecies]):
                    plt.text(index, value + 1, str(round(value, 4)), rotation=70)
                plt.bar(x=range(len(sorted_indecies)), height=times[sorted_indecies] - min(times) + 10,
                        tick_label=sorted_indecies,
                        color="green", bottom=min(times) - 10)

                # train rmse
                plt.subplot(2, 2, 3)
                train = np.array(data["mean_train_rmse"])
                MIN = min(train) - 0.1
                plt.grid(axis='y', alpha=0.75)
                plt.yticks(np.arange(MIN, max(train), max(0.01, (max(train) - MIN) / 30)))
                plt.ylabel("mean RMSE")
                plt.xlabel('Params index')
                plt.title(f'mean RMSE for {algo_name} on train')

                for index, value in enumerate(train[sorted_indecies]):
                    plt.text(index, value + 0.01, str(round(value, 4)), rotation=70)
                plt.bar(x=range(len(sorted_indecies)), height=(train[sorted_indecies] - MIN),
                        tick_label=sorted_indecies,
                        color="blue", bottom=MIN)

                plt.subplot(2, 2, 4)
                params = data["params"]
                table_data = None
                for i, param in enumerate(params):
                    if table_data is None:
                        table_data = np.array([[i, *param.values()]])
                    else:
                        table_data = np.vstack((table_data, np.array([i, *param.values()])))

                df = pd.DataFrame(table_data, columns=["index", *params[0]])
                table = plt.table(cellText=df.values, colLabels=df.columns, loc='center',)
                plt.axis('off')


                plt.savefig(out_path + "/" + algo_name + ".png")
                # plt.show()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-i", "--input_path", help="Path to the json files", required=True)
    parser.add_argument("-o", "--output_path", help="Path to the output directory with plots", required=True)
    args = parser.parse_args()

    main(args)
