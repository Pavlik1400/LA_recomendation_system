import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import os
import json
from math import inf


def main(args):
    reses_path = args.input_path
    out_path = args.output_path
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    for result in os.listdir(reses_path):
        if result.endswith(".json"):
            with open(reses_path + "/" + result, 'r') as res_file:
                data = json.load(res_file)
                plt.figure(figsize=(20, 10))
                algo_name = result[:-5]


                # find split with best rmse
                test_rmses = np.array([inf for _ in data["split0_test_rmse"]])
                min_rmse = inf
                min_rmse_index = ""
                for param in data:
                    if param.startswith("split") and "test" in param:
                        cur_min = min(data[param])
                        if cur_min < min_rmse:
                            min_rmse = cur_min
                            min_rmse_index = param.replace("split", "").replace("_test_rmse", "")
                            test_rmses = np.array(data[param])
                sorted_indecies = np.argsort(test_rmses)

                MIN = 0.8
                plt.subplot(1, 1, 1)
                plt.grid(axis='y', alpha=0.75)
                plt.yticks(np.arange(MIN, max(test_rmses) + 1, max(0.01, (max(test_rmses) - MIN)/30)))
                plt.ylabel("RMSE")
                plt.xlabel('Params index')
                plt.title(f'RMSE for {algo_name}')

                # print(data["params"][np.where(test_rmses == np.max(test_rmses))[0][0]])

                for index, value in enumerate(test_rmses[sorted_indecies]):
                    plt.text(index, value+0.01, str(round(value, 4)), rotation=70)
                plt.bar(x=range(len(sorted_indecies)), height=(test_rmses[sorted_indecies]-MIN), tick_label=sorted_indecies,
                        color="red", bottom=MIN)
                plt.savefig(out_path + "/" + algo_name + "_RMSE" + ".png")
                plt.show()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-i", "--input_path", help="Path to the json files", required=True)
    parser.add_argument("-o", "--output_path", help="Path to the output directory with plots", required=True)
    args = parser.parse_args()

    main(args)
