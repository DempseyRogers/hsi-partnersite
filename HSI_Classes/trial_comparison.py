import numpy as np
import json
from datetime import datetime
import pandas as pd
import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "-s",
    "--site",
    help="The site name as seen in the results directory path. Ex: site6",
)
parser.add_argument(
    "-m",
    "--model",
    help="The model name to run as seen in the results directory path. Ex: cb_custom",
)
parser.add_argument(
    "-k",
    "--key",
    help="The static key that ramins the same through out all trials",
)
parser.add_argument(
    "-t",
    "--today",
    help="The base date used through out all trials in +%Y-%m-%d",
)
parser.add_argument("-n", "--num_trials", help="The number of trials to compare Ex: 8")
args = parser.parse_args()

site = args.site
static_key = args.key
model = args.model
path = f"/opt/mlshare/output/hsi-partnersite/{site}/{model}/"   # Trial comp will only be used in dev.  Fix this path to the dev results dir
num_trials = int(args.num_trials)
today= args.today

def get_index_list(path: str = "", num_trials: int = 4, static_keyu: str = "first_uid", today = None) -> list:
    results_index = []
    for r in range(1, 1 + num_trials):
        index_list = []
        
        try:
            file = f"{path}Results_{today}{r}.json"
            data = pd.read_json(file)

            for i in range(len(data)):
                index_list.append(data.loc[i][static_key])
        
        except Exception as e:
            print(f"Did NOT FIND: {file}\n{e}")
            index_list.append('')
        results_index.append(index_list)

    return results_index


def similarity_matrix(results_index: list, num_trials: int = 4):  # df
    similarity_matrix = np.ones([num_trials, num_trials]) * 100
    keys = []
    for i in range(len(results_index)):
        keys.append(f"Trial_{i}")
        for j in range(i + 1, len(results_index)):
            common = len(set(results_index[i]).intersection(set(results_index[j])))
            max_len = max(len(results_index[i]), len(results_index[j]))

            if max_len:
                similarity_matrix[i, j] = np.round(100 * common / max_len, 2)
                similarity_matrix[j, i] = np.round(100 * common / max_len, 2)

            else:  # both have 0 pred
                similarity_matrix[i, j] = 100
                similarity_matrix[j, i] = 100

    similarity_df = pd.DataFrame(similarity_matrix, index=keys, columns=keys)
    similarity_df["mean_perc_sim"] = similarity_df.mean()
    similarity_df["std_perc_sim"] = similarity_df.std()

    return similarity_df


results_index = get_index_list(path, num_trials, static_key, today)
similarity_df = similarity_matrix(results_index, num_trials)
print(similarity_df)
print(
    f"Mean(Mean(perc_sim): {np.round(similarity_df['mean_perc_sim'].mean(),2)}%, Mean(STD(perc_sim): {np.round(similarity_df['std_perc_sim'].mean(),2)}%"
)
print(f"Mean predictions {np.round(np.mean([len(x) for x in results_index]),2)}")
