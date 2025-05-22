from pathlib import Path
from datetime import datetime
from collections import deque
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
args = parser.parse_args()

site = args.site
model = args.model

date = datetime.today().strftime("%Y-%m-%d")
file = f"HSI_log_{date}.log"
path = f"/opt/mlshare/logs/hsi-partnersite/{site}/{model}/logs/"

file_name = Path(path + file)
if file_name.is_file():
    print(f"{file} Exists")
    with open(file_name) as f:
        last_line = deque(f, 1)[0]
        if " | SUCCESS  | " in last_line:
            print(f"{model} SUCCESSFULLY ran on {date}.")
        else:
            print(f"{model} FAILED to run on {date}.")
else:
    print(f"{file} Does Not Exist")
