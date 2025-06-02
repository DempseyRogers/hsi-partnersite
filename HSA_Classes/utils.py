from pathlib import Path
import pandas as pd
import numpy as np
import datetime
import splunklib.client as client
import splunklib.results as results
from tqdm import tqdm
import os

keys_path = "/opt/mlshare"
import sys

sys.path.append(keys_path)
from keys import *
import json
import yaml
import datetime

SEVERITY_MAP = {
    "normal": ("normal", "normal"),
    "informational": ("low", "low"),
    "low": ("low", "medium"),
    "medium": ("medium", "medium"),
    "high": ("medium", "high"),
    "critical": ("high", "high"),
}


def read_yaml(file_path):
    """Read a YAML file and return its results"""
    with open(file_path, "r") as f:
        return yaml.safe_load(f)


def config_get_or_default(config, key, default=None):
    """Given a config (dictionary) return the value at location \"key\" or a given default (None if not provided)"""
    try:
        return config[key]
    except:
        if default:
            return default
        else:
            return config["DEFAULT"]


def filter_by_site(df, site_number):
    """Filter a dataframe to a given site using \"orig_splunk_server\" DEPRECATED"""
    return df[
        (df["orig_splunk_server"].str.slice(1, 2) == f"{site_number}")
        | (df["orig_splunk_server"] == f"indexer1.site{site_number}.cs.dhs")
    ]


def get_site(df):
    """Add \"site\" to dataframe with orig_splunk_server DEPRECATED"""
    df["site"] = -1
    for site in range(10):
        df.loc[
            (df["orig_splunk_server"].str.slice(1, 5) == f"{site}IDX")
            | (df["orig_splunk_server"] == f"indexer1.site{site}.cs.dhs"),
            "site",
        ] = site

    return df


## Earliest and latest are given as integer days.
def chunk_query(query, earliest, latest=1):
    """Query splunk using a given query, filtering by relative date given as integer offsets from the current date.
    Also chunks the results which seems to be faster and more accurate"""
    splunk_args = get_splunk_args(earliest, latest)

    data = []
    for args in splunk_args:
        data.append(get_data_from_splunk(query, args))

    return pd.concat(data, ignore_index=True)


def get_splunk_args(earliest, latest):
    """Converts earliest and latest which are number of day offsets to many hourly offsets to facilitate chunking a query"""
    num_chunks = 24 * (earliest - latest)

    kwargs = []
    for offset in range(num_chunks):
        if offset < 24:
            kwargs.append(
                {
                    "earliest_time": f"-{earliest}d@d+{offset}h",
                    "latest_time": f"-{earliest}d@d+{offset+1}h",
                }
            )
        else:
            offset_days = int(offset / 24)
            offset_hours = offset % 24

            kwargs.append(
                {
                    "earliest_time": f"-{earliest-offset_days}d@d+{offset_hours}h",
                    "latest_time": f"-{earliest-offset_days}d@d+{offset_hours+1}h",
                }
            )

    return kwargs


def get_data_from_splunk(query, kwargs):
    """Simplest way to get data from splunk."""
    job = create_splunk_job(query, kwargs, use_ml2=False)
    wait_for_job_to_complete(job)
    return get_results_from_job(job)


def create_splunk_job(query, kwargs, use_ml2=False):
    """Creates and returns a splunk job for a given query"""
    if use_ml2:

        HOST = "10.5.5.59"
        PORT = 8089

    else:

        HOST = "10.5.8.10"
        PORT = "8089"

    # print(f"This is the HOST:{HOST} and the PORT:{PORT}")
    service = client.connect(
        host=HOST, port=PORT, username=user, password=password, verify=False
    )

    job = service.jobs.create(query, **kwargs)

    return job


def wait_for_job_to_complete(job):
    while not job.is_done():
        pass


def get_results_from_job(job):
    """Gets the results from a completed job in increments of 49000 since more than that is prohibited"""
    search_results_json = []
    get_offset = 0
    max_get = 49000
    result_count = int(job["resultCount"])

    while get_offset < result_count:
        r = job.results(
            **{"count": max_get, "offset": get_offset, "output_mode": "json"}
        )
        obj = json.loads(r.read())
        search_results_json.extend(obj["results"])
        get_offset += max_get

    df = pd.DataFrame(search_results_json)

    return df


def query_yesterday(query, earliest=None, latest=None, site=None):
    """Create a splunk query between earliest and latest which are relative day offsets from today. If not provided it defaults to yesterday. DEPRECATED"""
    HOST = "10.5.8.10"
    PORT = 8089

    service = client.connect(
        host=HOST, port=PORT, username=user, password=password, verify=False
    )

    if earliest is None:
        yesterday = datetime.datetime.now() - datetime.timedelta(days=1)
        kwargs = {"earliest_time": yesterday.strftime("%Y-%m-%dT%H:%M:%S"), "count": 0}
        if latest is not None:
            kwargs["latest_time"] = latest.strftime("%Y-%m-%dT%H:%M:%S")
    elif type(earliest) is int:
        kwargs = {"earliest_time": f"-{earliest}d@d", "count": 0}
        if latest is not None:
            kwargs["latest_time"] = f"-{latest}d@d"
    else:
        print("Incorrect format for yesterday")

    print(f"Getting data")

    job = service.jobs.create(query, **kwargs)

    print("waiting ", end="")
    while not job.is_done():
        print(".", end="")
    print(" Done!")

    search_results_json = []
    get_offset = 0
    max_get = 49000
    result_count = int(job["resultCount"])

    while get_offset < result_count:
        r = job.results(
            **{"count": max_get, "offset": get_offset, "output_mode": "json"}
        )
        obj = json.loads(r.read())
        search_results_json.extend(obj["results"])
        get_offset += max_get

    df = pd.DataFrame(search_results_json)

    if site is not None:
        return filter_by_site(df, site)

    return df


def get_yesterday():
    return datetime.datetime.now() - datetime.timedelta(days=1)


## This is used by kerberos and cannot be removed currently
def query_last_week(query, time_back_start=7, time_back_end=1, site=None):
    # HOST='10.5.8.10'
    # PORT=8089

    df = chunk_query(query, earliest=time_back_start, latest=time_back_end)

    if site is not None:
        return filter_by_site(df, site)

    return df


def get_last_week():
    return datetime.datetime.now() - datetime.timedelta(days=7)


def remove_predefined_ips(ip_df, ip_key):
    """INPUTS ip_df: pd.DataFrame(), ip_key: str OUTPUTS quads_df --> pd_DataFrame['q0','q1','q2','q3'] filtered_ips_df --> pd_DataFrame['ips'].
    Takes a dataframe and filters out private ranges, loop back ranges, link local ranges, test ranges, and multicast ranges
    as defined by: https://www.auvik.com/franklyit/blog/special-ip-address-ranges/ per Joseph McDanagh's recommendation on 9/16/2024
    """

    # Input Data Frame Checks  all logging istances here will be at the critical level.
    def check_df_type(df, ip_keys, col_type):
        for ip_key in ip_keys:
            if isinstance(df, pd.DataFrame):
                if df.map(type).nunique()[ip_key] > 1:
                    # loguru logger and log --> More than 1 dtype in IP Col
                    print(f"more than one type")
                    sys.exit(0)
                else:
                    if type(df[ip_key][0]) == col_type:
                        pass
                    else:
                        # loguru logger and log --> IPS NOT STR
                        print(f"ip not str")
                        sys.exit(0)
            else:
                # import loguru logger and log --> NOT a DF
                print("not a pd.df")
                sys.exit(0)

    # check_df_type(ip_df, [ip_key], str)  # check input DF

    # Input ip_key checks
    if type(ip_key) is str:
        if ip_key in ip_df.keys():
            pass
        else:
            # loguru log --> key not in ip_df.keys()
            print(f"{ip_key} not in {ip_df.keys()}")
            sys.exit(0)
    else:
        # loguru log --> ip_key is not a str
        print("not str")
        sys.exit(0)

    # quads seperated as integers so that logic can be applied to rm special ips
    quads_df = (
        ip_df[ip_key]
        .str.split(".", expand=True)
        .astype(int)
        .rename(columns={0: "q0", 1: "q1", 2: "q2", 3: "q3"})
    )

    # Remove special ips with /8 Private Range 10.0.0.0/8 and Loop Back Range 127.0.0.0/8
    q0_8 = [10, 127]
    for i in q0_8:
        quads_df.drop(quads_df[(quads_df["q0"] == i)].index, axis=0, inplace=True)

    # Remove special ips with /16  Private Ranges 192.168.0.0/16 and Link Local 169.254.0.0/16
    q0_16 = [192, 169]
    q1_16 = [168, 254]
    for i in range(len(q0_16)):
        quads_df.drop(
            quads_df[(quads_df["q0"] == q0_16[i]) & (quads_df["q1"] == q1_16[i])].index,
            axis=0,
            inplace=True,
        )

    # Remove special ips with /24  Private Ranges 192.0.0.0/24 and Test Ranges 192.0.2.0/24 198.51.100.0/24 203.0.113.0/24
    q0_24 = [192, 192, 198, 203]
    q1_24 = [0, 0, 51, 0]
    q2_24 = [0, 2, 100, 113]
    for i in range(len(q0_24)):
        quads_df.drop(
            quads_df[
                (quads_df["q0"] == q0_24[i])
                & (quads_df["q1"] == q1_24[i])
                & (quads_df["q2"] == q2_24[i])
            ].index,
            axis=0,
            inplace=True,
        )

    # Remove special ips with /4 MultiCast 224.0.0.0-239.255.255.255
    q0_4 = range(224, 240)
    for i in q0_4:
        quads_df.drop(quads_df[(quads_df["q0"] == i)].index, axis=0, inplace=True)

    # Remove special ips with /12 Private Range 172.16.0.0-192.31.255.255
    q0_12 = [172]
    q1_12 = range(16, 32)
    for i in q1_12:
        quads_df.drop(
            quads_df[(quads_df["q0"] == q0_12[0]) & (quads_df["q1"] == i)].index,
            axis=0,
            inplace=True,
        )

    filtered_ips_df = pd.DataFrame()
    filtered_ips_df["ips"] = (
        quads_df["q0"].astype(str)
        + "."
        + quads_df["q1"].astype(str)
        + "."
        + quads_df["q2"].astype(str)
        + "."
        + quads_df["q3"].astype(str)
    )

    # check outputs
    if len(ip_df.iloc[filtered_ips_df.index]) == 0:
        print("FILTERED TO 0")
        # logger log
        sys.exit()
    return quads_df, filtered_ips_df.index

def filter_prior_predictions(
    results_path, lookback_days, static_key, todays_df=None, run_date=None
):  # , results_file_pattern):
    """INPUTS
    results path --> (str) path to results.json
    lookback_days --> (int) number of days that results should not be repeated in
    static_key --> (str) key from results_df that will be static across comparison dates
    *results_file_pattern --> f-string with d as datetime var ex: f"Results_{'{:04d}'.format(d.year)}-{'{:02d}'.format(d.month)}-{'{:02d}'.format(d.day)}.json"

    OUTPUTS
    todays_results --> (pd.DataFrame()) results from resent run with repeated results removed.
    """

    def get_results_lookback(results_path, lookback_days):  # , results_file_pattern):
        """INPUTS
        results path --> (str) path to results.json
        lookback_days --> (int) number of days that results should not be repeated in

        OUTPUTS
        prior_json --> (list) list of results.json to filter against
        time_range_results --> (pd.DataFrame()) set of prior predictions over lookback_days
        unfound_results--> (list) list of results whose dates are not present in the results_path
        """
        today = run_date
        date_list = [
            today - datetime.timedelta(days=x) for x in range(1, 1 + lookback_days)
        ]
        prior_json = []
        for d in date_list:
            prior_json.append(
                f"Results_{'{:04d}'.format(d.year)}-{'{:02d}'.format(d.month)}-{'{:02d}'.format(d.day)}.json"
            )
            # I want to make this better, more general so that anyone's results*.json can be used
            # prior_json.append(results_file_pattern)
        time_range_results_df = pd.DataFrame()
        unfound_results = []
        for j in prior_json:
            try:
                time_range_results = pd.read_json(results_path + j)
                time_range_results_df = pd.concat(
                    [time_range_results_df, time_range_results]
                )
            except:
                unfound_results.append(j)
        time_range_results_df=time_range_results_df.loc[time_range_results_df.astype(str).drop_duplicates().index]
        return prior_json, time_range_results_df, unfound_results

    def filter_time_range_results(time_range_results_df, todays_df=None):
        """INPUTS
        time_range_results_df --> (pd.DataFrame()) set of predictions over past lookback_days

        OUTPUTS
        todays_results --> (pd.DataFrame()) current predictions without predictions found in time_range_results
        """
        d = datetime.datetime.today()
        todays_file = f"{results_path}Results_{run_date}.json"
        # I want to make this better, more general so that anyone's results*.json can be used
        # todays_file= results_file_pattern
        if isinstance(todays_df, pd.DataFrame):
            todays_results = todays_df
        else:
            todays_results = pd.read_json(todays_file)
        if len(time_range_results_df):
            for e in list(time_range_results_df[static_key]):
                if e in list(todays_results[static_key]):
                    todays_results.drop(
                        todays_results[todays_results[static_key] == e].index[0],
                        inplace=True,
                        axis=0,
                    )
        else:
            print("No Prior Results")
        return todays_results
    prior_json, time_range_results_df, unfound_results = get_results_lookback(
        results_path, lookback_days
    )
    todays_results = filter_time_range_results(time_range_results_df, todays_df)
    return todays_results


def port_type_counter(data, key, dicts, port_types):
    """ Simple accounting of ratio of each port type. NOT WEIGHTED BY NUMBER OF PACKETS SENT.
    INPUTS
    data --> df
    key --> df.key of port
    dicts --> mapping of port

    OUTPUTS
    data --> df with new keys key-port_types[:]
    """
    for (
        port_type
    ) in port_types:  # Generates new keys for the port type and the port classes
        port_key = f"{key}-{port_type}"
        data[port_key] = np.zeros((len(data), 1))

    def count_vals(
        rows,
    ):# Gets the ratio of port classes for a given set of connections
        if (" " in rows[key]):  # For single ports only
            port_list=rows[key].split()
        elif (type(rows[key]) == str):
            port_list= [rows[key]]
        else:
            port_list= rows[key]

        count_list = np.zeros(len(port_types))
        for p in port_list:
            port = dicts[int(p)]
            for i in range(len(port_types)):
                if port_types[i] == port:
                    count_list[i] += 1 / len(port_list)
        for i in range(len(port_types)):
            port_key = f"{key}-{port_types[i]}"
            rows[port_key] = count_list[i]
        rows.drop(key, inplace=True)
        return rows
        

    data = data.apply(count_vals, axis=1)
    return data