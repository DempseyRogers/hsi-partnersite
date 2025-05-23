import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time
from datetime import datetime
import logging
import os
import torch as t
from torch.utils.data import Dataset
import sys
import re
from sklearn.feature_extraction.text import TfidfVectorizer as vecorizer
from sklearn.decomposition import PCA as PCA
from sklearn.preprocessing import StandardScaler
from  loguru import logger as loguru_logger

class HSI_Image_preprocessing:
    def __init__(
        self,
        decomposer: PCA,
        scaler: StandardScaler,
        raw_path: str = "",
    ):
        self.raw_path = raw_path
        self.raw_data = pd.read_pickle(self.raw_path)
        self.decomposer = decomposer
        self.scaler = scaler

    def read_raw_get_dummies(
        self,
        drop_keys: list = [],
        max_spawn_dummies: int = 0,
    ):
        """Read dataframe for single user and get dummies for keys containing categorical data and objects
        if max_spawn dummies is given will drop keys that generate more dummies than specified.
        the data frame is then scaled in preparation of pca."""

        df = pd.read_pickle(self.raw_path)
        df.drop(drop_keys, axis=1, inplace=True)
        for k in df.keys():
            if df[k].dtype in ["category", "object"]:
                if len(df[k].unique()) > max_spawn_dummies:
                    df.drop(k, axis=1, inplace=True)
                else:
                    d = pd.get_dummies(df[k])
                    for dk in d.keys():
                        df[dk] = d[dk]
                    df.drop(k, axis=1, inplace=True)
        time = []
        for t in df["duration"]:
            time.append(t.total_seconds())
        df.drop("duration", axis=1, inplace=True)
        df["duration"] = time
        scaled = self.scaler.fit_transform(df)
        df = pd.DataFrame(scaled, columns=df.keys())
        self.df = df

    def select_number_comps(
        self,
        percent_variance_exp: float = 0.95,
        min_additional_percent_variance_exp: float = 0.01,
    ):
        """Pass the decomposer of choice, pca, and both the percent_variance to explain,
        and the minimum percent of the variance that the addition of another component
        must achieve. Loops will break when percent_variance_exp is achieved, or when
        min_additional_percent_variance_exp is not achieved."""

        pca = self.decomposer.fit(self.df)
        diff = []
        sum_exp_var = 0
        per_exp = percent_variance_exp
        min_additional_percent_variance_exp = min_additional_percent_variance_exp
        for num_comp in range(len(pca.explained_variance_ratio_)):
            temp = sum_exp_var
            sum_exp_var += pca.explained_variance_ratio_[num_comp]
            diff.append(sum_exp_var - temp)
            if sum_exp_var > per_exp:
                print(
                    f"{num_comp} components account for %{np.round(100*sum_exp_var,2)} of variance\nAcheived %{100*percent_variance_exp}"
                )
                break
            if diff[-1] < min_additional_percent_variance_exp:
                print(
                    f"{num_comp} components account for %{np.round(100*sum_exp_var,2)} of variance\nMore features add less than %{100*min_additional_percent_variance_exp} explanation of variance"
                )
                break
        self.decomposer.set_params(n_components=num_comp)
        self.df = self.decomposer.fit_transform(self.df)
        self.num_comp = num_comp
