from copy import deepcopy
import sys
import os

import HSI_Classes.Preprocessing as p
import HSI_Classes.DataLoader as d
import HSI_Classes.Model as m
import HSI_Classes.Viz as v
import HSI_Classes.Explainability
from sklearn.decomposition import PCA as PCA
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from  loguru import logger as loguru_logger
import numpy as np
import pandas as pd
import time
import re
import torch
from torch.utils.data import DataLoader
from datetime import datetime, timedelta


################################################################################
# HSI Script functionality calls keys, utils, HSI_class, and unique_query_preprocessing
# For modularity and consistency the only two scripts that need modified for future models are unique_query
# and potentially the HSI_class.  The unique query script contains storage directory locations, query, and the tunable
# hyper parameters. The HSI_class contains the preprocessing class. Depending on analyst needs a custom function
# may be needed in the preprocessing class. Otherwise the HSI_class will remain the same.
# All tunable hyper parameters are contained in the unique_query_preprocessing.py
# HSI_auto should now be generic for all sites and queries
# HSI_class may contain unique preprocessing functions in the preprocessing class depending on the query and specific needs
# keys contains log in information to access splunk and other databases
# utils contains custom functions for all models to collect .json data from splunk.
################################################################################
class HSI_pipeline:

    def __init__(
        self,
        query: str,
        type_map: dict,
        drop_keys: list,
        penalty_ratio: float,
        cutoff_dist: float,
        lr: float,
        anomaly_std_toll: float,
        bin_count: int,
        max_spawn_dummies: int,
        percent_variance_exp: float,
        min_additional_percent_variance_exp: int,
        logger: loguru_logger,
        logging_level: str,
        verbose: bool = 0,
        production_results_directory: bool = None,
        ip_keys_filtered: str = None,
        static_key: str = None,
        plot_figures: bool = None,  # Show weights and model prediction heatmaps
        save_figures: bool = None,
        save_preprocessed_np: bool = None,
        num_samples: int = 2000,  # Batch size for data loader
        multi_filters: int = 15,  # How many multifilter steps to take
        converge_toll: float = 1e-30,  # Defines convergence of anomaly score during opt
        affinity_matrix_iterations: int = 20,  # Number of powers of affinity matrix to generate
        iterations: int = int(5e3),  # 5e2 Number of optim.adam steps
        base__directory: str = None,
        num_workers: int = 10,
    ):
        self.penalty_ratio = penalty_ratio
        self.cutoff_dist = cutoff_dist
        self.lr = lr
        self.anomaly_std_toll = anomaly_std_toll
        self.bin_count = bin_count
        self.max_spawn_dummies = max_spawn_dummies
        self.percent_variance_exp = percent_variance_exp
        self.min_additional_percent_variance_exp = min_additional_percent_variance_exp
        self.verbose = verbose
        self.logger = logger
        self.plot_figures = plot_figures
        self.save_figures = save_figures
        self.logger.trace("HSI_pipeline has been generated")
        self.logger.info(
            f"Trial Info:\ntype_map: {self.type_map}, penalty_ratio: {self.penalty_ratio}, cutoff_dist: {self.cutoff_dist}, lr: {self.lr}, anomaly_std_toll: {self.anomaly_std_toll}, bin_count: {self.bin_count}, max_spawn_dummies: {self.max_spawn_dummies}, percent_variance_exp: {self.percent_variance_exp}"
        )
        self.production_results_directory = production_results_directory
        self.ip_keys_filtered = ip_keys_filtered
        self.static_key = static_key
        self.save_preprocessed_np = save_preprocessed_np

        start_date, end_date = re.findall("\d+d@d", query)
        date_offset = int(end_date[:-3]) - 1
        self.run_date = (datetime.today() - timedelta(days=date_offset)).strftime(
            "%Y-%m-%d"
        )
        self.run_date_date_obj = datetime.today() - timedelta(days=date_offset)
        self.base_directory = base_directory
        self.plot_directory = f"{self.base_directory}/plots"  # Storage location
        self.log_directory = f"{self.base_directory}/logs"  # Storage location
        self.results_directory = f"{self.base_directory}/results"  # Storage location

        self.num_samples = num_samples
        self.multi_filters = multi_filters
        self.converge_toll = converge_toll
        self.affinity_matrix_iterations = affinity_matrix_iterations
        self.iterations = iterations
        self.num_workers = num_workers

        dir_list = [
            self.base__directory,
            self.plot_directory,
            self.log_directory,
            self.results_directory,
        ]  # Set up for model __init__
        try:
            for directory in dir_list:  # Ensure dirs exist for model __init__
                os.makedirs(directory, exist_ok=True)
            self.logger.debug("All output directories currently exist.")
        except Exception as e:
            self.logger.critical(f"An output directory is missing.\n {e}")
            sys.exit()
        self.current_logger = self.logger.add(
            sink=f"{self.log_directory}/HSI_log_{self.run_date}.log",
            level=logging_level,
        )

    def infer(
        self,
        df,
    ):
        self.logger.debug("HSI_pipeline inference has begun")
        time_start = time.time()

        # %% Get data from splunk
        # Data preprocessing -- queries are unique to each HSI model. query, drop_keys, and type maps are contained in the unique_query_preprocessing.py
        df_raw = deepcopy(df)

        self.logger.info("Preprocessing has begun.")
        pca = PCA()
        scaler = StandardScaler()
        prep = p.HSI_preprocessing(
            pca,
            scaler,
            self.logger,
            df=df,
            drop_keys=self.drop_keys,
            type_map=self.type_map,
        )
        max_spawn = prep.read_raw_get_dummies(max_spawn_dummies=self.max_spawn_dummies)

        select_comps = prep.select_number_comps(
            percent_variance_exp=self.percent_variance_exp,
            min_additional_percent_variance_exp=self.min_additional_percent_variance_exp,
        )

        preprocessed_np = prep.df
        self.logger.debug("Preprocessing is complete, data has been processed as a DF.")
        if len(preprocessed_np) == 0:
            preprocess_warning = (
                "Did not return data from preprocessing! HSI_Preprocessing.py"
            )
            self.logger.critical(preprocess_warning)
            if self.verbose:
                print(preprocess_warning)
            sys.exit()

        df = prep.preprocessed_df
        if self.save_preprocessed_np:
            df.to_pickle(f"{self.base__directory}/prep.pkl")
        # Initialize HSI model
        # Hyper params -- unique to each HSI model, are contained in unique HSI_Model_Configs/
        # Number of workers used in optimization steps
        self.logger.trace("Hyper and Batch parameters are being passed to HSI_model.")
        model = m.HSI_model(
            self.penalty_ratio,
            self.cutoff_dist,
            self.converge_toll,
            self.anomaly_std_toll,
            self.affinity_matrix_iterations,
            self.lr,
            self.logger,
            multifilter_flag,
        )
        self.logger.trace("Data is passed to HSI_dataset to make torch dataset.")
        dataset = d.HSI_dataset(preprocessed_np, self.logger)
        self.logger.info("Dataset is passed to Dataloader.")
        loader = DataLoader(dataset, batch_size=num_samples, num_workers=num_workers)

        # Set storage location for outputs
        total_anomaly_index = np.array([])

        if os.path.isfile(f"{self.results_directory}/parameter_df.pkl"):
            parameter_df = pd.read_pickle(f"{self.results_directory}/parameter_df.pkl")
        else:
            parameter_df = pd.DataFrame()
        self.logger.trace("Hyperparameters for this trial have been pickled.")

        # Set logging and directories
        model.set_directoryectories(self.log_directory, self.results_directory)
        self.logger.success(
            f"Query Generated {len(df_raw)} samples. After PreProcessing step {len(preprocessed_np)} were passed to the model."
        )
        self.logger.debug(f"Initial Keys:dtypes {self.type_map} ")
        self.logger.info(
            f"After {pca}, data shape: {preprocessed_np.shape}, broken into {len(loader)} data loaders"
        )
        self.logger.info(f"Features dropped due to max_spawn: {max_spawn}")

        try:
            i = 0  # for displaying epoch progress as completed
            self.logger.info("Starting to run through the dataloader on initial pass.")
            for data in loader:  # setting up gpus
                model.set_trial(i * num_samples, num_samples, self.unique_id_str)
                if (
                    self.verbose
                    and int(len(loader) / 8)
                    and i % int(len(loader) / 8) == 0
                ):
                    print(f"\rEpoch {np.round(100*i/len(loader),2)}%")

                # Model set up and weight generation
                model.readData(
                    data_multifilter_df=data.squeeze(0)
                ).vertWeights_distances().affinityGen().graphEvo()
                # Training steps
                model.torch_optimize_POF(iterations=iterations)
                # Prediction step
                model.model_Predictions(df)
                # Store anomalous predictions throughout all batches for use in multi filter
                total_anomaly_index = np.append(total_anomaly_index, model.x_label)
                i += 1
        except:
            self.logger.critical(f"Initial pass FAILED. {sys.exc_info()[:]}")
            sys.exit()

        self.logger.success("Completed dataloader on initial pass.")
        # Compare anomalous predictions with non anomalous data spanning the set (set initial for multifilter)
        mix_index, mix_data, anomaly_index = model.global_collect_multifilter_df(
            preprocessed_np,
            total_anomaly_index[: len(preprocessed_np)].astype(int),
            mf_num_samples=9 * len(total_anomaly_index),
        )
        anomaly_pred_freq_df = pd.DataFrame()
        anomaly_pred_freq_df["User DF Index"] = anomaly_index
        anomaly_pred_freq_df.set_index("User DF Index")
        anomaly_pred_freq_df["Anom Pred Count"] = np.zeros(len(anomaly_index))

        # Randomly shuffle anomalies from all batches in unison
        model.uni_shuffle_multifilter_df(
            mix_index.astype(int), mix_data.astype(int), anomaly_index.astype(int)
        )
        mf_data = model.all_data
        self.logger.debug(
            "Anomalous data has been colleted into first multifilter dataset."
        )

        # mf_location=model.current_anomaly_index
        user_location = model.all_index_user
        tend = time.time()
        if len(anomaly_index) > 0:
            if self.verbose:
                print(f"len anom 1st: {len(anomaly_index)}")
            self.logger.info(
                f"HSI 1st pass detected {len(anomaly_index)} to be passed to the multifilter."
            )
            ################################################################################
            # Multifilter Model
            try:
                for i in range(multi_filters):
                    batch_dataset = d.HSI_dataset(mf_data, self.logger)
                    batch_loader = DataLoader(
                        batch_dataset, batch_size=num_samples, num_workers=num_workers
                    )
                    j = 0
                    for data in batch_loader:
                        # # Set up multi filter model
                        MF_model = m.HSI_model(
                            self.penalty_ratio,
                            self.cutoff_dist,
                            converge_toll,
                            self.anomaly_std_toll,
                            affinity_matrix_iterations,
                            self.lr,
                            self.logger,
                            multifilter_flag=1,
                        )
                        MF_model.set_directoryectories(
                            self.log_directory, self.results_directory
                        )
                        MF_model.set_trial(j * len(data), len(data), self.unique_id_str)
                        MF_model.readData(
                            data_multifilter_df=data.squeeze(0)
                        ).vertWeights_distances().affinityGen().graphEvo()

                        # # Train MF_MODEL
                        MF_model.torch_optimize_POF(iterations=iterations)
                        MF_model.model_Predictions(
                            df, multifilter_flag=1, user_location=user_location
                        )
                        j += 1
                    anomaly_pred_freq_df.loc[
                        anomaly_pred_freq_df["User DF Index"].isin(MF_model.x_label),
                        "Anom Pred Count",
                    ] += 1

                self.logger.trace(
                    f"Multifilter {i} of {multi_filters} multifilters is complete."
                )
                # # Global multifilter
                mix_index, mix_data, anomaly_index = (
                    model.global_collect_multifilter_df(
                        preprocessed_np,
                        total_anomaly_index[: len(preprocessed_np)].astype(int),
                        mf_num_samples=9 * len(total_anomaly_index),
                    )
                )
                MF_model.uni_shuffle_multifilter_df(mix_index, mix_data, anomaly_index)
                mf_data = MF_model.all_data
                user_location = MF_model.all_index_user
            except:
                self.logger.critical(
                    f"Multifilter FAILED on filter.  {sys.exc_info()[0]}"
                )
                sys.exit()
            self.logger.success("Multifilter Complete.")

            viz = v.HSI_viz(
                MF_model.m,
                MF_model.preprocessed_np,
                num_samples,
                0,
                self.verbose,
                self.plot_figures,
                self.save_figures,
                self.plot_directory,
                self.unique_id_str,
                self.logger,
            )
            viz.heatmap_bin_predictions_vert(
                self.anomaly_std_toll,
                MF_model.bin_score,
                MF_model.x_ticks,
                MF_model.x_label,
            )
            self.logger.trace("Bin_predictions Heatmap Compete.")

            # Count how many times an anomaly occurs in the multifilter --> log
            for u in anomaly_pred_freq_df["Anom Pred Count"].unique():
                c = len(
                    anomaly_pred_freq_df[anomaly_pred_freq_df["Anom Pred Count"] == u]
                )
                self.logger.info(
                    f"{c} Anomalies were predicted {u} times in MultiFilter"
                )

            # Cast model predictions for anomalies as pd.DF or csv for storage
            pd.set_option("display.max_columns", None)
            pd.options.mode.copy_on_write = True
            results_df = df_raw.iloc[
                anomaly_pred_freq_df[
                    anomaly_pred_freq_df["Anom Pred Count"] > self.bin_count
                ]["User DF Index"]
            ]
            results_df["Anom Pred Count"] = list(
                anomaly_pred_freq_df[
                    anomaly_pred_freq_df["Anom Pred Count"] > self.bin_count
                ]["Anom Pred Count"]
            )
            results_df["User DF Index"] = list(
                anomaly_pred_freq_df[
                    anomaly_pred_freq_df["Anom Pred Count"] > self.bin_count
                ]["User DF Index"]
            )
            if self.verbose:
                print("Results DF:", results_df.sort_values("Anom Pred Count", ascending=False))
            self.logger.trace("Results DF has been generated.")
            # Logging and save results df
            self.logger.debug(
                f"Standard run time {np.round((tend-time_start)/60,2)} for loader size of {num_samples}"
            )
            bin_perc = (
                100
                * len(
                    anomaly_pred_freq_df[
                        anomaly_pred_freq_df["Anom Pred Count"] > self.bin_count
                    ]["User DF Index"]
                )
                / len(anomaly_pred_freq_df)
            )

            # %% Updating parameter df with this runs info
            temp_df = pd.DataFrame(
                {
                    "penalty_ratio": [self.penalty_ratio],
                    "cutoff_dist": [self.cutoff_dist],
                    "anomaly_std_toll": [self.anomaly_std_toll],
                    "lr": [self.lr],
                    "bin_count": [self.bin_count],
                    "bin_perc": [bin_perc],
                }
            )
            parameter_df = pd.concat([parameter_df, temp_df], ignore_index=True)
            self.logger.debug(
                "Parameter df has been updated with multifilter consistency metrics"
            )
            if self.verbose:
                print("-------------------------\n")
                time2 = time.time()
                print(
                    f"Inference time on {len(df_raw)} samples took {np.round(time2-time_start, 2)} sec"
                )
            old_mask = os.umask(
                0
            )  # set read write permissions for universal read so universal forwarder can see output.
            results_file = f"Results_{self.run_date}.json"
            if (
                self.production_results_directory
            ):  # Write results /opt/mlshare/results/HSI/*model
                path = f"{self.production_results_directory}/"
            else:
                path = f"{self.results_directory}/"
            write = path + results_file
            if self.lookback_days:
                results_df = utils.filter_prior_preds(
                    path,
                    self.lookback_days,
                    self.static_key,
                    results_df,
                    self.run_date_date_obj,
                )

            self.logger.success(
                f"Results DF saved to {write}, run is complete. There are {len(results_df)} anomalies predicted."
            )

            if "level_0" in results_df.keys():
                results_df.drop("level_0", axis=1, inplace=True)

            results_df.set_index("User DF Index").to_json(write, orient="records")
            parameter_df.to_pickle(f"{path}parameter_df.pkl")

            return results_df
        else:
            self.logger.success(
                f"No Anomalies found in data during 1st HSI pass. COMPLETE {self.run_date}"
            )
