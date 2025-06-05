from copy import deepcopy
import sys
import os

import Preprocessing as hsa_preprocessing
import DataSet as hsa_dataset
import Model as hsa_model
import Viz as hsa_viz
from sklearn.decomposition import PCA as PCA
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from loguru import logger as loguru_logger
import numpy as np
import pandas as pd
import time
from torch.utils.data import DataLoader


################################################################################
# HSA Script functionality calls keys, utils, HSA_class, and unique_query_preprocessing
################################################################################
class HSA_pipeline:

    def __init__(
        self,
        penalty_ratio: float,
        cutoff_distance: float,
        lr: float,
        anomaly_std_tolerance: float,
        bin_count: int,
        max_spawn_dummies: int,
        percent_variance_explained: float,
        min_additional_percent_variance_exp: int,
        logger: loguru_logger,
        logging_level: str,
        verbose: bool = 0,
        plot_figures: bool = None,  # Show weights and model prediction heatmaps
        save_figures: bool = None,
        save_preprocessed_np: bool = None,
        batch_size: int = 2000,  # Batch size for data loader
        multi_filters: int = 15,  # How many multifilter steps to take
        converge_toll: float = 1e-30,  # Defines convergence of anomaly score during opt
        affinity_matrix_iterations: int = 20,  # Number of powers of affinity matrix to generate
        iterations: int = int(5e3),  # 5e2 Number of optim.adam steps
        base_directory: str = None,
        num_workers: int = 10,
    ):
        self.penalty_ratio = penalty_ratio
        self.cutoff_distance = cutoff_distance
        self.lr = lr
        self.anomaly_std_tolerance = anomaly_std_tolerance
        self.bin_count = bin_count
        self.max_spawn_dummies = max_spawn_dummies
        self.percent_variance_explained = percent_variance_explained
        self.min_additional_percent_variance_exp = min_additional_percent_variance_exp
        self.verbose = verbose
        self.logger = logger
        self.plot_figures = plot_figures
        self.save_figures = save_figures
        self.logger.trace("HSA_pipeline has been generated")
        self.logger.info(
            f"penalty_ratio: {self.penalty_ratio}, cutoff_distance: {self.cutoff_distance}, lr: {self.lr}, anomaly_std_tolerance: {self.anomaly_std_tolerance}, bin_count: {self.bin_count}, max_spawn_dummies: {self.max_spawn_dummies}, percent_variance_explained: {self.percent_variance_explained}"
        )
        # self.static_key = static_key
        self.save_preprocessed_np = save_preprocessed_np

        self.base_directory = base_directory
        self.plot_directory = f"{self.base_directory}/plots"  # Storage location
        self.log_directory = f"{self.base_directory}/logs"  # Storage location
        self.results_directory = f"{self.base_directory}/results"  # Storage location

        self.batch_size = batch_size
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
            sink=f"{self.log_directory}/HSA_log_{self.run_date}.log",
            level=logging_level,
        )

    def infer(
        self,
        df,
    ):
        self.logger.debug("HSA_pipeline inference has begun")
        time_start = time.time()
        df_raw = deepcopy(df)

        self.logger.info("Preprocessing has begun.")
        pca = PCA()
        scaler = StandardScaler()
        prep = hsa_preprocessing.HSA_preprocessing(
            pca,
            scaler,
            self.logger,
            df=df,
            drop_keys=self.drop_keys,
            type_map=self.type_map,
        )
        max_spawn = prep.read_raw_get_dummies(max_spawn_dummies=self.max_spawn_dummies)

        select_comps = prep.select_number_comps(
            percent_variance_explained=self.percent_variance_explained,
            min_additional_percent_variance_exp=self.min_additional_percent_variance_exp,
        )

        preprocessed_np = prep.np
        self.logger.debug(
            "Preprocessing is complete, data has been processed as a np.array ready for use in Torch."
        )
        if len(preprocessed_np) == 0:
            preprocess_warning = (
                "Did not return data from preprocessing! HSA_Preprocessing.py"
            )
            self.logger.critical(preprocess_warning)
            if self.verbose:
                print(preprocess_warning)
            sys.exit()

        df = prep.preprocessed_df
        if self.save_preprocessed_np:
            df.to_pickle(f"{self.base__directory}/prep.pkl")
        # Initialize HSA model
        # Hyper params -- unique to each HSA model, are contained in unique HSA_Model_Configs/
        # Number of workers used in optimization steps
        self.logger.trace("Hyper and Batch parameters are being passed to HSA_model.")
        model = hsa_model.HSA_model(
            self.penalty_ratio,
            self.cutoff_distance,
            self.converge_toll,
            self.anomaly_std_tolerance,
            self.affinity_matrix_iterations,
            self.lr,
            self.logger,
            multifilter_flag=0,
        )
        self.logger.trace("Data is passed to HSA_dataset to make torch dataset.")
        dataset = hsa_dataset.HSA_dataset(preprocessed_np, self.logger)
        self.logger.info("Dataset is passed to Dataloader.")
        loader = DataLoader(
            dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )

        # Set storage location for outputs
        total_anomaly_index = np.array([])

        self.logger.trace("Hyperparameters for this trial have been pickled.")

        # Set logging and directories
        model.set_directories(self.log_directory, self.results_directory)
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
                model.set_trial(
                    i * self.batch_size, self.batch_size, self.unique_id_str
                )
                if (
                    self.verbose
                    and int(len(loader) / 8)
                    and i % int(len(loader) / 8) == 0
                ):
                    print(f"\rEpoch {np.round(100*i/len(loader),2)}%")

                # Model set up and weight generation
                model.read_data(
                    data_multifilter_df=data.squeeze(0)
                ).vertex_weights_distances().weight_generation().graph_evolution()
                # Training steps
                model.train(iterations=self.iterations)
                # Prediction step
                model.infer(df)
                # Store anomalous predictions throughout all batches for use in multi filter
                total_anomaly_index = np.append(total_anomaly_index, model.anomaly_index_raw)
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
        anomaly_prediction_frequency_df = pd.DataFrame()
        anomaly_prediction_frequency_df["User DF Index"] = anomaly_index
        anomaly_prediction_frequency_df.set_index("User DF Index")
        anomaly_prediction_frequency_df["Anomaly Bin Count"] = np.zeros(
            len(anomaly_index)
        )

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
                print(f"Count of 1st rank anomalies: {len(anomaly_index)}")
            self.logger.info(
                f"HSA 1st pass detected {len(anomaly_index)} to be passed to the multifilter."
            )
            ################################################################################
            # Multifilter Model
            try:
                for i in range(self.multi_filters):
                    batch_dataset = hsa_dataset.HSA_dataset(mf_data, self.logger)
                    batch_loader = DataLoader(
                        batch_dataset,
                        batch_size=self.batch_size,
                        num_workers=self.num_workers,
                    )
                    j = 0
                    for data in batch_loader:
                        # # Set up multi filter model
                        MF_model = hsa_model.HSA_model(
                            self.penalty_ratio,
                            self.cutoff_distance,
                            self.converge_toll,
                            self.anomaly_std_tolerance,
                            self.affinity_matrix_iterations,
                            self.lr,
                            self.logger,
                            multifilter_flag=1,
                        )
                        MF_model.set_directories(
                            self.log_directory, self.results_directory
                        )
                        MF_model.set_trial(j * len(data), len(data), self.unique_id_str)
                        MF_model.read_data(
                            data_multifilter_df=data.squeeze(0)
                        ).vertex_weights_distances().weight_generation().graph_evolution()

                        # # Train MF_MODEL
                        MF_model.train(iterations=self.iterations)
                        MF_model.infer(
                            df, multifilter_flag=1, user_location=user_location
                        )
                        j += 1
                    anomaly_prediction_frequency_df.loc[
                        anomaly_prediction_frequency_df["User DF Index"].isin(
                            MF_model.anomaly_index_raw
                        ),
                        "Anomaly Bin Count",
                    ] += 1

                self.logger.trace(
                    f"Multifilter {i} of {self.multi_filters} multifilters is complete."
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

            viz = hsa_viz.HSA_viz(
                MF_model.hsa_model,
                MF_model.preprocessed_np,
                self.batch_size,
                0,
                self.verbose,
                self.plot_figures,
                self.save_figures,
                self.plot_directory,
                self.unique_id_str,
                self.logger,
            )
            viz.heatmap_bin_predictions_vert(
                self.anomaly_std_tolerance,
                MF_model.bin_score,
                MF_model.x_ticks,
                MF_model.anomaly_index_raw,
            )
            self.logger.trace("Bin_predictions Heatmap Compete.")

            # Count how many times an anomaly occurs in the multifilter --> log
            for u in anomaly_prediction_frequency_df["Anomaly Bin Count"].unique():
                c = len(
                    anomaly_prediction_frequency_df[
                        anomaly_prediction_frequency_df["Anomaly Bin Count"] == u
                    ]
                )
                self.logger.info(
                    f"{c} Anomalies were predicted {u} times in MultiFilter"
                )

            # Cast model predictions for anomalies as pd.DF or csv for storage
            pd.set_option("display.max_columns", None)
            pd.options.mode.copy_on_write = True
            results_df = df_raw.iloc[
                anomaly_prediction_frequency_df[
                    anomaly_prediction_frequency_df["Anomaly Bin Count"]
                    > self.bin_count
                ]["User DF Index"]
            ]
            results_df["Anomaly Bin Count"] = list(
                anomaly_prediction_frequency_df[
                    anomaly_prediction_frequency_df["Anomaly Bin Count"]
                    > self.bin_count
                ]["Anomaly Bin Count"]
            )
            results_df["User DF Index"] = list(
                anomaly_prediction_frequency_df[
                    anomaly_prediction_frequency_df["Anomaly Bin Count"]
                    > self.bin_count
                ]["User DF Index"]
            )
            if self.verbose:
                print(
                    "Results DF:",
                    results_df.sort_values("Anomaly Bin Count", ascending=False),
                )
            self.logger.trace("Results DF has been generated.")
            self.logger.debug(
                f"Standard run time {np.round((tend-time_start)/60,2)} for loader size of {self.batch_size}"
            )
            bin_percentage = (
                100
                * len(
                    anomaly_prediction_frequency_df[
                        anomaly_prediction_frequency_df["Anomaly Bin Count"]
                        > self.bin_count
                    ]["User DF Index"]
                )
                / len(anomaly_prediction_frequency_df)
            )

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

            path = f"{self.results_directory}/"
            write = path + results_file

            # if self.lookback_days:
            #     results_df = utils.filter_prior_predictions(
            #         path,
            #         self.lookback_days,
            #         self.static_key,
            #         results_df,
            #         self.run_date_date_obj,
            #     )

            if "level_0" in results_df.keys():
                results_df.drop("level_0", axis=1, inplace=True)

            results_df.set_index("User DF Index").to_json(write, orient="records")

            self.logger.success(
                f"Results DF saved to {write}, run is complete. There are {len(results_df)} anomalies predicted."
            )
            return results_df
        else:
            self.logger.success(
                f"No Anomalies found in data during 1st HSA pass. COMPLETE {self.run_date}"
            )
