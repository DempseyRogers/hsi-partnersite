import numpy as np
import pandas as pd


class HSI_explainability:

    def __init__(
        self,
        preprocessor,
        preprocessed_np: np.ndarray,
        results_df: pd.DataFrame,
        number_components_explained: float,
    ):
        self.preprocessor = preprocessor
        self.preprocessed_np = preprocessed_np
        self.results_df = results_df
        self.number_components_explained = number_components_explained

    def pca_lin_combo(
        self,
    ):
        id_m = np.identity(self.preprocessor.pre_pca.shape[1])
        id_df = pd.DataFrame(id_m, columns=self.preprocessor.pre_pca.keys())
        lin_combo = self.preprocessor.decomposer.transform(id_df)
        self.exp_df = pd.DataFrame(lin_combo, index=self.preprocessor.pre_pca.keys())
        return self

    def scale_perc_var_ratio(
        self,
    ):
        self.scaled_exp_df = pd.DataFrame(index=self.preprocessor.pre_pca.keys())
        for i, percent in enumerate(self.preprocessor.decomposer.explained_variance_ratio_):
            self.scaled_exp_df[f"pca_comp_{i}"] = (
                self.exp_df[[1]]
                * percent
                / sum(self.preprocessor.decomposer.explained_variance_ratio_)
            )
        return self

    def explain_pred(
        self,
    ):
        feature_means = np.average(self.preprocessed_np, axis=0)
        feature_stds = np.std(self.preprocessed_np, axis=0)
        results_preprocessed_np = self.preprocessed_np[self.results_df["User DF Index"]]
        self.bad_index = []
        for res in results_preprocessed_np:
            zscore = (abs(feature_means - res)) / feature_stds
            scaled_zscore_df = np.array(zscore) * self.scaled_exp_df
            old = []
            for k in scaled_zscore_df.keys():
                temp = np.argpartition(abs(scaled_zscore_df[k]), -self.number_components_explained)[
                    -self.number_components_explained :
                ]
                for t in temp:
                    if t not in old:
                        old.append(t)
            self.bad_index.append(old)
        return self

    def set_original_features(
        self,
    ):
        self.bad_dict = {}
        bad_set = np.unique(np.array(self.bad_index))
        for k in self.preprocessed_df.keys():
            bad_list = []
            for b in self.scaled_exp_df.index[bad_set]:
                with warnings.catch_warnings():  # https://stackoverflow.com/questions/40659212/futurewarning-elementwise-comparison-failed-returning-scalar-but-in-the-futur
                    warnings.simplefilter(action="ignore", category=FutureWarning)
                    if b in self.preprocessed_df[k].unique():
                        bad_list.append(b)
                if len(bad_list) > 0:
                    self.bad_dict[k] = bad_list
        return self
