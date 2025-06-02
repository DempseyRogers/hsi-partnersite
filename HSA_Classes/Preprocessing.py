import numpy as np
import pandas as pd
import re
from time import time as t 
from sklearn.feature_extraction.text import TfidfVectorizer as vecorizer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from loguru import logger as loguru_logger


class HSA_preprocessing:

    def __init__(
        self,
        decomposer: PCA,
        scaler: StandardScaler,
        logger: loguru_logger,
        raw_path: str = "",
        df: str = "",
        drop_keys: list = [],
        type_map: dict = {},
        max_spawn_dummies: int = 500,
        max_spawn_dummies_multi=50,  # This is being set to reduce scanners in model pred dtye == 'multi'
        allow_dict: dict = {},
    ):
        self.drop_keys = drop_keys
        self.logger = logger
        self.max_spawn_dummies = max_spawn_dummies
        self.max_spawn_dummies_multi = max_spawn_dummies_multi
        self.allow_dict = allow_dict

        def allow_lister(df, allow_dict):

            def str_to_list(
                row,
                allow_dict,     
            ):
                for allow_type in allow_dict:
                    for key in allow_dict[allow_type].keys():
                        row[key] = row[key].split()
                return(row)

            def keep_if_one(row, allow_dict):

                for df_keys in allow_dict['keep_if_one'].keys():
                    cond=0

                    df_set=set(row[df_keys])
                    dict_set=set(allow_dict['keep_if_one'][df_keys])
                    if not df_set.isdisjoint(dict_set):
                        cond+=1 

                if cond > 0:
                    return(row)

                else:
                    row[df_keys]=np.nan # cannot figure out how to drop here.. make nan and dropna
                    return(row)

            def drop_if_one(row, allow_dict):
                cond=0

                for df_keys in allow_dict['drop_if_one'].keys():
                    df_set=set(row[df_keys])
                    dict_set=set(allow_dict['drop_if_one'][df_keys])

                    if df_set.isdisjoint(dict_set):
                        cond+=1 

                if cond > 0:
                    return(row)

                else:
                    row[df_keys]=np.nan # cannot figure out how to drop here.. make nan and dropna
                    return(row)

            def drop_if_all(row, allow_dict):
                cond=0

                for df_keys in allow_dict['drop_if_all'].keys():
                    df_set=set(row[df_keys])
                    dict_set=set(allow_dict['drop_if_all'][df_keys])

                    if df_set.issubset(dict_set):
                        cond+=1 

                if cond == len(allow_dict['drop_if_all'].keys()):
                    row[df_keys]=np.nan # cannot figure out how to drop here.. make nan and dropna
                    return(row)

                else:
                    return(row)   

            def keep_if_all(row, allow_dict):
                cond=0

                for df_keys in allow_dict['keep_if_all'].keys():
                    df_set=set(row[df_keys])
                    dict_set=set(allow_dict['keep_if_all'][df_keys])

                    if df_set.issubset(dict_set):
                        cond+=1 

                if cond == len(allow_dict['keep_if_all'].keys()):
                    return(row)

                else:
                    row[df_keys]=np.nan # cannot figure out how to drop here.. make nan and dropna
                    return(row)

            args=(allow_dict,)
            df=df.apply(str_to_list, 
                        args=args, 
                        axis=1,
                        )

            if len(self.allow_dict['keep_if_one']):
                df=df.apply(keep_if_one, 
                        args=args,
                        axis=1,
                        )

            if len(self.allow_dict['drop_if_one']):
                df=df.apply(drop_if_one, 
                        args=args,
                        axis=1,
                        )

            if len(self.allow_dict['drop_if_all']):
                df=df.apply(drop_if_all, 
                        args=args,
                        axis=1,
                        )

            if len(self.allow_dict['keep_if_all']):
                df=df.apply(keep_if_all, 
                        args=args,
                        axis=1,
                        )

            df.dropna(inplace=True, axis=0)
            return(df)

        def multi_connection_data_encoder(
            key : str,
            df : pd.DataFrame,
            max_spawn_dummies : int,
        ): 
            """ Takes a DF with space separated data. The data point with multiple connections are encoded to columns of the data frame. The new encoded columns are populated with the percent the value represents in the event.
            
            - key : key of the multivalued data
            - df : DF containing data
            - max_spawn_dummies : limit to the number of columns encoder can spawn 
            """

            def str_to_list_mc(
                row,
                key,
                row_set,
            ):
                if type(row[key]) == list:
                    row[f"{key}_list"] = row[key]
                elif type(row[key]) == str:
                    row[f"{key}_list"] = row[key].split()
                return(row)

            def list_to_set(
                row,
                key,
                row_set,
            ):
                if type(row[f"{key}_list"]) == list:
                    row_set += list(set(row[f"{key}_list"]))
                else:
                    row_set += [row[f"{key}_list"]]
                row_set = set(row_set)
                try:
                    row_set.remove(np.nan)
                except Exception as e:
                    self.logger.critical(f"list_to_set: {e}")
                    sys.exit()
                return(row_set)

            def set_to_encoded_df_cols(
                row,
                key,
                row_set,
            ):
                if type(row[f"{key}_list"]) == list:
                    for spawn_key in row[f"{key}_list"]:
                        row[f"{spawn_key}_{key}"] += 1 / (len(row[f"{key}_list"]))
                else:
                    row[f'{row[f"{key}_list"]}_{key}'] = 1 
                return(row)

            def clean_up(
                df,
                key,
            ):
                df.drop(key, inplace=True, axis=1)
                df.drop(f"{key}_list", inplace=True, axis=1)
                return(df)

            row_set = []
            args = (key, row_set)
            df = df.apply(
                str_to_list_mc,
                args=args,
                axis=1,
            )

            list_of_lists=list(df[f"{key}_list"])
            row_set=set()
            for sublist in list_of_lists:
                row=[]
                if type(sublist) == list:
                    for element in sublist:
                        row.append(element)
                else:
                    row.append(sublist)
                row_set=row_set.union(set(row))

            if len(row_set) < max_spawn_dummies:
                for spawn_key in row_set:
                    temp = pd.DataFrame(
                        data=np.zeros(len(df)),
                        columns=[f"{spawn_key}_{key}"],
                    )

                    df = pd.concat(
                        [df, temp],
                        axis=1,
                    )

                df = df.apply(
                    set_to_encoded_df_cols,
                    args=args,
                    axis=1,
                )

            df = clean_up(
                df,
                key,
            )            
            return(df)

        def df_dtype_gen(
            df: pd.DataFrame,
            type_map: dict = {},
        ):
            """Casts columns of SPLUNK data to predefined datatype, else drops drill-down information from model consideration."""
            # cast data into specified types
            df.drop(set(df.keys()).difference(set(type_map.keys())), axis=1, inplace=True)

            if self.allow_dict != None:
                df = allow_lister(df, self.allow_dict)
                df.dropna(inplace=True)
                df.to_pickle("~/allow.pkl")

            for k in [key for key in df if type_map[key] != "multi"]:
                df[k] = df[k].astype(type_map[k])
            for k in [key for key in type_map if type_map[key] == "multi"]:
                df = multi_connection_data_encoder(
                        k,
                        df,
                        self.max_spawn_dummies_multi,
                    )
            return df

        if len(raw_path) > 0:
            self.raw_path = raw_path
            df = df_dtype_gen(pd.read_pickle(self.raw_path), type_map)
        else:
            df = df_dtype_gen(df, type_map)

        self.preprocessed_df = df

        self.decomposer = decomposer
        self.scaler = scaler
        self.logger.debug("Scaler and Decomposer passed to df_dtype gen.")

    # def ip_tfidf(
    #     self,
    #     tfidf_ip_count: int,
    #     ip_keys: list,
    #     ip_type: str = "",
    # ):
    #     """Takes ip strings and preforms TFIDF on their individual quads. To generate new vocab words for octets of numbers being repeated in different quads, each quad is shifted by a factor of 256. NOTE: 0-10  were protected vocab, so each octet is also shifted by 10."""

    #     # Encode all ips with term frequency inverse doc freq
    #     @self.logger.catch(level="CRITICAL")
    #     def generate_sentences(row):
    #         offset_arr = [
    #             (i * 256) + 10 for i in range(4)
    #         ]  # separate out quads so 123.123.123 has different vals for each quad

    #         vals = row[f"{ip_type}_ip"].split(".")
    #         vals = [int(j) + offset_arr[i] for i, j in enumerate(vals)]
    #         return " ".join([str(i) for i in vals])

    #     @self.logger.catch(level="CRITICAL")
    #     def unprotect_1(
    #         rows,
    #     ):  # vectorizer has the word "1" as a protected token that is not documented, this returns no score if used
    #         row = rows["sentence"]
    #         rex = "(^1 )"
    #         if re.match(rex, row):
    #             return re.sub(
    #                 rex, "01 ", row
    #             )  # give ip 1.xxx.xxx the value of 01.xxx.xxx to avoid protected vals
    #         else:
    #             return row

    #     @self.logger.catch(level="CRITICAL")
    #     def extract(
    #         results: list,
    #         n: int,
    #     ):  # separate words to match words' scores
    #         return [item[n] for item in results]

    #     for k in ip_keys:
    #         self.preprocessed_df = self.preprocessed_df[self.preprocessed_df[k].isna() == False]

    #     if tfidf_ip_count == 1:
    #         ip_df = self.preprocessed_df
    #         ip_list = list(ip_df[ip_keys[0]].values)

    #         ip_df = pd.DataFrame(ip_list, columns=[f"{ip_type}_ip"])
    #         ip_df["sentence"] = ip_df.apply(generate_sentences, axis=1)

    #         vocab = {f"{i}": i for i in range(1034)}

    #         v = vecorizer(vocabulary=vocab)
    #         v.fit([ip_df["sentence"][0]])
    #         outs = v.transform(ip_df["sentence"]).todense().tolist()
    #         results = []
    #         for o in outs:
    #             results.append([i for i in o if i > 0])

    #         quad_df = pd.DataFrame()
    #         for i in range(4):
    #             quad_df[f"quad_{i}"] = extract(results, i)
    #         self.preprocessed_df.drop(ip_keys[0], inplace=True, axis=1)
    #         self.preprocessed_df = pd.concat([self.preprocessed_df, quad_df], axis=1)
    #         self.logger.debug("Only 1 IP in use, TFIDF Complete.")

    #     if tfidf_ip_count == 2:
    #         ip_df = self.preprocessed_df

    #         pass
    #         ip_list = list(ip_df[ip_keys[0]].values)
    #         ip_list += list(ip_df[ip_keys[1]].values)  # use all quads as vocb items

    #         ip_df = pd.DataFrame(ip_list, columns=[f"{ip_type}_ip"])
    #         ip_df["sentence"] = ip_df.apply(generate_sentences, axis=1)

    #         vocab = {f"{i}": i for i in range(1034)}

    #         v = vecorizer(vocabulary=vocab)
    #         v.fit(ip_df["sentence"])
    #         outs = v.transform(ip_df["sentence"]).todense().tolist()
    #         results = []
    #         for o in outs:
    #             results.append([i for i in o if i > 0])

    #         quad_df = pd.DataFrame()
    #         for i in range(4):
    #             quad_df[f"quad_{i}"] = extract(results, i)

    #         dest_quad = pd.DataFrame()
    #         src_quad = pd.DataFrame()
    #         dest_quad[["dest_q0", "dest_q1", "dest_q2", "dest_q3"]] = quad_df.loc[
    #             : len(quad_df) // 2 - 1
    #         ]  # unravel total voacb list back to src and dest ips
    #         src_quad[["src_q0", "src_q1", "src_q2", "src_q3"]] = quad_df.loc[
    #             len(quad_df) // 2 :
    #         ]

    #         src_quad.reset_index(inplace=True)
    #         src_quad.drop("index", inplace=True, axis=1)
    #         self.dest_quad=dest_quad
    #         self.src_quad=src_quad

    #         self.preprocessed_df.drop(ip_keys, inplace=True, axis=1)
    #         self.preprocessed_df.reset_index(inplace=True)
    #         tfidf_keys=list(self.preprocessed_df.keys().values)+list(self.dest_quad.keys().values)+list(self.src_quad.keys().values)
    #         self.preprocessed_df = pd.concat(
    #             [self.preprocessed_df, dest_quad, src_quad], axis=1, ignore_index=True
    #         )

    #         self.preprocessed_df.columns=tfidf_keys
    #         self.preprocessed_df.drop("index", inplace=True, axis=1)
    #         self.logger.debug("There are 2 IPs in use, TFIDF Complete.")
    #     return self

    def read_raw_get_dummies(
        self,
        max_spawn_dummies: int = 0,
    ):
        """Read dataframe for single user and get dummies for keys containing categorical data and objects. If max_spawn dummies is given will drop keys that generate more dummies than specified. The data frame is then scaled in preparation of pca."""

        df = self.preprocessed_df
        if self.drop_keys:
            df.drop(self.drop_keys, axis=1, inplace=True)
        max_spawn = []

        for k in df.keys():
            if df[k].dtype in ["category", "object"]:
                if len(df[k].unique()) > max_spawn_dummies:
                    df.drop(k, axis=1, inplace=True)
                    max_spawn.append(k)

        d = df[df.select_dtypes(include=["category", "object"]).columns]
        d_keys = d.keys()
        dummies = pd.get_dummies(d)
        df.drop(d_keys, axis=1, inplace=True)
        df = pd.concat([df, dummies], axis=1)

        time = []
        if ("duration" in df.keys()):  
            if (df["duration"].dtype != float):
                for t in df["duration"]:
                    time.append(t.total_seconds())
                df.drop("duration", axis=1, inplace=True)
                df["duration"] = time

        df.columns = df.columns.astype(str)

        for k in df.keys():
            if df[k].nunique() ==1:
                df.drop(k, inplace = True, axis=1)
                self.logger.info(f"Key: {k} Dropped due to STD == 0")
        scaled = self.scaler.fit_transform(df)

        df = pd.DataFrame(scaled, columns=df.keys())
        self.preprocessed_df = df
        self.logger.trace("Preprocessing Encoding Complete. ")
        for key in self.preprocessed_df.keys():
            if self.df[key].isnull().any():
                if len(self.preprocessed_df[key].unique()) ==1:
                    self.preprocessed_df.drop(key, axis = 1, inplace = True)
        self.preprocessed_df.dropna(inplace=True)

        return max_spawn

    def select_number_comps(
        self,
        percent_variance_explained: float = 0.95,
        min_additional_percent_variance_exp: float = 0.01,
    ):
        """Pass the decomposer of choice, pca, and both the percent_variance to explain,
        and the minimum percent of the variance that the addition of another component
        must achieve. Loops will break when percent_variance_explained is achieved, or when
        min_additional_percent_variance_exp is not achieved."""
        # pca = self.decomposer.fit(self.df)
        diff = []
        sum_exp_var = 0
        per_exp = percent_variance_explained
        min_additional_percent_variance_exp = min_additional_percent_variance_exp
        for n_components in range(len(pca.explained_variance_ratio_)):
            temp = sum_exp_var
            sum_exp_var += self.decomposer.explained_variance_ratio_[n_components]
            diff.append(sum_exp_var - temp)
            if sum_exp_var > per_exp:
                select_comps = f"{n_components} components account for %{np.round(100*sum_exp_var,2)} of variance\nAchieved %{100*percent_variance_explained}"
                break
            if diff[-1] < min_additional_percent_variance_exp:
                select_comps = f"{n_components} components account for %{np.round(100*sum_exp_var,2)} of variance\nMore features add less than %{100*min_additional_percent_variance_exp} explanation of variance"
                break
        self.decomposer.set_params(n_components=n_components)
        self.np = self.decomposer.fit_transform(self.df)
        self.n_components = n_components
        self.logger.debug(
            "Number of components selected by percent variance explained completed."
        )
        return select_comps
