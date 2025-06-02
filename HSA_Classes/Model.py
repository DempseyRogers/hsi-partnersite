import numpy as np
import pandas as pd
import torch as t
from sys import exit
from  loguru import logger as loguru_logger

class HSI_model:

    def __init__(
        self,
        penalty_ratio: float,
        cutoff_dist: float,
        converge_toll: float,
        anomaly_std_toll: float,
        affinity_matrix_iterations: int,
        lr: float,
        logger: loguru_logger,
        m0: int = 0,
        multifilter_flag: int = 0,
    ):
        self.logger = logger

        @self.logger.catch
        def get_free_gpu():
            """Looks at allocated memory on all available devices and returns device with most available memory."""

            allocated_mem = 1000
            free_device = "cuda:0"
            if t.cuda.is_available():
                for device in range(t.cuda.device_count()):
                    device_name = f"cuda:{device}"
                    device = t.device(device_name)
                    if device.type == "cuda":
                        mem = t.cuda.memory_allocated(0) / 1024**3
                        if mem < allocated_mem:
                            free_device = device_name
                            allocated_mem = mem
            else:
                free_device = "cpu"
                self.logger.warning("No GPU available, running on CPU.")
            return free_device

        self.device = get_free_gpu()
        self.penalty_ratio = penalty_ratio
        self.dc = cutoff_dist
        self.stopping_toll = converge_toll
        self.std_toll = anomaly_std_toll
        self.affinity_matrix_iterations = affinity_matrix_iterations
        self.m0 = m0
        self.multifilter_flag = multifilter_flag
        self.lr = lr
        self.logger = logger

    def set_directories(
        self,
        log_directory: str,
        results_directory: str,
    ):
        self.log_directory = log_directory
        self.results_directory = results_directory
        self.logger.trace("HSI Model directories set.")
        return self

    def set_trial(
        self,
        start_idx: int,
        num_samples: int,
        unique_id_str: str,
        verbose: int = 0,
    ):
        self.start_idx = start_idx
        self.num_samples = num_samples
        self.unique_id_str = unique_id_str
        self.verbose = verbose
        self.logger.trace("HSI Model trials set.")
        return self

    def readData(
        self,
        data_multifilter_df: list = [],
    ):
        """Generates the pixel vector to detect anomalies in.  Can also generate an anomaly at a given idx. This anomaly will be the mean of each feature by a random number of anomaly scale."""

        if type(data_multifilter_df) == t.Tensor:
            self.preprocessed_np = data_multifilter_df
        else:
            self.preprocessed_np = t.from_numpy(data_multifilter_df)
        self.num_samples = min(self.num_samples, len(self.preprocessed_np))
        self.user_plt = (
            self.results_directory
            + self.unique_id_str
            + f"_start_idx-{str(self.start_idx)}_num_samples-{str(self.num_samples)}"
        )
        if self.m0 == 0:
            self.m_old = (
                np.ones(len(self.preprocessed_np)) * 1000
            )  # make the initial m for difference greater than stopping_toll
            self.m = np.random.rand(
                self.num_samples
            )  # make the initial m for difference greater than stopping_toll
        self.logger.debug("Initial random anomaly index set.")
        return self

    def vertWeights_distances(
        self,
    ):
        """Defines model weights dependant on pixel separation in function space
        Returns both the pairwise distances and summed differences for each pixel set"""

        p = self.preprocessed_np.to(self.device)
        pl0 = len(self.preprocessed_np)
        p_mat = p.unsqueeze(0)
        p_mat = p_mat.expand(pl0, pl0, -1)
        p_matT = p_mat.transpose(0, 1)
        difference = t.subtract(p_mat, p_matT)
        sq_diff = t.square(difference)
        sum_sqr = t.sum(sq_diff, -1)

        self.distances = t.sqrt(sum_sqr).requires_grad_(False)
        self.vertWeights = (
            t.sum(t.exp(-t.square(t.div(self.distances, self.dc))), 1)
            .unsqueeze(-1)
            .requires_grad_(False)
            .type(t.DoubleTensor)
            .to(self.device)
        )
        self.edgeWeights = (
            t.exp(-t.div(self.distances, self.dc**2))
            .requires_grad_(False)
            .type(t.DoubleTensor)
            .to(self.device)
        )
        self.logger.debug("HSI Model Weight Generated.")
        return self

    def affinityGen(
        self,
    ):
        """Equation 2: Vertex weights * Identity to generate gamma matrix in eq 7
        Equation 6: S\tilde (sim matrix here) use edge weights and scale with d_matrix  --CHOOSING TO USE GRB OVER EUCLIDEAN DIST
                d-matrix  sum of rows of sim_matrix * identity
        Equation 7: gamma*sim*gamma  ~vertex*edge*vertex info

        Returns the 3 [nxn] matrices for Affinity Matrix= aff_matrix, Gamma Matrix= gam_matrix,
        and D matrix= d_matrix"""

        d_vec = t.pow(t.sum(self.edgeWeights, 0), (-1 / 2))
        d_matrix = t.diag(d_vec)
        gam_matrix = t.diag(t.reshape(self.vertWeights, (self.vertWeights.shape[0],)))

        temp = t.matmul(d_matrix, self.edgeWeights)
        sim_matrix = t.matmul(temp, d_matrix)

        temp = t.matmul(gam_matrix, sim_matrix)
        aff_matrix = t.matmul(temp, gam_matrix)

        self.aff_matrix = aff_matrix
        self.sim_matrix = sim_matrix
        self.gam_matrix = gam_matrix
        self.d_matrix = d_matrix
        self.logger.debug("HSI Graphs generated.")
        return self

    def graphEvo(
        self,
    ):
        """Edge weight information is evolved by powers of the affinity  and D matrices. Vertex weights do not need to evolve. Each power of these is needed for optimization of the quadratic in Equation 10. Returns List of sets of matrix to power:  [[A^1,A^2, A^3...A^k],[D^1,D^2, D^3...D^k]]"""

        matrix_list = [self.aff_matrix, self.d_matrix]
        sets = []
        for m in matrix_list:
            m_set = [m]
            for i in range(1, self.affinity_matrix_iterations):
                power = t.matmul(m_set[-1], m_set[0])
                m_set.append(power)
            sets.append(m_set)
        self.sets = sets
        self.logger.debug("HSI Graph Theory complete.")
        return self

    def torch_POF(
        m: t.Tensor,
        affinity_m: t.Tensor,
        vertWeights: t.Tensor,
        d_matrix: t.Tensor,
        power: int,
        penalty_ratio: float,
        device: str,
    ):
        """Equation 10: The quadratic objective fxn  = obj [nx1]
        Equation 11: Best interpretation of constraint eqn/ Penalty terms
        Equation 12: Function to be minimized by choice of anomaly scores
            - best interp: no defn of U, no summation preformed here
        r: penalty scaling that must approach 0 as nu -->k=iter_steps"""

        obj = (1 / 2) * t.matmul(
            t.matmul(t.transpose(m.unsqueeze(1), 0, 1), affinity_m), m.unsqueeze(1)
        )
        c = t.matmul(t.matmul(t.transpose(vertWeights, 0, 1), d_matrix), m.unsqueeze(1))
        neg_constraint = t.lt(m, 0)
        ones = t.ones(m.size()).to(device)
        ge1_constraint = t.gt(t.subtract(m, ones), 0)

        constraint = t.mul(t.logical_or(neg_constraint, ge1_constraint).double(), m)
        pen = t.mul(
            t.pow(penalty_ratio, power), t.pow(constraint, 4)
        )  # 4 here is a set scaling of penalty ratio (arb could be changed)

        phi = obj + c + t.sum(pen)
        return phi.to(device)

    def torch_optimize_POF(
        self,
        torch_POF=torch_POF,
        iterations: float = 1e4,
    ):
        """Using the evolved edge weight information and initialized anomaly scores minimize the penalized objective fxn. Optimize for each power of the evolution using the previous best anomaly score."""

        lr = self.lr
        m_old = t.from_numpy(self.m_old).to(self.device)
        vert_weight = self.vertWeights
        pr = t.tensor(self.penalty_ratio).requires_grad_(True).to(self.device)
        anomaly_score = t.from_numpy(self.m).to(self.device)

        def mac_opt_loop(
            _sets: list,
            _stopping_toll: float,
            _anomaly_score: t.Tensor,
            _pr: float,
            _vert_weight: t.tensor,
            _m_old: t.Tensor,
            _m_mid: t.Tensor,
        ):
            for i in range(len(self.sets[0])):
                m = _anomaly_score.requires_grad_(False)
                affinity_m = _sets[0][i]
                d_matrix = _sets[1][i]

                power = t.tensor(i + 1).requires_grad_(False).to(self.device)
                params = [m]

                optimizer = t.optim.Adam(params, lr=t.tensor(lr), fused=True)
                for j in range(iterations):
                    optimizer.zero_grad()

                    loss = (
                        torch_POF(
                            m,
                            affinity_m,
                            _vert_weight,
                            d_matrix,
                            power,
                            _pr,
                            self.device,
                        )
                        .requires_grad_(True)
                        .to(self.device)
                    )
                    loss.backward()
                    optimizer.step()

                    _anomaly_score = params[0]
                    if t.le(
                        t.sqrt(t.sum((t.pow(t.sub(_anomaly_score, _m_mid), 2)))),
                        t.tensor(_stopping_toll),
                    ):
                        break
                    _m_mid = _anomaly_score

                if t.le(
                    t.sqrt(t.sum((t.pow(t.sub(_anomaly_score, _m_old), 2)))),
                    t.tensor(_stopping_toll),
                ):
                    break
                _m_old = (
                    _anomaly_score  # Update the previous best guess at anomaly score
                )
            return _anomaly_score

        cmp_opt_loop = t.compile(mac_opt_loop, fullgraph=False)
        # cmp_opt_loop = mac_opt_loop #5.8s
        
        # cmp_opt_loop = t.compile(mac_opt_loop, backend="eager", dynamic=True )
        anomaly_score = cmp_opt_loop(
            self.sets, self.stopping_toll, anomaly_score, pr, vert_weight, m_old, m_old
        )

        self.m = anomaly_score.cpu().detach().numpy()
        self.logger.debug("HSI Torch optimization complete.")

    def model_Predictions(
        self, df: pd.DataFrame, multifilter_flag: int = 0, user_location: list = []
    ):
        """Measures the distance from the mean in std for each minimized anomaly score.
        Returns the number of std from mean if greater than the std_anomaly_thresh
        Returns the raw data for anomalous pixels in the bin_df
        Returns the x_ticks for heat-maps (location in sub preprocessed_np)
        Returns the x_label for heat-maps (location in total_preprocessed_np and raw data)
        """

        m_mean = np.mean(self.m)
        m_std = np.std(self.m)

        self.bin_score = abs(self.m - m_mean)

        self.bin_score[np.where(self.bin_score / m_std < self.std_toll)] = 0
        self.bin_score[np.where(self.bin_score / m_std > self.std_toll)] = np.round(
            self.bin_score[np.where(self.bin_score / m_std > self.std_toll)] / m_std, 1
        )
        self.x_ticks = np.where(self.bin_score > 0)[
            0
        ]  # index in current preprocessed_np

        if multifilter_flag:
            self.anomaly_index = user_location[self.x_ticks]
            self.x_label = self.anomaly_index
        else:
            self.anomaly_index = (
                self.x_ticks + self.start_idx
            )  # index in raw data not pcap
            self.x_label = self.anomaly_index

        bin_df = df.iloc[self.anomaly_index]
        bin_df.insert(len(bin_df.keys()), "Bin Score", self.bin_score[self.x_ticks])
        self.bin_df = bin_df
        self.logger.debug("HSI model predictions complete.")

    def local_collect_multifilter_df(
        self,
    ):
        """Collects anomalous pixel data, sub preprocessed_np index, and raw data index
        Returns a data frame consisting of 10% anomalies and 90% background from
        preprocessed_np with start_idx!=0"""

        predicted_anomaly_idx = (np.array(self.x_ticks)).astype(
            int
        )  # Grab the pred anomaly from the sub preprocessed_np

        prob_vec = np.ones(len(self.preprocessed_np)) * (
            1 / (len(self.preprocessed_np) - len(predicted_anomaly_idx))
        )  # assign equal prob of selecting good
        prob_vec[predicted_anomaly_idx] = 0  # assign 0 prob of grabbing anomaly again

        padded_nonanomaly_index = np.random.choice(
            len(self.preprocessed_np),
            self.num_samples - len(predicted_anomaly_idx),
            p=prob_vec,
        )  # random select nonAnom data from current preprocessed_np
        predicted_anomaly_data = self.preprocessed_np[predicted_anomaly_idx]
        mix_data = np.append(
            predicted_anomaly_data,
            self.preprocessed_np[padded_nonanomaly_index],
            axis=0,
        )

        predicted_anom = (
            predicted_anomaly_idx + self.start_idx
        )  # add start idx to match raw user_df data (whole without pca)
        index_padding = padded_nonanomaly_index + self.start_idx
        mix_index = np.append(
            predicted_anom, index_padding
        )  # concat so that anomaly~%10
        self.logger.trace("Local Multifilter Complete")
        return mix_index, mix_data, predicted_anom

    def global_collect_multifilter_df(
        self,
        total_preprocessed_np: pd.DataFrame,
        total_anomaly_index: int,
        mf_num_samples: int,
    ):
        """Given a fixed set of anomalies collects random background pixels
        from throughout the entire preprocessed_np"""

        prob_vec = np.ones(
            len(total_preprocessed_np)
        )  # *1/(len(total_preprocessed_np)-len(total_anomaly_index))
        prob_vec[total_anomaly_index] = 0
        prob_vec = prob_vec / sum(prob_vec)

        padded_nonanomaly_index = np.random.choice(
            len(total_preprocessed_np), mf_num_samples, p=prob_vec
        )  # random select nonAnom data from current preprocessed_np
        mix_data = np.append(
            total_preprocessed_np[total_anomaly_index],
            total_preprocessed_np[padded_nonanomaly_index],
            axis=0,
        )

        mix_index = np.append(
            total_anomaly_index, padded_nonanomaly_index
        )  # concat so that anomaly~%10

        self.logger.trace("Global Multifilter Complete.")
        return mix_index, mix_data, total_anomaly_index

    def uni_shuffle_multifilter_df(
        self,
        mix_index: np.ndarray,
        mix_data: np.ndarray,
        predicted_anom: int,
    ):
        """Works in conjunction with the collect_multifilter_df methods
        by randomly shuffling the anomalies and background data and indices
        in unison for later recall to raw data"""

        ## shuffle after all data is collected
        self.shuffler = np.random.permutation(
            len(mix_index)
        )  # generate a shuffle in unison for index and data
        self.all_index_user = mix_index[self.shuffler]  # shuffle index
        self.all_data = mix_data[self.shuffler]  # same shuffle for data

        self.current_anomaly_index = np.where(
            np.in1d(self.all_index_user, predicted_anom)
        )
        self.all_index_mf = self.all_index_user - self.start_idx
        self.logger.trace("Multifiltered data has been shuffled.")
        return self
