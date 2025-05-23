# Hyperspectral Imaging (HSI) Anomaly Detection
### Model Status: Development On ML2
This model will preprocess, generate model weights, infer anomalies, and preform a multifilter function to reduce false positives. 

## Model Requirements
- matplotlib==3.7.4
- numpy==1.24.4
- pandas==2.0.3
- pillow==10.2.0
- python-dateutil==2.8.2
- PyYAML==6.0.1
- scikit-learn==1.3.2
- scipy==1.10.1
- seaborn==0.13.1
- splunk-sdk==1.7.4
- sympy==1.12
- torch==2.1.2
- tqdm==4.66.1

The .venv located at /opt/mlshare/aidav/.venv is currently being updated to support these requirements on all ML servers. The model is being developed on a .venv located at /home/cs.dhs/drogers/projects/.venv. 

# Hyper Spectral Anlomaly Detection
The use of the HSI anomaly detection model takes place over a DEV and a Prod workflow. The DEV workflow includes selecting appropriate data, preprocessing, and hyper parameter selection. Once the model is well fit and production directories are watch by the Splunk CLI the model is moved to the PROD workflow. 

## DEV Workflow 
Collaboration with the INL analyst team is encouraged for all DEV work flow challenges. A first challenge is selecting a data set where anomalous behavior implies malicious actions. While many network connections, even originating from a single user or machine, may contain anomalous behaviors. It is the goal of the Data Scientist to select data where anomalies are indicative of malicious behavior, and the resulting data passed to the Analyst team is actionable. It is recommended that initial Splunk queries used to generate data are written as a collaboration between the Data Scientist and the Analyst. 

### Efficiently Fitting the Model
The Splunk CLI is used to pull data from Splunk, called in the utils.py get_data_from_splunk module. This method is not particularly fast. It is recommended to download a data set generated from your query. 

- HSI_Classes/download_data.py
  - Edit lines 4 and 14, to reflect your query and desired data download location respectively. 
- HSI_Model_Configs/DEVELOPMENT/{new_model_config}.py
  - Edit the downloaded_data_dir inside teh new models configuration file to match your data downloaded location. 
    - If this variable is left as the default: None, data will be downloaded at each model run (PROD).
    - Ex: downloaded_data_dir="../data/cl_attack_data.pkl"
- HSI_Model_Configs/HSI_fitter.sh
  - Edit lines 6-9, to specify a unique site name or site_ALL, config.py script to be fit, and model name for logs and results storage. 
    - This shell script will then call the model with a fixed set of hyper parameters. The model will be run num_trials times. Five panel stats will be record and printed upon completion. 
      - This data shows how many times a certain anomaly is predicted across all trials. 
      - A well fit model should return a high mean percent similar among trials, and low mean standard deviation among trials.
      - Other indicators of a good fit
        - Few anomalous predictions after multi filter. 
        - Large separation of multi fitter scores between "false positives" and predictions.  False positives found during the first pass of the HSI should have a bin count near 0, while valid predictions should have a bin count near the bin count threshold set int the config file (default is 15).
- Model hyperparameter and their results' five panel stats are appended to results/HSI/{site}/{model}/results/parameter_df.pkl for future reference (not implemented - clustering to find fit in hyperparameter space.)
- Logs, plots, and results are located at: results/HSI/{site}/{model}/ based of the naming convention specified it the model's config file. 
#### Hyperparameter Explanation and Selection (defaults provided)
- Preprocessing
    - max_spawn_dummies = 500
      - Sets an upper limit on the size of the encoding space per feature.
    - percent_variance_exp = 0.95
      - Sets a stopping limit for percent variance explained while using PCA
    - min_exp = 0.005
      - Sets minimum percent variance explained for a feature to be kept in dataset after PCA
    - remove_keys = []
        - Provides method of dropping features determined unimportant after generating data with Splunk CLI
    - tfidf_ip_count = 1  
      - tfidf_ip_count can be 0, 1, or 2 if no ips should be considered, only a single, or dest and src
    - ip_keys = ['src_ip', 'dest_ip']
      - IP keys that should be broken in to quads and passed to TFIDF for encoding
    - ip_keys_filtered = [ip_keys[0]] 
      - Removes data points where IP field contains specified ranges ie loop back, feed forward, .etc. 

- Modeling
    - penalty_ratio = 0.75  
      -  Penalty$^n$ for powers of affinity matrix, applied in HSI_Classes/HSI_Model.py Penalized Objective Function.
      -  Increases the importance of data similarity across different topological scales.
      -  Increasing this value increases model generalizability and anomalies predicted.
    - cutoff_dist = 3  
      - Cut off distance for weight sets
      - Increasing this value increases generalizability and anomalies predicted.
      - Increases width of gaussian which informs the similarity metric.
    - lr = 2.73  
      - Learning rate for torch.optim.adam

- Reporting Thresholds
    - anom_std_toll = 1.5 
      - Minimum number of STD form mean to define an anomaly 
    - bin_count = 14 
      - The number of times an anomaly needs to be predicted during the multifilter stage to be passed to the results df

## PROD Work Flow
After buy in form analyst team, the analyst will submit an avanti ticket. This ticket will specify the directory the Splunk universal forwarder will watch for new data. At this point Spunk index and sourcetypes will be specified for future threat hunting. 

### Moving from DEV to PROD
All major changes occur in HSI_Model_Configs/DEVELOPMENT/{new_config}.py. After the model is fit and it is aggreeed that it can be moved to production move HSI_Model_Configs/DEVELOPMENT/{new_config}.py to HSI_Model_Configs/PRODUCTION/{new_config}.py.
- Verify that results directory is set correctly by changing the production flag from production = 0 to production = 1.
- Verify that an appropriate production logging level is set by setting logging_level = "INFO".

Add the new config file to the current job scheduler. As of 1/13/2025, this is done by editing the HSI_runner_shell.sh. In the future this will likely be handled with Slurm. To run the model daily:
- Add ipython {new_config}.py to HSI_runner_shell.sh
- HSI requires GPU to infer, so results must be rsynced to the producion directory specified in the avanti ticket.
- Follow the following format to ensure that the umap is correct when files are migrated to ML1. 
  - rsync -og --perms --chmod=775 --chown=svcmlbind@cs.dhs:mlusers /opt/mlshare/results/HSI/site6/cb_custom/Results_$today.json svcmlbind@cs.dhs@csml1.cslinux.dhs:/opt/mlshare/results/HSI/cb_custom_s6/
- Commit and merge the branch through Git

NOTE: The HSI_runner_shell.sh is called by a cronjob owned by svcmlbind.  If a different run cadence is required it may be necessary to call the config.py from the crontab, do not forget to source the .venv. 

After the migration is complete, verify that the job is running as expected. 

# Production Models
Production Models configurations are set in the project HSI_Model_Configs. Standard config files include a splunk query, data type index to be included in modeling, and an instantiation of the  HSI_pipe.HSI_pipeline([desired hyper params]).infer() 
```
import HSI_Classes.HSI_Pipeline as HSI_pipe
from loguru import logger

query_date= f"search earliest={-8+d}d@d latest={-1+d}d@d index=summary source=corelight_notice_long_connection "

query_context= '''| where src_bytes <= 500
| regex src_ip="^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$"
| regex dest_ip="^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$"
| table *'''

query= query_date+query_context

model = HSI_pipe.HSI_pipeline(
    query,
    typemap,
    remove_keys,
    tfidf_ip_count,
    ip_keys,
    output_path,
    penalty_ratio,
    cutoff_dist,
    lr,
    anom_std_toll,
    bin_count,
    unique_id_str,
    max_spawn_dummies,
    percent_variance_exp,
    min_exp,
    site,
    logger,
    query_name,
    logging_level,
    ip_keys_filtered=ip_keys_filtered,
    production_results_dir=production_results_dir,
    lookback_days=7,
    static_key="first_uid",
    multi_port_conn=multi_port_set,
    downloaded_data_dir= downloaded_data_dir
)
results_df = model.infer()
```
 See HSI_Model_Configs for more concrete examples. 

## HSI Data Acquisition
Data is sourced from Splunk using the utils.py module found at /opt/mlshare/aidav. R&D is still being done on the live data to optimize the model's performance. An example of the current state of the queries:

```
import utils

query='search index="central_summary" earliest="-1d@d" source=summary_acct_login sourcetype=stash host="s11sh1cslinuxdhs" orig_sourcetype=corelight_kerberos 
| table *'

df= utils.get_data_from_splunk(query, {}) 
```

# Model Classes  
The model is composed of three primary classes, Preprocessing, HSI_Model, and Visualization. 

## Preprocessing
After collecting data from Splunk, module df_type_gen, parses the data to determine the datatypes contained in each feature by sampling the first specified number of samples in each. This step is skipped if a column_types dict is provided.  After determining the respective feature data types, each feature is cast into its type. read_raw_get_dummies encodes categorical and object data,   converts time-like objects to data types supported by PyTorch, and scales the data from the np.array provided by df_type_gen. Optional features include filtering common pairs of IPs and port type counter (port_type dict is required). Finally, select_number_comps uses a sklearn.decomposer (pca is recommended) to select relevant features from the provided data. This module returns relevant features determined by principal component analysis, namely by selecting the features needed to explain a specified minimum amount of variance in the data (default: percent_variance_exp=.95). The second option is to stop adding features when they explain less than a specified amount of variance of the data (default: min_exp=.01).

After using this class, the model has access to a pandas dataframe of the raw data from the Splunk query for reporting, a scaled, encoded, and pca nd.array of the model for training, and the fit sklearn decomposer and scaler. 

## HSI_model
This is the primary class, it consists of several mundane modules for setting directories and functionality flags, which we will not discuss.  The bulk of the model can be summed up in the following modules.  Model weights are generated by vertWeighs_distances. In this module we choose to diviate from the processes outlined in the source document by Xianchang et al, while the resulting mathematics remains the same. Instead of looping through the data, data is copied into a (3,m,m) matrix, a transposed copy of this matrix is also generated. Computations are then made into matrix operations allowing for significant performance speed ups. This module returns the edgeWeights, vertexWeights, and distances, which are outlined in the section How HSI Detection Works. By the end of this module all data is in the form of PyTorch tensors. 
The affinityGen module takes the weights and distances provided by vertWeight_distances to generate the affinity matrix, similarity matrix, gamma matrix, and D matrix as discussed/derived in the How HSI Anomaly Detection Works section. These matrices are then passed into the graphEvo module. This module raises both the affinity and D matrix through a range of powers, allowing for data relations to be compared on different spatial scales. 
To determine which pixels are anomalous we define a penalized objective function in the module torch_POF. This module defines the constraints placed on the anomaly score vector and the penalties placed on the function. torch_POF is then passed to the optimize_torch_POF module. optimize_troch_POF calls the torch.optim.adam to optimize the torch_POF, namely minimize the anomaly scores for each event in the data, as prescribed by the POF and its parameters. The module loops through a specified number of optimization steps until the anomaly scores converge to a specified tolerance (1e-60 by default). Once convergance is achieved for the first powers of the evolved matrices, provided by graphEvo, the result is used as the starting point for the next power of the matrix. If the resulting anomaly scores are with in the stopping tolerance then inference is complete.   

The final modules model_predictions global_collect_multifilter, and uni_shuffel_multifilter work in conjunction. The purpose of these modules is to determine which anomaly scores are greater than a specified threshold of standard deviations from the mean of the anomaly score vector. These are the outliers that we have predicted to be anomalous. They then collect these predictions into a separate dataframe for later filtering, as well as note their locations in the raw data provided by preprocessing. As all batches in the epoch are inferred upon the anomalies are collected and their raw data locations are recorded. After the epoch is completed the data predicted non anomalous is randomly sampled and added to the anomalous predictions, until the anomalous predictions account for 10% of the data in the new data. This data is then randomly shuffled. A second HSI model then predicts on the shuffled data. In this step we count the number of times that data events are predicted as anomalous.

### Using Collected Anomalies in Multifilter 
This multifilter step serves several purposes. Most importantly it allows events predicted as anomalous within a given batch to be compared with others throughout the entire epoch. This reduces false positives by allowing comparisons between events in different batches. For example, normal scheduled processes that occurring long period of time apart, will likely be in different batches and not compared in the initial pass.  A process may be tagged anomalous because it only occurs once a day, but when several of these events are compared at once in the  multifilter, they no longer appear anomalous.  The other purpose is to measure the model's "certainty"/fit on its anomalous predictions.  If all anomalous events are predicted as anomalous through each iteration of the multifilter, this event is less likely to be a false positive. However, if an anomalous event is predicted non anomalous, once compared to events throughout the epoch, multiple times it is more likely to be a false positive.   

After this step is completed, we have access to the raw data from Splunk for the events predicted as anomalous, as well as the number of times that they were predicted anomalous in the multifilter. Aditional Optional features include filter_prior_preds (FPP). FPP is intended for use on models that run multiple times in a given query time range (time range passed to splunk). The FPP requires results from prior runs. If selected it will remove previously seen predictions in the most current run. This avoids repeating predictions on run-dates with access to the same data. 

## HSI_viz
The final class is used to visualize predictions and provide a measure of explainability for the model. This class contains a heatmap_WeightsMatrix that when passed a model generates heat maps of the model wights, matrices, and preprocessed data. The resulting image can be saved by specifying a directory and passing the save_fig flag. This image demonstrates that the anomalous preprocessed data events lead to hot spots in the model weights and matrices that propagate through the optimization step. The primary module, heatmap_bin_predictions, generates two heatmaps and a line plot. These pertain to the preprocessed data passed to the multifilter model, the resulting anomaly score locations, and the resulting anomaly score values. This image may be used to visually inspect that anomolus events coespond to larger anomaly scores, and which components of a particular event stand out from the others.  

# How HSI Anomaly Detection Works
Unsupervised HSI and Graph Evolution
Motivation for this model comes from the paper: Graph Evolution-Based Vertex Extraction for Hyperspectral Anomaly Detection, by Xianchang et al. The model discussed in this paper was used to detect anomalies in HSI images. This method has been adapted to detect anomalous connections in network communications in partner site 0 conn log data. This method was selected for its ability to compare pixels in function space, placing less importance on similarity in close proximity in geometric space. Anomaly scores are computed by measuring a similarity like weight by considering a pixels distance from its neighbors along multi channel data. Anomaly scores also depend on the density of similar pixels throughout the domain of comparison.
Description
The model selects anomalous data though comparisons using density and distance based measures after computing a euclidean distance between points.
Edge Weight Generation:
For a list of data, $\vec{X}\in \real^{n,m}$, of length $n$ with $m$ features, we first calculate the distance of every point to all others: 

$$d_{ij}=||\vec{x_i}-\vec{x_j}|| $$

Edge weights are determined using a GRB function: 

$$ \gamma_i=\sum_{i\neq j}e^{-\left( d_{ij}/d_c\right)^2} \quad:\quad\vec{\gamma}\in \real^n $$ 

where the cut off distance, $d_c$, is a tunable hyperparameter which selects how much a pixel's separation effects the edge weight in the density measurement. 
Similarity Based Edge Weight Generation:
The similarity matrix, $\vec{S}\in \real^{n,n}$, also encodes information pertaining to the separation of pixels:

$$s_{ij}=e^{\left(- d_{ij}/d_c^2 \right)}$$

The similarity matrix is then normalized using a distance matrix, $\vec{D}\in \real^{n,n}$. 

$$ d_{i=j}=\left( \left(\sum_j^n s_{ij}\right)^{-1/2} \right) \quad d_{i\neq j}=0$$

The normalized similarity matrix, $\tilde{S}\in \real^{n,n}$, is defined as: 

$$ \tilde{S}=\vec{D}\vec{S}\vec{D}$$

Generation of the Affinity Matrix:
We now extend on some principals from graph theory. To consider how pixel relations evolve on different spatial scales in function space, we define an affinity matrix, $\vec{A}\in \real^{n,n}$, similar to the adjacency matrix from simply connected graphs in graph theory. 

$$ \vec{A}=\vec{\Gamma}\tilde{S}\vec{\Gamma} $$

Where $\vec{\Gamma} \in \real^{n,n}$, is defined as: 
$$ {\gamma_{i=j}'}={\gamma_i}\quad \gamma_{i\neq j}=0$$

### Graph Evolution:
We use the affinity matrix in parallel to the use of the adjacency matrix from a simply connected graph in graph theory. Classically the adjacency matrix is used to count the number of connections between vertices in a simply connected graph. This relationship may be used to investigate vertex relations on different topological scales by rasing, or evolving, the adjacency matrix to a specified power. In our case we keep a set of evolved $\vec{A}$ and $\vec{D}$ matrices to consider pixel relations on different topological scales.

 $$\bold{A}=\set{\vec{A}, \vec{A}^2,\vec{A}^3,\dots,\vec{A}^k} $$ 
 
 and 
 
 $$\bold{D}=\set{\vec{D}, \vec{D}^2,\vec{D}^3,\dots,\vec{D}^k} $$ 
 
 where $k\in \mathbb{Z}$ is sufficiently large, such that optimization of the penalized objective function, discussed next, occurs as the set is exhausted.

### Penalized Objective Function / Vertex Extraction:
Our goal is to extract pixels with the least similarity to those in the background. An assumption is that there are far fewer anomalous pixels then background pixels with strong similarity. We generate a similarity metric, $\vec{m}\in\real^{n}$, also known as an anomaly score to measure pixel relationships. We define an objective function, $f(\vec{m})$ that is quadratic in $\vec{m}$ to evaluate the cohesive properties encoded in the density information contained in $\vec{A}$, and is linear in $\vec{m}$ to evaluate the spacial information encoded in $\vec\Gamma^T\vec{D}$.

$$ f\left(\vec{m}\right)=\frac{1}{2}\vec{m}^T\vec{A}^k\vec{m}+\left(\vec{\gamma}^T\vec{D}\right)^T\vec{m} $$

We assume that anomalous pixels are less cohesive then the background/non anomalous pixels. This leads to the constrain equations, $0\le {m_i}\le1$ for $m_i\in\vec{m}$. When these constraints are violated we the objective function is penalized according to the penalty function, $p(\vec{m_p})$.
$$ \vec{p}(\vec{m_p})=r^k\vec{m_p}^q $$ 
where 
$$ q \in \mathbb{Z}:q%2=0$$ 
determines the strength of the penalty and 

$$ \vec{m_{p}} \left\{\begin{aligned} 
m_{p}&=0:\quad \quad 0\le m_{i}\le 1
\\ \ m_{p}&=-m_i:\quad 0\ge m_{i}
\\ m_{p}&=m_i:\quad\quad 1\le m_{i} \end{aligned} \right. $$


Anomaly scores are then determined by minimizing the completed penalized objective function, ${\Phi}(\vec{m})\equiv f(\vec{m})+\sum_{i}^n {p_i}(\vec{m})$, using PyTorch's Adam Optimizer and corresponding hyperparameter set.

### Anomaly Score Threshold

After retrieving the anomaly score vector $\vec{m}$ that minimizes the penalized objective function, each $m_i\in\vec{m}$ quantifies how anomalous a pixel is in comparison to the other pixels in the set. These continuous values can be sorted into discrete anomalous bins by measuring the distance from the mean score in standard deviations. The anom_std_toll sets the cutoff number of std's defining anomalies for the HSI.HSI_viz class. By default anom_std_toll=3, which maps scores with less than 3 std from the mean to 0 in sns.heatmap when predicting (set to background pixel color).

### Multi Filter

False positives may occur during single passes over a data set. This is because during predicting on a subset of the data, a pixel may appear anomalous in its subset. However, this pixel may have many similar neighboring pixels contained in a following data subset, making the anomalous prediction a false positive.
To reduce this false positive rate, anomalous pixels are collected across each batch. Once the epoch is completed and anomalies from the entire data set are collected, we also collect non anomalous pixels. The multi filter data set is composed of $%10$ anomalous pixels and $%90$ background pixels. The model then predicts on the multi filter dataset, allowing pixels considered anomalous in a local neighborhood of the image to be compared to pixels found throughout the entire image.
This process is repeated multiple times. Every time a anomalous pixel is classified anomalous in these iterations the count is updated. At the end of the iterations if a previously anomalous pixel has not been classified as anomalous in the multi filter data set, it is removed as a false positive. Multiple epochs can be undertaken until the set of anomalous pixels converges.
