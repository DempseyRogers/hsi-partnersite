# Hyper Spectral Anomaly Detection Parameter Description

## DataSet

### Parameters 

-
-

### Objects

- \_\_len\_\_ :
- \_\_getitem\_\_ :

### Outputs

-
-

## Explainability

### Parameters 

- bad_dict :
- bad_index :
- exp_df :
- number_components_explained :
- preprocessed_np :
- preprocessor :
- results_df :
- scaled_exp_df :

### Objects

- explain_prediction :
- pca_linear_combo :
- scale_percent_variance_ratio :
- set_original_features :

### Outputs

-
-


## ImagePreprocessing

### Parameters 

- decomposer :
- df :
- number_components :
- preprocessed_df :
- raw_path :
- scaler :

### Objects

- read_raw_get_dummies :
- select_number_comps :

### Outputs

-
-


## Model

### Parameters 

- aff_matrix :
- affinity_matrix_iterations :
- all_data :
- all_index_mf :
- all_index_user :
- anomalous_location :
- anomaly_index_raw :
- anomaly_prediction_frequency_df :
- anomaly_score_old :
- batch_size :
- bin_df :
- bin_score :
- current_anomaly_index :
- cutoff_distance :
- d_matrix :
- device :
- distances :
- edgeWeights :
- gam_matrix :
- init_anomaly_score :
- iterations :
- log_directory :
- logger :
- lr :
- m :
- multifilter_flag :
- penalty_ratio :
- preprocessed_df :
- preprocessed_np :
- results_directory :
- sets :
- shuffler :
- sim_matrix :
- start_idx :
- std_toll :
- stopping_toll :
- total_anomaly_index :
- unique_id_str :
- user_plt :
- verbose :
- vertex_weights :

### Objects

- apf_df_generation :
- global_collect_multifilter_df :
- graph_evolution :
- infer :
- local_collect_multifilter_df :
- read_data :
- set_directories :
- set_trial :
- torch_POF :
- train :
- uni_shuffle_multifilter_df :
- vertex_weights_distances :
- weight_generation :

### Outputs

-
-


## MultiFilter

### Parameters 

-
-

### Objects

- hsa_dataset :
- hsa_model :
- logger :
- multifilter :

### Outputs

-
-


## Pipeline

### Parameters 

- aff_matrix :
- affinity_matrix_iterations :
- all_data :
- all_index_mf :
- all_index_user :
- anomalous_location :
- anomaly_index_raw :
- anomaly_prediction_frequency_df :
- anomaly_score_old :
- batch_size :
- bin_df :
- bin_score :
- current_anomaly_index :
- cutoff_distance :
- d_matrix :
- device :
- distances :
- edgeWeights :
- gam_matrix :
- init_anomaly_score :
- iterations :
- log_directory :
- logger :
- lr :
- m :
- multifilter_flag :
- penalty_ratio :
- preprocessed_df :
- preprocessed_np :
- results_directory :
- sets :
- shuffler :
- sim_matrix :
- start_idx :
- std_toll :
- stopping_toll :
- total_anomaly_index :
- unique_id_str :
- user_plt :
- verbose :
- vertex_weights :

### Objects

- DataLoader :
- HSA_pipeline :
- MaxAbsScaler :
- PCA :
- StandardScaler :
- deepcopy :
- hsa_dataset :
- hsa_model :
- hsa_preprocessing :
- hsa_viz :

### Outputs

-
-


## Preprocessing

### Parameters 

- allow_dict :
- decomposer :
- drop_keys :
- logger :
- max_spawn_dummies :
- max_spawn_dummies_multi :
- n_components :
- np :
- preprocessed_df :
- raw_path :
- scaler :

### Objects

- read_raw_get_dummies :
- select_number_comps :

### Outputs

-
-


## Viz

### Parameters 

- batch_size :
- figures :
- logger :
- m :
- plots_directory :
- preprocessed_np :
- save_fig :
- start_idx :
- unique_id_str :
- verbose :

### Objects

- heatmap_bin_predictions :
- heatmap_bin_predictions_vert :
- heatmap_weights_matrix :

### Outputs

-
-


## trial_comparison

### Parameters 

-
-

### Objects

-
-

### Outputs

-
-

### Parameters 

-
-
## utils

### Objects

-
-

### Outputs

- filter_prior_predictions :
- np :
- pd :
- port_type_counter :
- remove_predefined_ips :
