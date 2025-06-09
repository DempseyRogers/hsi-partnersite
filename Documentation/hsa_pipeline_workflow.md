# HSA Pipeline
Here we will investigate how model's objects are transformed from preprocessed_df to inference. 

## class HSA_pipeline()

```
class HSA_pipeline():
    def __init__(
        self,
        bin_count: int,
        min_additional_percent_variance_exp: int,
        max_spawn_dummies: int,
        anomaly_std_tolerance: float,
        cutoff_distance: float,
        lr: float,
        penalty_ratio: float,
        percent_variance_explained: float,
        base_directory: str,
        logger: loguru_logger,
        logging_level: str,
        multi_filters: int = 15,  
        affinity_matrix_iterations: int = 20,  
        batch_size: int = 2000, 
        iterations: int = int(5e3),  
        num_workers: int = 10,
        converge_toll: float = 1e-30,  
        plot_figures: bool = False,  
        save_figures: bool = False,
        save_preprocessed_np: bool = False,
        verbose: bool = False,
        ):
```
### Parameter definitions and intended uses:
- __bin_count__: Multifilter threshold for anomalous prediction 
- __min_additional_percent_variance_exp__: Minimum percent variance contribution for feature retention in principal component analysis
- __max_spawn_dummies__: Maximum acceptable spawned features from pd.get_dummies()
- __cutoff_distance__: Model hyperparameter parameterizing both similarity and density calculations 
- __lr__: torch.optim.adam() learning rate used in optimization
- __penalty_ratio__: Decay rate of contributions from topographical scales in penalized objective function
- __percent_variance_explained__: Stopping tolerance for adding additional features in principal component analysis
- __base_directory__: Storage location for model outputs__: results/, logs/, and plots/
- __logger__: Loguru logger with default logging configuration
- __logging_level__: Define loguru.logger level
- __affinity_matrix_iterations__: Defines the number of topologies used in penalized objective function. 
- __batch_size__: Batch size for data loader 
- __iterations__:  torch.optim.adam() iterations used in optimization
- __multi_filters__: Number of multifilters to complete 
- __num_workers__: Number of available workers
- __anomaly_std_tolerance__: Hyperparameter defining model sensitivity for selection of first rank anomalies and increases in anomaly_prediction_frequency_df
- __converge_toll__: Convergence tolerance of anomaly score vector during optimization
- __plot_figures__: Show weights and model prediction heatmaps
- __save_figures__: Determines if figures are saved in {base_directory}/plots/
- __save_preprocessed_np__: Determines if preprocessed_np is saved in {base_directory}/results/
- __verbose__: Determines if logged statements are printed in terminal

        if_preprocess: bool = True
## attribute HSA_pipeline.pipeline()
```
def pipeline(
        self,
        df: pd.DataFrame = None,
        if_preprocess: bool = True,
        ):
```
### Inputs:
- __df__: Model's input data. 
- __if_preprocess__: If true sklearn.preprocessing.StandardScaler() and sklearn.decomposition.PCA() are applied to df. Requires: __min_additional_percent_variance_exp__, __max_spawn_dummies__, and __percent_variance_explained__.

### Output Objects:


