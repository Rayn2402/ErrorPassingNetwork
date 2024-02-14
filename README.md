# Error Passing Network (EPN)
This repository stores the code implemented to generate the results of the paper:  
*Development of Error Passing Network for Optimizing the Prediction of VO2 peak in Childhood Acute Leukemia Survivors*

**The datasets analysed during the study are not publicly available for confidentiality purposes.
However, randomly generated datasets with the same format as used in our experiments are publicly
shared in the** ```data``` **directory**.

## Installation
To have all the requirements needed, you must do the following actions:
- Open a terminal
- Clone this repo: ```git clone git@github.com:Rayn2402/ErrorPassingNetwork.git```
- Move into the directory: ```cd ErrorPassingNetwork/```
- Create a virtual environment with conda: ```conda env create --file settings/env.yml```
- Install Pytorch [library](https://pytorch.org/get-started/locally/) according to your hardware requirements
  - Select the latest stable ```Pytorch Build```
  - Select the appropriate version for ```Your OS```
  - Select ```Conda``` as the ```Package```
  - Select ```Python``` as the ```Language```
  - Select ```Default``` as the ```Compute Platform```
  - Copy and paste the command provided, but remove ```torchvision``` and ```torchaudio```.
- Activate your new environment: ```conda activate epn```
  
## Test the implementation
You can write the following lines in a terminal to replicate the experiments of the manuscript 
using the **randomly generated** data stored in the ```data``` directory. Records of the experiments 
will be stored in ```records/experiments``` directory as they will be completed. For the manuscript,
```--nb_trials``` was set to ```500```. However, here, we set it to ```50``` to reduce the execution
time of test runs.


### Experiments **with** walk variables

#### - Labonté equation
```time python scripts/experiments/original_equation.py --from_csv```

#### - Labonté equation + EPN
```
python scripts/experiments/model_evaluations.py \
--from_csv \
--remove_sex_variable \
--epn \
--path records/experiments/labonte/ \
--additional_tag labonte \
--nb_trials 50
```
### Experiments **without** walk variables

#### - Linear regression, Random Forest and XGBoost 
```
python scripts/experiments/model_evaluations.py \
--from_csv \
--remove_walk_variables \
--linear \
--random_forest \
--xgboost \
--nb_trials 50
```

#### - Linear regression + EPN
```
python scripts/experiments/model_evaluations.py \
--from_csv \
--remove_walk_variables \
--epn \
--path records/experiments/LR_nw/ \
--additional_tag LR \
--nb_trials 50
```

#### - Random Forest + EPN
```
python scripts/experiments/model_evaluations.py \
--from_csv \
--remove_walk_variables \
--epn \
--path records/experiments/RF_nw/ \
--additional_tag RF \
--nb_trials 50
```

#### - XGBoost + EPN
```
python scripts/experiments/model_evaluations.py \
--from_csv \
--remove_walk_variables \
--epn \
--path records/experiments/XGBoost_nw/ \
--additional_tag XGBoost \
--nb_trials 50
```

### Summarizing all results
To summarize the results of test runs, run the command below. The output will be stored in ```records/csv/results.csv```.
```
python scripts/utils/get_scores_csv.py --path records/experiments/ --filename results
```

Specs of our computer and execution times recorded for each experiment
are displayed below.   

- Computer model:  Alienware Aurora Ryzen Edition
- Linux version: Ubuntu 20.04.4 LTS
- CPU: AMD Ryzen 9 3900X 12-Core Processor
- GPU: None were used for our experiments

| Experiment (with walk variables) | Time  |
|----------------------------------|-------|
| Labonté                          | 3s    |
| Labonté + EPN                    | 3m12s |

| Experiment (without walk variables) | Time  |
|-------------------------------------|-------|
| Linear Regression                   | 6s    |
| Random Forest                       | 5m55s |
| XGBoost                             | 17s   |
| Linear Regression + EPN             | 3m10s |
| Random Forest + EPN                 | 3m17s |
| XGBoost + EPN                       | 3m29s |


## Project Tree
```
├── checkpoints                   <- Temporary state dictionaries save by the EarlyStopper module
├── data
│   └── vo2_dataset.csv           <- Synthetic dataset for the VO2 peak prediction task
|
├── hps                           <- Python files used to store sets of hyperparameter values and search spaces
├── masks                         <- JSON files used to store random stratified sampling masks
├── models                        <- State dictionaries associated to the best models
├── records                       <- Directories in which results and summaries of data analyses are stored
|
├── scripts
│   ├── experiments               <- Scripts to run individual experiments
│   ├── post_analyses             <- Scripts to run post analyses
│   └── utils                     <- Scripts to execute different subtasks
|
├── settings                      <- Files used for the setup of the project environment
|
├── src                           <- All project modules
│   ├── data
│   │   ├── extraction            <- Modules related to data extraction from PostgreSQL
│   │   └── processing            <- Modules related to data processing
│   ├── evaluation                <- Modules related to the evaluation and tuning of the models
│   ├── models
│   │   ├── abstract_models       <- Abstract classes from which new models have to inherit
│   │   ├── blocks                <- Neural network architecture blocks
│   │   └── wrappers              <- Abstract classes used to wrap existing models
│   ├── recording                 <- Recording module
│   └── utils                     <- Modules associated to visualization, metrics, hps and more
└── README.md
```