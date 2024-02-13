# Error Passing Network
This repository stores the code implemented to generate the results of the paper:  
*Development of Error Passing Network for Optimizing the Prediction of VO2 peak in Childhood Acute Leukemia Survivors*

**The datasets analysed during the study are not publicly available for confidentiality purposes.
However, randomly generated datasets with the same format as used in our experiments are publicly
shared in the** ```data``` **directory**.

## Installation
To have all the requirements needed, you must do the following actions:
- Open a terminal
- Clone this repo: ```git clone --```
- Move into the directory: ```cd --/```
- Create a virtual environment with conda: ```conda env create --file settings/env.yml```
- Activate your new environment: ```conda activate epn```
  
## Test the implementation
You can write the following lines in a terminal to replicate our experiments using the **randomly generated** data stored
in the ```data``` directory. Records of the experiments will be stored in ```records/experiments``` directory
as they will be completed.

### Labonté equation
```python scripts/experiments/original_equation.py --from_csv```

Specs of our computer and execution times recorded for each experiment
are displayed below.   

- Computer model:  Alienware Aurora Ryzen Edition
- Linux version: Ubuntu 20.04.4 LTS
- CPU: AMD Ryzen 9 3900X 12-Core Processor
- GPU: None were used for our experiments

| Experiment (with walk variables) | Time |
|----------------------------------|------|
| Labonté                          | -    |
| Labonté + EPN                    | -    |

| Experiment (without walk variables) | Time |
|-------------------------------------|------|
| Linear Regression                   | -    |
| Random Forest                       | -    |
| XGBoost                             | -    |
| Linear Regression + EPN             | -    |
| Random Forest + EPN                 | -    |
| XGBoost + EPN                       | -    |


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