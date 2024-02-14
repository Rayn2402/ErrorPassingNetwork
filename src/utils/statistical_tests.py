"""
Description: This file contains all functions related to statistical tests
"""
import json
import os

import numpy as np
from scipy.stats import ttest_ind, spearmanr
from sklearn.metrics import mean_absolute_percentage_error as mape, mean_absolute_error as mae
from tqdm import tqdm
import pandas as pd


def compare_distributions_biological_sex(dataset: pd.DataFrame,
                                         column_comparison: str = '34500 Sex',
                                         ) -> None:
    """
    Compares the distributions of males and females in the dataset for each predictor
    by reporting means, stds and a p-value.

    Args:
        dataset: (N, M) all samples with their values for each predictor.
        column_comparison: biological_sex column

    Returns: None
    """
    df_statistics = pd.DataFrame(columns=['Variable name',
                                          'Survivors',
                                          'Female survivors',
                                          'Male survivors',
                                          'T-test p-value'])

    df_statistics.loc[len(df_statistics)] = ['Total (n)',
                                             len(dataset),
                                             len(dataset[dataset[column_comparison] == 0]),
                                             len(dataset[dataset[column_comparison] == 1]),
                                             '--'
                                             '--']

    for column in dataset.columns:
        if column not in [column_comparison, 'Participant']:
            # Get male and female survivors
            male_data = dataset[(dataset[column_comparison] == 1) & (dataset[column].notna())][column]
            female_data = dataset[(dataset[column_comparison] == 0) & (dataset[column].notna())][column]
            # Get the mean and standard deviation
            mean_col = np.round([dataset[column].mean(), female_data.mean(), male_data.mean()], 2)
            std_col = np.round([dataset[column].std(), female_data.std(), male_data.std()], 2)
            # Get the p-values
            _, p_value = ttest_ind(male_data, female_data, equal_var=False)
            # Record the values
            p_value = round(p_value, 3) if p_value >= 0.001 else '<0.001'
            df_statistics.loc[len(df_statistics)] = [column,
                                                     f'{mean_col[0]}+-{std_col[0]}',
                                                     f'{mean_col[1]}+-{std_col[1]}',
                                                     f'{mean_col[2]}+-{std_col[2]}',
                                                     p_value]


def bootstrapping_p_values(model: str,
                           target_predictions: pd.DataFrame,
                           saving_dir: str,
                           file_name: str,
                           nboot: int = 100000
                           ) -> None:
    """
    Creates a histogram plot with mean and standard deviations
    of a model and model+EPN residuals

    Args:
        model: baseline model name. ex: Labonté. model must be a column in target_predictions
        target_predictions: dataframe with targets and predictions of each sample in the dataset
        saving_dir: path to the directory where to save the figure
        file_name: name of the file in which the figure is saved
        nboot: number of bootstrappings

    Returns: None
    """
    document = {}
    for i in range(5):
        scores_dic = {'MAPE': {},
                      'MAE': {},
                      'SPR': {}}
        # Get the targets and the scores
        targets = target_predictions[target_predictions['Fold'] == i]['target'].to_numpy()
        n_patients = targets.shape[0]
        scores = target_predictions[target_predictions['Fold'] == i][model].to_numpy()
        scores_epn = target_predictions[target_predictions['Fold'] == i][model + '+EPN'].to_numpy()

        # Get primary score
        scores_dic['MAPE']['primary'] = mape(targets, scores_epn) - mape(targets, scores)
        scores_dic['MAE']['primary'] = mae(targets, scores_epn) - mae(targets, scores)
        scores_dic['SPR']['primary'] = spearmanr(targets, scores_epn).statistic - spearmanr(targets, scores).statistic

        # Combine both scores
        combined_scores = np.concatenate((scores_epn, scores))
        combined_targets = np.concatenate((targets, targets))

        # Initialize
        scores_dic['MAPE']['sampled'] = []
        scores_dic['MAE']['sampled'] = []
        scores_dic['SPR']['sampled'] = []

        # Bootstrapping
        with tqdm(total=nboot) as bar:
            for boot in range(nboot):
                np.random.seed(boot + i)
                sampled_indexes = np.random.choice(combined_scores.shape[0], size=combined_scores.shape[0],
                                                   replace=True)
                sampled_scores = combined_scores[sampled_indexes]
                sampled_targets = combined_targets[sampled_indexes]
                sampled_scores_epn, targets_scores_epn = sampled_scores[:n_patients], sampled_targets[:n_patients]
                sampled_scores, targets_scores = sampled_scores[n_patients:], sampled_targets[n_patients:]

                scores_dic['MAPE']['sampled'].append(
                    mape(targets_scores_epn, sampled_scores_epn) - mape(targets_scores, sampled_scores))
                scores_dic['MAE']['sampled'].append(
                    mae(targets_scores_epn, sampled_scores_epn) - mae(targets_scores, sampled_scores))
                scores_dic['SPR']['sampled'].append(
                    spearmanr(targets_scores_epn, sampled_scores_epn).statistic -
                    spearmanr(targets_scores, sampled_scores).statistic)
                bar.update()

        # compute p-values
        document[str(i)] = {}
        for metric in scores_dic.keys():
            document[str(i)][metric] = {}
            sampled_diff = scores_dic[metric]['sampled']
            primary_difference = scores_dic[metric]['primary']
            document[str(i)][metric]['primary_difference'] = primary_difference
            if primary_difference > 0:
                p_value = np.sum(sampled_diff >= primary_difference) / nboot
            else:
                p_value = np.sum(sampled_diff <= primary_difference) / nboot

            document[str(i)][metric]['p_value'] = p_value

    with open(os.path.join(saving_dir, f'{file_name}.json'), "w") as file:
        json.dump(document, file, indent=True)
