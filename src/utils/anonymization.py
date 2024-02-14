"""
Description: This file contains the class which generates anonymized data
"""
from os.path import join
from typing import List, Optional

import numpy as np
from numpy import array, unique, nan, append, where
from numpy.random import normal, choice as np_choice, seed
from pandas import DataFrame, concat


class RandomAnonymizer:

    def __init__(self,
                 path_directory: str,
                 random_state: int) -> None:
        """
        Sets protected attributes and random seed

        Args:
            path_directory: path to the directory where to save the generated random data
            random_state: random seed
        """
        # Set the protected attributes
        self._path_directory = path_directory

        # Set the random state
        seed(random_state)

    def anonymize(self,
                  df: DataFrame,
                  num_cols: List[str],
                  cat_cols: List[str],
                  stratify: str,
                  file_name: Optional[str] = None) -> DataFrame:
        """
        Generate random numeric data from a gaussian distribution
        and categorical data from a categorical distributoion

        Args:
            df: original dataset
            num_cols: numeric columns of the dataset
            cat_cols: categorical columns of the dataset
            stratify: column from which to stratify the random generation
            file_name: file name to save the generated data
        """

        # Get the column according to the random generation will be done, if none is specified the whole dataset
        # is used at once
        columns_to_stratify = unique(df[stratify])

        frames = []
        for k, cat in enumerate(columns_to_stratify):
            # Get the data and number of data to generate
            dataset = df[df[stratify] == cat]
            n = len(dataset)
            stratify_col = {stratify: [cat] * n}

            # Get the mean and standard deviation of each numerical column
            mean, std = dataset[num_cols].mean().values, dataset[num_cols].std().values

            # Generate n data using a gaussian distribution for each numerical column
            num_data = normal(mean, std, size=(n, len(mean)))
            num_dict = {}

            for i, col in enumerate(num_cols):
                # Store the numeric data in a dictionary
                num_dict[col] = np.round(num_data[:, i], 2)

                while True:
                    # Get the indexes of negative values
                    negative_indexes = where(num_data[:, i] < 0.)[0].tolist()
                    if len(negative_indexes) == 0:
                        break

                    # Generate new numerical data
                    new_num_data = normal(mean[i], std[i], size=(len(negative_indexes), 1))
                    num_data[list(negative_indexes), i] = np.round(new_num_data[:, 0], 2)

            cat_dict = {}
            for col in cat_cols:
                # Get the categories appearing probabilities of each categorical column
                categories_probabilities = dataset[col].value_counts() / len(dataset)
                values, probabilities = array(categories_probabilities.index), categories_probabilities.values

                # Handle cases where there are NaN values in the dataset
                if sum(probabilities) < 1:
                    values = append(values, nan)
                    probabilities = append(probabilities, 1 - sum(probabilities))

                # Generate n data for each categorical column respecting the categories proportions in the original
                # dataset
                cat_dict[col] = np_choice(a=values, size=n, p=probabilities)

            # Join the numerical and categorical data generated
            frames.append(DataFrame.from_dict({**num_dict, **cat_dict, **stratify_col}))

        # Concatenate the datasets
        anonymous_dataset = concat(frames)
        anonymous_dataset['Participant'] = [f'P{i:0{len(str(len(df)))}}' for i in range(1, len(df) + 1)]
        anonymous_dataset = anonymous_dataset[['Participant', stratify] + cat_cols + num_cols]

        # Save the anonymous dataset generated in a csv file
        if file_name is not None:
            self._store_in_csv(anonymous_dataset, file_name)

        return anonymous_dataset

    def _store_in_csv(self,
                      dataset: DataFrame,
                      file_name: str):
        dataset.to_csv(join(self._path_directory, file_name), index=False)
