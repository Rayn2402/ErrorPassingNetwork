"""
Description: Script used to extract attention between patients of first test set
             and all patients from the training set. Patients are ordered by sex
             to illustrate attention patterns in a heatmap.
"""

import sys
from json import load as jsload

import numpy as np
from matplotlib import pyplot as plt
from os.path import dirname, join, realpath

from matplotlib.patches import Rectangle, FancyArrowPatch
from pandas import DataFrame
from seaborn import heatmap
from torch import load
from typing import List

if __name__ == '__main__':

    # Imports specific to project
    sys.path.append(dirname(dirname(dirname(realpath(__file__)))))
    from settings.paths import Paths
    from src.data.extraction.constants import PARTICIPANT, SEX
    from src.data.processing.datasets import MaskType, PetaleDataset
    from src.data.processing.sampling import extract_masks, get_VO2_data
    from src.models.epn import PetaleEPN
    from src.recording.recording import Recorder

    # Set font of matplotlib
    plt.rcParams["font.family"] = "serif"

    # Set the type of heatmap to generate
    SINGLE_ATT = 'single_attention'
    SEX_ATT = 'per_sex_attention'
    ATTENTION = SEX_ATT

    # 1. Set the number of the data split is in the test set
    SPLIT: int = 1

    # 2. Load the data
    df, target, cont_cols, cat_cols = get_VO2_data()

    # 3. Create the dataset
    dts = PetaleDataset(df, target, cont_cols, cat_cols, to_tensor=True)

    # 4. Add the predictions of the past model as a variable

    # 4.0 Load the predictions
    pred_path = join(Paths.EXPERIMENTS_RECORDS, 'with_walk', 'labonte')
    with open(join(pred_path, f"Split_{SPLIT}", Recorder.RECORDS_FILE), "r") as read_file:
        data = jsload(read_file)

    # 4.1 Create the conversion function to extract predictions from strings
    def convert(x: str) -> List[float]:
        return [float(x)]

    # 4.2 Extract the predictions
    pred = {}
    for section in [Recorder.TRAIN_RESULTS, Recorder.TEST_RESULTS, Recorder.VALID_RESULTS]:
        if section in data.keys():
            pred = {**pred, **{p_id: [p_id, *convert(v[Recorder.PREDICTION])] for p_id, v in data[section].items()}}

    # 4.3 Creation a pandas dataframe
    df_pred = DataFrame.from_dict(pred, orient='index', columns=[PARTICIPANT, 'pred0'])

    # 5. Create the new augmented dataset
    dts = dts.create_superset(data=df_pred, categorical=False)

    # 6. Extract the masks
    masks = extract_masks(Paths.VO2_MASK, k=5, l=0)
    test_mask = masks[SPLIT][MaskType.TEST]
    train_mask = masks[SPLIT][MaskType.TRAIN]

    # 7. Order patient in the masks according to their biological sex
    test_mask.sort(key=lambda x: df.iloc[x][SEX])
    train_mask.sort(key=lambda x: df.iloc[x][SEX])

    # 8. Set the masks
    dts.update_masks(train_mask=train_mask,
                     valid_mask=masks[SPLIT][MaskType.VALID],
                     test_mask=test_mask)

    # 9. Create the model
    mu, std, _ = dts.current_train_stats()
    epn_wrapper = PetaleEPN(previous_pred_idx=len(dts.cont_idx) - 1,
                            pred_mu=mu.loc['pred0'],
                            pred_std=std.loc['pred0'],
                            num_cont_col=len(dts.cont_idx),
                            cat_idx=dts.cat_idx,
                            cat_sizes=dts.cat_sizes,
                            cat_emb_sizes=dts.cat_sizes,
                            similarity_kernel='cosine',
                            )

    # 10. Load the parameters of the model
    path = join(Paths.EXPERIMENTS_RECORDS,
                'EPN-labonte_cos', f'Split_{SPLIT}', 'torch_model.pt')

    epn_wrapper.model.load_state_dict(load(path))

    # 11. Execute the forward pass and load the attention scores
    y = epn_wrapper.predict(dts)
    attn = epn_wrapper.model.attn_cache

    # 12. Remove attention scores associated to test elements
    attn = attn[:, :-len(test_mask)]

    # 13. Count the number of men in test and train
    nb_men_train = dts.x_cat[train_mask].sum()
    nb_men_test = dts.x_cat[test_mask].sum()

    # 14. Create a heatmap
    ax = heatmap(attn, yticklabels=True, xticklabels=False, cmap='viridis', linecolor='none', linewidths=0,
                 rasterized=True)

    plt.ylabel('Test patients', labelpad=15)
    plt.xlabel('Training patients', labelpad=15)

    if ATTENTION == SEX_ATT:
        ax.set_yticklabels(['' for i in test_mask])
        rect = Rectangle((0, 0),
                         len(train_mask) - nb_men_train,
                         len(test_mask) - nb_men_test,
                         linewidth=1,
                         edgecolor='red',
                         facecolor='none',
                         ls='--',
                         zorder=1000)
        ax.add_patch(rect)
        women_label = 'Women'
        ax.text(-5, (len(test_mask) - nb_men_test) / 2, women_label, ha='center', va='center', rotation='vertical')
        ax.text((len(train_mask) - nb_men_train) / 2, -1, women_label, ha='center', va='center', rotation='horizontal')
        ax.text(len(train_mask) + 2, (len(test_mask) - nb_men_test) * 1.5, 'Men', ha='center', va='center',
                rotation=270)
        ax.text((len(train_mask) - nb_men_train) * 1.5, len(test_mask) + 1, 'Men', ha='center', va='center',
                rotation='horizontal')

        rect = Rectangle((len(train_mask) - nb_men_train,
                          len(test_mask) - nb_men_test),
                         nb_men_train,
                         nb_men_test,
                         linewidth=1,
                         edgecolor='red',
                         facecolor='none',
                         ls='--',
                         zorder=1000)
        ax.add_patch(rect)
    elif ATTENTION == SINGLE_ATT:
        # Locate patient P057 in the dataset
        row_idx = dts.ids_to_row_idx['P057']

        # Locate the patient in the test mask
        mask_pos = test_mask.index(row_idx)

        ax.set_yticklabels(['' if i != row_idx else '$P_{test}$' for i in test_mask])
        rect = Rectangle((0, mask_pos), len(attn[0]), 1, linewidth=1, edgecolor='red',
                         facecolor='none', ls='--', zorder=1000)
        ax.add_patch(rect)

    plt.savefig('sex_cosine_map_with_sex.pdf')
    plt.close()
