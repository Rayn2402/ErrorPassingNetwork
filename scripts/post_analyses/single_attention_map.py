"""
Description: Script used to extract attention between patient 057 and other
             patients from the training set. An attention map is also
             generated.
"""

import sys
from json import load as jsload
from matplotlib import pyplot as plt
from os.path import dirname, join, realpath
from pandas import DataFrame
from seaborn import heatmap
from torch import load, topk
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

    # 1. Set the number of the data split for which P057 is in the test set
    SPLIT: int = 1

    # 2. Load the data
    df, target, cont_cols, cat_cols = get_VO2_data()

    # 3. Remove the sex from the categorical columns
    # df.drop([SEX], axis=1, inplace=True)
    cat_cols = None

    # 4. Create the dataset
    dts = PetaleDataset(df, target, cont_cols, cat_cols, to_tensor=True)

    # 5. Add the predictions of the past model as a variable

    # 5.0 Load the predictions
    pred_path = join(Paths.EXPERIMENTS_RECORDS, 'with_walk', 'labonte')
    with open(join(pred_path, f"Split_{SPLIT}", Recorder.RECORDS_FILE), "r") as read_file:
        data = jsload(read_file)

    # 5.1 Create the conversion function to extract predictions from strings
    def convert(x: str) -> List[float]:
        return [float(x)]

    # 5.2 Extract the predictions
    pred = {}
    for section in [Recorder.TRAIN_RESULTS, Recorder.TEST_RESULTS, Recorder.VALID_RESULTS]:
        if section in data.keys():
            pred = {**pred, **{p_id: [p_id, *convert(v[Recorder.PREDICTION])] for p_id, v in data[section].items()}}

    # 5.3 Creation a pandas dataframe
    df_temp = DataFrame.from_dict(pred, orient='index', columns=[PARTICIPANT, 'pred0'])

    # 6. Create the new augmented dataset
    dts = dts.create_superset(data=df_temp, categorical=False)

    # 7. Extract the masks
    masks = extract_masks(Paths.VO2_MASK, k=3, l=0)
    test_mask = masks[SPLIT][MaskType.TEST]
    train_mask = masks[SPLIT][MaskType.TRAIN]

    # 8. Order patient in the masks according to their biological sex
    test_mask.sort(key=lambda x: df.iloc[x][SEX])
    train_mask.sort(key=lambda x: df.iloc[x][SEX])

    # 9. Set the masks
    dts.update_masks(train_mask=train_mask,
                     valid_mask=masks[SPLIT][MaskType.VALID],
                     test_mask=test_mask)

    # 10. Locate patient P057 in the dataset
    row_idx = dts.ids_to_row_idx['P057']

    # 11. Locate the patient in the test mask
    mask_pos = test_mask.index(row_idx)

    # 12. Create the model
    mu, std, _ = dts.current_train_stats()
    epn_wrapper = PetaleEPN(previous_pred_idx=len(dts.cont_idx) - 1,
                            pred_mu=mu.loc['pred0'],
                            pred_std=std.loc['pred0'],
                            num_cont_col=len(dts.cont_idx),
                            cat_idx=dts.cat_idx,
                            cat_sizes=dts.cat_sizes,
                            cat_emb_sizes=dts.cat_sizes)

    # 13. Load the parameters of the model
    path = join(Paths.EXPERIMENTS_RECORDS, 'with_walk',
                'EPN-L', f'Split_{SPLIT}', 'torch_model.pt')

    epn_wrapper.model.load_state_dict(load(path))

    # 14. Execute the forward pass and load the attention scores
    y = epn_wrapper.predict(dts)
    attn = epn_wrapper.model.attn_cache

    # 15. Extract the row associated to P057 and
    # identify the 10 patients with the highest attention scores
    top10_attn, pos_idx = topk(attn[mask_pos], k=10)

    # 16. Identify their original position in the dataset
    batch_idx = dts.train_mask + dts.test_mask
    idx = [batch_idx[i] for i in list(pos_idx)]

    # 17. Extract patient ids
    print(f'Patients: {[dts.ids[i] for i in idx]}')
    print(f'Attn scores: {top10_attn}')

    # 18. Remove attention scores associated to test elements
    attn = attn[:, :-len(test_mask)]

    # 19. Count the number of patients for which the sex is unknown
    unknown_sex = df[SEX].isna().sum()
    print(f'Nb of patients with unknown sex: {unknown_sex}')

    # 20. Count the number of men and women in test and train
    nb_men_train = df.iloc[train_mask][SEX].sum()
    nb_men_test = df.iloc[test_mask][SEX].sum()
    print(f'Nb of men in training: {nb_men_train}')
    print(f'Nb of women in training: {len(train_mask) - nb_men_train}')
    print(f'Nb of men in test: {nb_men_test}')
    print(f'Nb of women in test: {len(test_mask) - nb_men_test}')

    # 21. Create a heatmap
    ax = heatmap(attn, yticklabels=False, xticklabels=False, cmap='viridis')
    plt.ylabel('Test patients')
    plt.xlabel('Training patients')
    plt.savefig('sex_attention_map.svg')
    plt.close()





