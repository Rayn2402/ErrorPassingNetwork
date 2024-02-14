"""
Description: Script used to extract attention between patient 057 and other
             patients from the training set.
"""

import sys
from json import load as jsload
from os.path import dirname, join, realpath
from pandas import DataFrame
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

    # 1. Set the number of the data split for which P057 is in the test set
    SPLIT: int = 1

    # 2. Load the data
    df, target, cont_cols, cat_cols = get_VO2_data()

    # 3. Remove the sex
    df.drop([SEX], axis=1, inplace=True)
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
    df = DataFrame.from_dict(pred, orient='index', columns=[PARTICIPANT, 'pred0'])

    # 6. Create the new augmented dataset
    dts = dts.create_superset(data=df, categorical=False)

    # 7. Extract and set the mask
    masks = extract_masks(Paths.VO2_MASK, k=3, l=0)
    test_mask = masks[SPLIT][MaskType.TEST]
    dts.update_masks(train_mask=masks[SPLIT][MaskType.TRAIN],
                     valid_mask=masks[SPLIT][MaskType.VALID],
                     test_mask=test_mask)

    # 8. Locate patient P057 in the dataset
    row_idx = dts.ids_to_row_idx['P057']

    # 9. Locate the patient in the test mask
    mask_pos = test_mask.index(row_idx)

    # 10. Create the model
    mu, std, _ = dts.current_train_stats()
    epn_wrapper = PetaleEPN(previous_pred_idx=len(dts.cont_idx) - 1,
                            pred_mu=mu.loc['pred0'],
                            pred_std=std.loc['pred0'],
                            num_cont_col=len(dts.cont_idx),
                            cat_idx=dts.cat_idx,
                            cat_sizes=dts.cat_sizes,
                            cat_emb_sizes=dts.cat_sizes)

    # 11. Load the parameters of the model
    path = join(Paths.EXPERIMENTS_RECORDS, 'with_walk',
                'EPN-L', f'Split_{SPLIT}', 'torch_model.pt')

    epn_wrapper.model.load_state_dict(load(path))

    # 12. Execute the forward pass and load the attention scores
    y = epn_wrapper.predict(dts)
    attn = epn_wrapper.model.attn_cache

    # 13. Extract the row associated to P057
    attn = attn[mask_pos]

    # 14. Identify the 10 patients with the highest attention scores
    top10_attn, pos_idx = topk(attn, k=10)

    # 15. Identify their original position in the dataset
    batch_idx = dts.train_mask + dts.test_mask
    idx = [batch_idx[i] for i in list(pos_idx)]

    # 16. Extract patient ids
    print(f'Patients: {[dts.ids[i] for i in idx]}')
    print(f'Attn scores: {top10_attn}')





