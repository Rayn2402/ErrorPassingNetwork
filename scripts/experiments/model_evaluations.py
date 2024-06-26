"""
Description: This file is used to execute all the model comparisons
             made on the VO2 peak dataset
"""
import sys
import time

from argparse import ArgumentParser
from os.path import dirname, realpath
from copy import deepcopy

def retrieve_arguments():
    """
    Creates a parser for VO2 peak prediction experiments
    """
    # Create a parser
    parser = ArgumentParser(usage='\n python [experiment file].py',
                            description="Runs the experiments associated to the VO2 dataset.")

    # Nb of inner splits and nb of outer splits
    parser.add_argument('-k', '--nb_outer_splits', type=int, default=5,
                        help='Number of outer splits used during the models evaluations.')
    parser.add_argument('-l', '--nb_inner_splits', type=int, default=5,
                        help='Number of inner splits used during the models evaluations.')

    # Data source
    parser.add_argument('-from_csv', '--from_csv', default=False, action='store_true',
                        help='If true, extract the data from the csv file instead of the database.')

    # Feature selection
    parser.add_argument('-r_w', '--remove_walk_variables', default=False, action='store_true',
                        help='If true, removes the six-minute walk test variables from the data.')
    parser.add_argument('-r_s', '--remove_sex_variable', default=False, action='store_true',
                        help='If true, removes the biological sex variable from the data.')
    parser.add_argument('-f', '--feature_selection', default=False, action='store_true',
                        help='If true, applies automatic feature selection')

    # Models selection
    parser.add_argument('-enet', '--enet', default=False, action='store_true',
                        help='If true, runs enet experiment')
    parser.add_argument('-linear', '--linear', default=False, action='store_true',
                        help='If true, runs linear regression experiment')
    parser.add_argument('-mlp', '--mlp', default=False, action='store_true',
                        help='If true, runs mlp experiment')
    parser.add_argument('-rf', '--random_forest', default=False, action='store_true',
                        help='If true, runs random forest experiment')
    parser.add_argument('-xg', '--xgboost', default=False, action='store_true',
                        help='If true, runs xgboost experiment')
    parser.add_argument('-epn', '--epn', default=False, action='store_true',
                        help='If true, runs Error Passing Network experiment')
    parser.add_argument('-gat', '--gat', default=False, action='store_true',
                        help='If true, runs Graph Attention Network experiment')
    parser.add_argument('-gcn', '--gcn', default=False, action='store_true',
                        help='If true, runs Graph Convolutional Network experiment')

    # Training parameters
    parser.add_argument('-epochs', '--epochs', type=int, default=100,
                        help='Maximal number of epochs during training')
    parser.add_argument('-kernel', '--kernel', type=str, default='attention',
                        choices=['attention', 'dot_product', 'cosine'],
                        help='Similarity kernel to use for connecting patients')
    parser.add_argument('-patience', '--patience', type=int, default=10,
                        help='Number of epochs allowed without improvement (for early stopping)')
    parser.add_argument('-nb_trials', '--nb_trials', type=int, default=500,
                        help='Number of hyperparameter sets sampled during hyperparameter optimization')

    # Graph construction parameters
    parser.add_argument('-w_sim', '--weighted_similarity', default=False, action='store_true',
                        help='If true, calculates patients similarities using weighted metrics')
    parser.add_argument('-cond_col', '--conditional_column', default=False, action='store_true',
                        help='If true, uses the sex as a conditional column in graph construction')
    parser.add_argument('-deg', '--degree', nargs='*', type=str, default=[7],
                        help="Maximum number of in-degrees for each node in the graph")

    # Activation of sharpness-aware minimization
    parser.add_argument('-rho', '--rho', type=float, default=0,
                        help='Rho parameter of Sharpness-Aware Minimization (SAM) Optimizer.'
                             'If >0, SAM is enabled')

    # Usage of predictions from another experiment
    parser.add_argument('-p', '--path', type=str, default=None,
                        help='Path leading to predictions of another model')

    # Seed
    parser.add_argument('-seed', '--seed', type=int, default=1010710,
                        help='Seed used during model evaluations')

    # Additional experiment tag
    parser.add_argument('-tag', '--additional_tag', type=str, default=None,
                        help='String that can be added to identify experiment folder')

    arguments = parser.parse_args()

    # Print arguments
    print("\nThe inputs are:")
    for arg in vars(arguments):
        print(f"{arg}: {getattr(arguments, arg)}")
    print("\n")

    return arguments

if __name__ == '__main__':

    # Imports specific to project
    sys.path.append((dirname(dirname(dirname(realpath(__file__))))))
    from hps import search_spaces as ss
    from settings.paths import Paths
    from src.data.extraction.constants import *
    from src.data.extraction.data_management import PetaleDataManager
    from src.data.processing.datasets import PetaleDataset
    from src.data.processing.gnn_datasets import PetaleKGNNDataset
    from src.data.processing.feature_selection import FeatureSelector
    from src.data.processing.sampling import extract_masks, get_VO2_data, push_valid_to_train
    from src.evaluation.evaluation import Evaluator
    from src.models.epn import PetaleEPN, EPNHP
    from src.models.gat import PetaleGATR, GATHP
    from src.models.gcn import PetaleGCNR, GCNHP
    from src.models.linear_regression import PetaleLR
    from src.models.mlp import PetaleMLPR, MLPHP
    from src.models.random_forest import PetaleRFR
    from src.models.xgboost_ import PetaleXGBR
    from src.utils.hyperparameters import Range
    from src.utils.metrics import AbsoluteError, MeanAbsolutePercentageError, RootMeanSquaredError, SpearmanR

    # Arguments parsing
    args = retrieve_arguments()

    # Initialization of a data manager
    manager = PetaleDataManager() if not args.from_csv else None

    # We extract needed data
    df, target, cont_cols, cat_cols = get_VO2_data(manager)

    # We filter baselines variables if needed
    if args.remove_walk_variables:
        df.drop([TDM6_HR_END, TDM6_DIST], axis=1, inplace=True)
        cont_cols = [c for c in cont_cols if c not in [TDM6_HR_END, TDM6_DIST]]
    if args.remove_sex_variable:
        df.drop([SEX], axis=1, inplace=True)
        cat_cols.remove(SEX)
        if len(cat_cols) == 0:
            cat_cols = None

    # Extraction of masks
    masks = extract_masks(Paths.VO2_MASK, k=args.nb_outer_splits, l=args.nb_inner_splits)

    # Creation of masks for tree-based models
    masks_without_val = deepcopy(masks)
    push_valid_to_train(masks_without_val)

    # Initialization of the dictionary containing the evaluation metrics
    evaluation_metrics = [AbsoluteError(), MeanAbsolutePercentageError(), SpearmanR(), RootMeanSquaredError()]

    # Initialization of a feature selector
    if args.feature_selection:
        feature_selector = FeatureSelector(threshold=[0.01], cumulative_imp=[False], seed=args.seed)
    else:
        feature_selector = None

    # We save the string that will help identify evaluations
    eval_id = ""
    if args.remove_walk_variables:
        eval_id += "_nw"
    if args.remove_sex_variable:
        eval_id += "_ns"
    if args.rho > 0:
        eval_id += "_sam"
        sam_search_space = {Range.MIN: 0, Range.MAX: args.rho}  # Sharpness-Aware Minimization search space
    else:
        sam_search_space = {Range.VALUE: 0}

    if args.additional_tag is not None:
        eval_id = f"-{args.additional_tag}{eval_id}"

    # We start a timer for the whole experiment
    first_start = time.time()

    """
    Linear regression experiment
    """
    if args.linear:

        # Start timer
        start = time.time()

        # Creation of the dataset
        dataset = PetaleDataset(df, target, cont_cols, cat_cols, classification=False)

        # Creation of the evaluator
        evaluator = Evaluator(model_constructor=PetaleLR,
                              dataset=dataset,
                              masks=masks_without_val,
                              evaluation_name=f"LR{eval_id}",
                              hps={},
                              n_trials=0, # This model has no hyperparameters
                              evaluation_metrics=evaluation_metrics,
                              feature_selector=feature_selector,
                              fixed_params={},
                              save_hps_importance=True,
                              save_optimization_history=True,
                              seed=args.seed,
                              pred_path=args.path)

        # Evaluation
        evaluator.evaluate()

        print(f"Time taken for linear regression (min): {(time.time() - start) / 60:.2f}")

    """
    Random Forest experiment
    """
    if args.random_forest:

        # Start timer
        start = time.time()

        # Creation of dataset
        dataset = PetaleDataset(df, target, cont_cols, cat_cols, classification=False)

        # Creation of the evaluator
        evaluator = Evaluator(model_constructor=PetaleRFR,
                              dataset=dataset,
                              masks=masks_without_val,
                              evaluation_name=f"RF{eval_id}",
                              hps=ss.RF_HPS,
                              n_trials=args.nb_trials,
                              evaluation_metrics=evaluation_metrics,
                              feature_selector=feature_selector,
                              save_hps_importance=True,
                              save_optimization_history=True,
                              seed=args.seed,
                              pred_path=args.path)

        # Evaluation
        evaluator.evaluate()

        print(f"Time taken for Random Forest (minutes): {(time.time() - start) / 60:.2f}")

    """
    XGBoost experiment
    """
    if args.xgboost:

        # Start timer
        start = time.time()

        # Creation of dataset
        dataset = PetaleDataset(df, target, cont_cols, cat_cols, classification=False)

        # Creation of the evaluator
        evaluator = Evaluator(model_constructor=PetaleXGBR,
                              dataset=dataset,
                              masks=masks_without_val,
                              evaluation_name=f"XGBoost{eval_id}",
                              hps=ss.XGBOOST_HPS,
                              n_trials=args.nb_trials,
                              evaluation_metrics=evaluation_metrics,
                              feature_selector=feature_selector,
                              save_hps_importance=True,
                              save_optimization_history=True,
                              seed=args.seed,
                              pred_path=args.path)

        # Evaluation
        evaluator.evaluate()

        print(f"Time taken for XGBoost (minutes): {(time.time() - start) / 60:.2f}")

    """
    MLP experiment
    """
    if args.mlp:

        # Start timer
        start = time.time()

        # Creation of the dataset
        dataset = PetaleDataset(df, target, cont_cols, cat_cols, to_tensor=True, classification=False)

        # Creation of a function to update fixed params
        def update_fixed_params(dts):
            return {'max_epochs': args.epochs,
                    'patience': args.patience,
                    'num_cont_col': len(dts.cont_idx),
                    'cat_idx': dts.cat_idx,
                    'cat_sizes': dts.cat_sizes,
                    'cat_emb_sizes': dts.cat_sizes}

        # Saving of the fixed params of MLP
        fixed_params = update_fixed_params(dataset)

        # Update of the hyperparameters
        ss.MLP_HPS[MLPHP.RHO.name] = sam_search_space
        cat_sizes_sum = sum(dataset.cat_sizes) if dataset.cat_sizes is not None else 0
        ss.MLP_HPS[MLPHP.N_UNIT.name] = {Range.VALUE: int((len(cont_cols) + cat_sizes_sum)/2)}

        # Creation of the evaluator
        evaluator = Evaluator(model_constructor=PetaleMLPR,
                              dataset=dataset,
                              masks=masks,
                              evaluation_name=f"MLP{eval_id}",
                              hps=ss.MLP_HPS,
                              n_trials=args.nb_trials,
                              evaluation_metrics=evaluation_metrics,
                              feature_selector=feature_selector,
                              fixed_params=fixed_params,
                              fixed_params_update_function=update_fixed_params,
                              save_hps_importance=True,
                              save_optimization_history=True,
                              seed=args.seed,
                              pred_path=args.path)

        # Evaluation
        evaluator.evaluate()

        print(f"Time taken for MLP (minutes): {(time.time() - start) / 60:.2f}")

    """
    ENET experiment
    """
    if args.enet:

        # Start timer
        start = time.time()

        # Creation of the dataset
        dataset = PetaleDataset(df, target, cont_cols, cat_cols, to_tensor=True, classification=False)

        # Creation of a function to update fixed params
        def update_fixed_params(dts):
            return {'max_epochs': args.epochs,
                    'patience': args.patience,
                    'num_cont_col': len(dts.cont_idx),
                    'cat_idx': dts.cat_idx,
                    'cat_sizes': dts.cat_sizes,
                    'cat_emb_sizes': dts.cat_sizes}

        # Saving of the fixed params of ENET
        fixed_params = update_fixed_params(dataset)

        # Update of the hyperparameters
        ss.ENET_HPS[MLPHP.RHO.name] = sam_search_space

        # Creation of the evaluator
        evaluator = Evaluator(model_constructor=PetaleMLPR,
                              dataset=dataset,
                              masks=masks,
                              evaluation_name=f"Enet{eval_id}",
                              hps=ss.ENET_HPS,
                              n_trials=args.nb_trials,
                              evaluation_metrics=evaluation_metrics,
                              feature_selector=feature_selector,
                              fixed_params=fixed_params,
                              fixed_params_update_function=update_fixed_params,
                              save_hps_importance=True,
                              save_optimization_history=True,
                              seed=args.seed,
                              pred_path=args.path)

        # Evaluation
        evaluator.evaluate()

        print(f"Time taken for Enet (minutes): {(time.time() - start) / 60:.2f}")

    """
    EPN experiment
    """
    if args.epn and (args.path is not None):
        # Start timer
        start = time.time()

        # Creation of the dataset
        dataset = PetaleDataset(df, target, cont_cols, cat_cols, to_tensor=True, classification=False)

        # Creation of a function to update fixed params
        def update_fixed_params(dts):

            fp = {'num_cont_col': len(dts.cont_idx),
                  'previous_pred_idx': len(dts.cont_idx) - 1,
                  'pred_mu': 0,
                  'pred_std': 1,
                  'cat_idx': dts.cat_idx,
                  'cat_sizes': dts.cat_sizes,
                  'cat_emb_sizes': dts.cat_sizes,
                  'max_epochs': args.epochs,
                  'similarity_kernel': args.kernel,
                  'patience': args.patience}

            if 'pred0' in dts.original_data.columns:
                mu, std, _ = dts.current_train_stats()
                fp['pred_mu'] = mu.loc['pred0']
                fp['pred_std'] = std.loc['pred0']

            return fp

        # Saving of the fixed params of GAS
        fixed_params = update_fixed_params(dataset)

        # Update of the hyperparameters
        ss.EPNHPS[EPNHP.RHO.name] = sam_search_space

        # Set the number of neighbors to maximum if attention-based similarity
        if args.kernel == 'attention':
            ss.EPNHPS['n_neighbors'] = {'value': None}

        # Creation of the evaluator
        evaluator = Evaluator(model_constructor=PetaleEPN,
                              dataset=dataset,
                              masks=masks,
                              evaluation_name=f"EPN{eval_id}",
                              hps=ss.EPNHPS,
                              n_trials=args.nb_trials,
                              evaluation_metrics=evaluation_metrics,
                              feature_selector=feature_selector,
                              fixed_params=fixed_params,
                              fixed_params_update_function=update_fixed_params,
                              save_hps_importance=True,
                              save_optimization_history=True,
                              seed=args.seed,
                              pred_path=args.path)

        # Evaluation
        evaluator.evaluate()

        print(f"Time taken for EPN (minutes): {(time.time() - start) / 60:.2f}")

    """
    GAT experiment
    """
    if args.gat:

        # Start timer
        start = time.time()

        for nb_neighbor in args.degree:

            # We change the type from str to int
            nb_neigh = int(nb_neighbor)

            # We set the conditional column
            cond_cat_col = SEX if args.conditional_column else None

            # We set the distance computations options
            GAT_options = [("", False)] if not args.weighted_similarity else [("", False), ("w", True)]

            for prefix, w_sim in GAT_options:

                # Creation of the dataset
                dataset = PetaleKGNNDataset(df, target, k=nb_neigh,
                                            weighted_similarity=w_sim,
                                            cont_cols=cont_cols, cat_cols=cat_cols,
                                            conditional_cat_col=cond_cat_col, classification=False)

                # Update of hyperparameter
                cat_sizes_sum = sum(dataset.cat_sizes) if dataset.cat_sizes is not None else 0
                ss.GATHPS[GATHP.HIDDEN_SIZE.name] = {Range.VALUE: int((len(cont_cols) + cat_sizes_sum)/2)}

                # Creation of a function to update fixed params
                def update_fixed_params(dts):
                    return {'num_cont_col': len(dts.cont_idx),
                            'cat_idx': dts.cat_idx,
                            'cat_sizes': dts.cat_sizes,
                            'cat_emb_sizes': dts.cat_sizes,
                            'max_epochs': args.epochs,
                            'patience': args.patience}

                # Saving of the fixed params pf GAT
                fixed_params = update_fixed_params(dataset)

                # Update of the hyperparameters
                ss.GATHPS[GATHP.RHO.name] = sam_search_space

                # Creation of the evaluator
                evaluator = Evaluator(model_constructor=PetaleGATR,
                                      dataset=dataset,
                                      masks=masks,
                                      evaluation_name=f"{prefix}GAT{nb_neighbor}{eval_id}",
                                      hps=ss.GATHPS,
                                      n_trials=args.nb_trials,
                                      evaluation_metrics=evaluation_metrics,
                                      fixed_params=fixed_params,
                                      fixed_params_update_function=update_fixed_params,
                                      feature_selector=feature_selector,
                                      save_hps_importance=True,
                                      save_optimization_history=True,
                                      seed=args.seed,
                                      pred_path=args.path)

                # Evaluation
                evaluator.evaluate()

        print(f"Time taken for GAT (minutes): {(time.time() - start) / 60:.2f}")

    """
    GCN experiment
    """
    if args.gcn:

        # Start timer
        start = time.time()

        for nb_neighbor in args.degree:

            # We change the type from str to int
            nb_neigh = int(nb_neighbor)

            # We set the conditional column
            cond_cat_col = SEX if args.conditional_column else None

            # We set the distance computations options
            GCN_options = [("", False)] if not args.weighted_similarity else [("", False), ("w", True)]

            for prefix, w_sim in GCN_options:

                # Creation of the dataset
                dataset = PetaleKGNNDataset(df, target, k=nb_neigh,
                                            weighted_similarity=w_sim,
                                            cont_cols=cont_cols, cat_cols=cat_cols,
                                            conditional_cat_col=cond_cat_col, classification=False)

                # Update of hyperparameter
                cat_sizes_sum = sum(dataset.cat_sizes) if dataset.cat_sizes is not None else 0
                ss.GCNHPS[GCNHP.HIDDEN_SIZE.name] = {Range.VALUE: int((len(cont_cols) + cat_sizes_sum)/2)}

                # Creation of a function to update fixed params
                def update_fixed_params(dts):
                    return {'num_cont_col': len(dts.cont_idx),
                            'cat_idx': dts.cat_idx,
                            'cat_sizes': dts.cat_sizes,
                            'cat_emb_sizes': dts.cat_sizes,
                            'max_epochs': args.epochs,
                            'patience': args.patience}

                # Saving of the fixed params of GCN
                fixed_params = update_fixed_params(dataset)

                # Update of the hyperparameters
                ss.GCNHPS[GCNHP.RHO.name] = sam_search_space

                # Creation of the evaluator
                evaluator = Evaluator(model_constructor=PetaleGCNR,
                                      dataset=dataset,
                                      masks=masks,
                                      evaluation_name=f"{prefix}GCN{nb_neighbor}{eval_id}",
                                      hps=ss.GCNHPS,
                                      n_trials=args.nb_trials,
                                      evaluation_metrics=evaluation_metrics,
                                      fixed_params=fixed_params,
                                      fixed_params_update_function=update_fixed_params,
                                      feature_selector=feature_selector,
                                      save_hps_importance=True,
                                      save_optimization_history=True,
                                      seed=args.seed,
                                      pred_path=args.path)

                # Evaluation
                evaluator.evaluate()

        print(f"Time taken for GCN (minutes): {(time.time() - start) / 60:.2f}")

    print(f"Overall time (minutes): {(time.time() - first_start) / 60:.2f}")
