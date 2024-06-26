"""
Description: Script used to retrieve predictions from multiple records file
"""

import sys
from argparse import ArgumentParser
from os.path import dirname, realpath

# Imports specific to project
sys.path.append(dirname(dirname(dirname(realpath(__file__)))))
from src.utils.results_analyses import get_all_fold_predictions


def paths_and_ids_parser():
    """
    Provides an argparser that retrieves multiple paths, ids identifying them
    and a filename in which the predictions will be stored
    """
    # Create a parser
    parser = ArgumentParser(usage='\n python get_predictions_csv.py -p [path1 path2 ... ] -ids [id1 id2 ... ]',
                            description="Stores multiple path")

    parser.add_argument('-p', '--paths', nargs='*', type=str, help='List of paths')
    parser.add_argument('-ids', '--ids', nargs='*', type=str, help='List of ids associated to the paths')
    parser.add_argument('-fn', '--filename', type=str, default='predictions',
                        help='Name of the file in which the predictions will be stored')

    arguments = parser.parse_args()

    return arguments


if __name__ == '__main__':

    # We retrieve paths and their ids
    args = paths_and_ids_parser()

    # We retrieve the predictions
    get_all_fold_predictions(paths=args.paths, model_ids=args.ids, filename=args.filename)
