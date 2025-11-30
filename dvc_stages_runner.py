import argparse
import os

from stages.data_preprocessing import DataPreprocessing
from stages.retrieval_models_testing import RetrievalModels
from stages.ranking_model import Ranking



if __name__ == '__main__':
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument('-s', '--stage', help='dvc stage name')
    args = vars(argument_parser.parse_args())

    stage_dict = {
        'data_preprocessing': DataPreprocessing,
        'retrieval_models_testing': RetrievalModels,
        'ranking_model': Ranking
    }

    stage = stage_dict[args['stage']]
    stage().run()
