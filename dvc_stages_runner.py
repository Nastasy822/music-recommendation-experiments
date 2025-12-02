import argparse
import os

from stages.data_preprocessing import DataPreprocessing
from stages.retrieval_models_training import RetrievalModelsTraining
from stages.retrieval_models_testing import RetrievalModelsTesting
from stages.ranking_model import Ranking
from stages.features_extractor import FeaturesExtractor
from stages.candidate_generator import CandidateGenerator

if __name__ == '__main__':
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument('-s', '--stage', help='dvc stage name')
    args = vars(argument_parser.parse_args())

    stage_dict = {
        'data_preprocessing': DataPreprocessing,
        'retrieval_models_training': RetrievalModelsTraining,
        'retrieval_models_testing': RetrievalModelsTesting,
        'features_extractor': FeaturesExtractor,
        'candidate_generator': CandidateGenerator,
        'ranking_model':  Ranking
    }

    stage = stage_dict[args['stage']]
    stage().run()
