import argparse
import os

from stages.data_preprocessing import DataPreprocessing

if __name__ == '__main__':
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument('-s', '--stage', help='dvc stage name')
    args = vars(argument_parser.parse_args())

    stage_dict = {
        'data_preprocessing': DataPreprocessing,
    }

    stage = stage_dict[args['stage']]
    stage().run()
