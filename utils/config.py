import os
import sys
import json

from pprint import pprint
from easydict import EasyDict

from utils.dirs import create_dirs


def get_config_from_json(json_file):
    """
    Get the config from a json file
    :param json_file: the path of the config file
    :return: config(namespace), config(dictionary)
    """
    try:
        # parse the configurations from the config json file provided
        with open(json_file, 'r') as config_file:
            config_dict = json.load(config_file)
    except FileNotFoundError:
        print("ERROR: Config file not found: {}".format(json_file), file=sys.stderr)
        exit(1)
    except json.decoder.JSONDecodeError:
        print("ERROR: Config file is not a proper JSON file!", file=sys.stderr)
        exit(1)

    # convert the dictionary to a namespace using bunch lib
    config = EasyDict(config_dict)

    return config, config_dict


def process_config(json_file):
    """
    Get the json file then editing the path of the experiments folder, creating the dir and return the config
    :param json_file: the path of the config file
    :return: config object(namespace)
    """
    config, _ = get_config_from_json(json_file)
    try:
        config.summary_dir = os.path.join("experiments", config.exp_name, "summaries/")
        config.checkpoint_dir = os.path.join("experiments", config.exp_name, "checkpoints/")
        config.out_dir = os.path.join("experiments", config.exp_name, "out/")
        create_dirs([config.summary_dir, config.checkpoint_dir, config.out_dir])

        print("\n\n The json config file is parsed successfully..\n\n Experiment dirs successfully created..")
        print("\n Please Make sure of the config \n")
        pprint(config)
        print("\n")

    except AttributeError as e:
        print("ERROR!!..Please provide the exp_name in json file..")
        exit(-1)
    return config
