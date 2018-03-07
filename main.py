"""
This Framework is authored and designed by Mo'men AbdelRazek and HIS TEAM (KoKoMind)

Main
-Capture the config file
-Process the json config passed
-Create an agent instance
-Run the agent
"""

import argparse
from utils.config import process_config

from agents import *


def main():
    # parse the path of the json config file
    arg_parser = argparse.ArgumentParser(description="Deep learning FRAMEWORK")
    arg_parser.add_argument(
        'config',
        metavar='config_json_file',
        default='None',
        help='The Configuration file in json format')
    args = arg_parser.parse_args()

    # parse the config json file
    config = process_config(args.config)

    # Get the class of the Agent and the Model
    agent_class = globals()[config.agent]

    # Create the Agent and pass all the configuration to it then run it..
    agent = agent_class(config)
    try:
        agent.run()
    except KeyboardInterrupt:
        agent.finalize()


if __name__ == '__main__':
    main()
