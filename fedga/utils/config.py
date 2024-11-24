import argparse
import os

import yaml
from box import Box

DEF_CONF = f'{os.path.dirname(__file__)}/config_default.yaml'


def parse_args(ipython=False):
    parser = argparse.ArgumentParser(description="options for FedLearn")

    parser.add_argument(
        "--cfg",
        help="custom config file",
        type=str,
    )

    parser.add_argument(
        "--id",
        help="id of multiple runs with same config",
        type=int
    )

    parser.add_argument(
        "opts",
        help="modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args() if not ipython else parser.parse_args([])
    return args


def get_config(ipython=False, frozen_box=False):
    args = parse_args(ipython)

    with open(DEF_CONF, 'r') as f:
        default_config = yaml.load(f, Loader=yaml.FullLoader)
    args_box = Box(default_config, box_dots=True)

    if args.cfg is not None:
        with open(args.cfg, 'r') as f:
            custom_config = yaml.load(f, Loader=yaml.FullLoader)
        cus_box = Box(custom_config, box_dots=True)
        args_box.merge_update(cus_box)

    if args.id is not None:
        args_box['id'] = args.id

    for i in range(0, len(args.opts), 2):
        args_box[args.opts[i]] = args.opts[i+1]

    args_box._box_config['frozen_box'] = frozen_box
    return args_box
