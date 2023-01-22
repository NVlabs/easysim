# Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the MIT License [see LICENSE for details].

import argparse

from easysim.config.config import get_cfg


def parse_config_args():
    """ """
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg-file", help="path to config file")
    parser.add_argument(
        "opts",
        nargs=argparse.REMAINDER,
        help=(
            """modify config options at the end of the command; use space-separated """
            """"PATH.KEY VALUE" pairs; see src/easysim/config.py for all options"""
        ),
    )
    args = parser.parse_args()
    return args


def get_config_from_args():
    """ """
    args = parse_config_args()
    cfg = get_cfg()
    if args.cfg_file is not None:
        cfg.merge_from_file(args.cfg_file)
    cfg.merge_from_list(args.opts)
    return cfg
