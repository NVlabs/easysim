# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import argparse

from easysim.config import get_cfg


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
