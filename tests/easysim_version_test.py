# Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the MIT License [see LICENSE for details].

"""Unit tests for the `easysim` package version."""

# SRL
import easysim


def test_easysim_version() -> None:
    """Test `easysim` package version is set."""
    assert easysim.__version__ is not None
    assert easysim.__version__ != ""
