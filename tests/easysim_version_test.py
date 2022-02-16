# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Unit tests for the `easysim` package version."""

# SRL
import easysim


def test_easysim_version() -> None:
    """Test `easysim` package version is set."""
    assert easysim.__version__ is not None
    assert easysim.__version__ != ""
