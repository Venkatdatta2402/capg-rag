"""Tests for Architecture A pipeline orchestration."""

import pytest

from config.architectures import ARCH_A, ARCH_B


class TestArchitectureConfig:
    """Tests for architecture configuration."""

    def test_arch_a_config(self):
        assert ARCH_A.name == "A"
        assert ARCH_A.parallel_raw_retrieval is True
        assert ARCH_A.separate_query_transform is False
        assert ARCH_A.hierarchical_retrieval is False
        assert ARCH_A.judge_reuses_generation_model is False

    def test_arch_b_config(self):
        assert ARCH_B.name == "B"
        assert ARCH_B.parallel_raw_retrieval is False
        assert ARCH_B.separate_query_transform is True
        assert ARCH_B.hierarchical_retrieval is True
        assert ARCH_B.judge_reuses_generation_model is True
