"""Architecture-specific feature flags and parameters.

Toggle between Architecture A (parallel) and Architecture B (sequential precision)
by setting CAPG_ARCHITECTURE=A or CAPG_ARCHITECTURE=B in the environment.
"""

from dataclasses import dataclass

from config.settings import settings


@dataclass(frozen=True)
class ArchitectureConfig:
    """Describes which components and strategies an architecture uses."""

    name: str

    # Agent topology
    separate_query_transform: bool  # B splits query transform from context builder
    parallel_raw_retrieval: bool    # A starts retrieval on raw query in parallel
    hierarchical_retrieval: bool    # B uses chapter → section → paragraph narrowing

    # Judge strategy
    judge_reuses_generation_model: bool  # B reuses generation model; A uses separate

    # Retry strategy
    retry_via_routing_signal: bool  # B uses routing signal; A uses retry_mode flag


ARCH_A = ArchitectureConfig(
    name="A",
    separate_query_transform=False,
    parallel_raw_retrieval=True,
    hierarchical_retrieval=False,
    judge_reuses_generation_model=False,
    retry_via_routing_signal=False,
)

ARCH_B = ArchitectureConfig(
    name="B",
    separate_query_transform=True,
    parallel_raw_retrieval=False,
    hierarchical_retrieval=True,
    judge_reuses_generation_model=True,
    retry_via_routing_signal=True,
)

_ARCHITECTURES = {"A": ARCH_A, "B": ARCH_B}


def get_architecture() -> ArchitectureConfig:
    """Return the active architecture based on settings."""
    key = settings.capg_architecture.upper()
    if key not in _ARCHITECTURES:
        raise ValueError(f"Unknown architecture '{key}'. Choose 'A' or 'B'.")
    return _ARCHITECTURES[key]
