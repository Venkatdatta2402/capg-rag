"""Orchestrator factory — returns the correct pipeline based on config."""

from config.architectures import get_architecture
from src.orchestrator.base import BaseOrchestrator


def get_orchestrator() -> BaseOrchestrator:
    """Create and return the orchestrator for the active architecture."""
    arch = get_architecture()

    if arch.name == "A":
        from src.orchestrator.arch_a_pipeline import ArchAPipeline
        return ArchAPipeline()
    elif arch.name == "B":
        from src.orchestrator.arch_b_pipeline import ArchBPipeline
        return ArchBPipeline()
    else:
        raise ValueError(f"Unknown architecture: {arch.name}")
