"""
Training modules for discrete diffusion and Fisher-Rao flow matching
graph generation.
"""

# Lazy imports to avoid requiring torch_geometric just to import the package


def __getattr__(name):
    """Lazy import trainers only when accessed."""
    _lazy = {
        "TrainerG2PTLLaDA": "deepgraphgen.trainers.trainer_llada",
        "TrainerG2PTScore": "deepgraphgen.trainers.trainer_score",
        "TrainerG2PTKG": "deepgraphgen.trainers.trainer_llada_kg",
        "TrainerG2PTFisherFM": "deepgraphgen.trainers.trainer_fisher_fm",
        "TrainerKGFisherFM": "deepgraphgen.trainers.trainer_fisher_fm",
    }
    if name in _lazy:
        import importlib
        mod = importlib.import_module(_lazy[name])
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
