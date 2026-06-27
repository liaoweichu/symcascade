"""Global configuration constants."""
from dataclasses import dataclass


@dataclass(frozen=True)
class CascadeConfig:
    """Runtime configuration for the cascade."""
    conformal_alpha: float = 0.1        # 1-alpha coverage
    ewma_decay: float = 0.95            # online smoothing
    retrain_every: int = 200            # discriminator retrain period
    sliding_window: int = 2000          # online learning window
    skeleton_sim_threshold: float = 0.75  # cosine threshold for skeleton match
    gemma_thinking: bool = False        # force thinking=False on low-cost tier
