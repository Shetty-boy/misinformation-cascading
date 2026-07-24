"""
Twitter16 adapter — thin wrapper around the shared Twitter15/16 loader.
The raw formats are identical; only the event_id label differs.
"""

from .twitter15 import load_twitter


def load_twitter16(dataset_root: str) -> list[dict]:
    return load_twitter(dataset_root, event_id="twitter16")
