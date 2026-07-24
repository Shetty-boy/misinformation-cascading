"""
Twitter15 / Twitter16 adapter.

Expected directory layout:
  <dataset_root>/
    source_tweets.txt   ← tab-separated: tweet_id <TAB> text
    label.txt           ← colon-separated: label:tweet_id
                           labels: true | false | unverified | non-rumor
    tree/               ← (optional) propagation trees, one file per cascade
      <cascade_id>.txt  ← each line: parent_tweet_id->child_tweet_id:user_id:timestamp
                           root line uses format: ROOT->cascade_id:0:0

Note: The downloaded dataset (gszswork repo) only contains source_tweets.txt and
label.txt (no tree/ directory). Propagation structure is therefore unavailable;
all rows will have parent_id=None and timestamp=0. Tweets missing from source_tweets.txt
are dropped per the project's design-doc policy.
"""

import os
from typing import Optional

from ..schema import SCHEMA_COLUMNS

# Label normalisation: dataset uses "non-rumor" (no 'u'), unify to "non-rumour"
_LABEL_MAP = {
    "true": "true",
    "false": "false",
    "unverified": "unverified",
    "non-rumor": "non-rumour",
    "non-rumour": "non-rumour",
}


def _load_labels(label_path: str) -> dict[str, str]:
    labels: dict[str, str] = {}
    try:
        with open(label_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or ":" not in line:
                    continue
                raw_label, tweet_id = line.split(":", 1)
                labels[tweet_id.strip()] = _LABEL_MAP.get(
                    raw_label.strip().lower(), "unknown"
                )
    except OSError:
        pass
    return labels


def _load_source_tweets(src_path: str) -> dict[str, str]:
    """Returns {tweet_id: text}."""
    tweets: dict[str, str] = {}
    try:
        with open(src_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t", 1)
                if len(parts) == 2:
                    tweets[parts[0].strip()] = parts[1].strip()
                else:
                    tweets[parts[0].strip()] = ""
    except OSError:
        pass
    return tweets


def _load_trees(tree_dir: str) -> dict[str, list[tuple[str, str, str, int]]]:
    """
    Returns {cascade_id: [(parent_id, tweet_id, user_id, timestamp_sec), ...]}
    Only called if a tree/ directory exists.
    """
    trees: dict[str, list[tuple[str, str, str, int]]] = {}
    if not os.path.isdir(tree_dir):
        return trees

    for fname in os.listdir(tree_dir):
        if not fname.endswith(".txt"):
            continue
        cascade_id = fname[:-4]
        edges = []
        try:
            with open(os.path.join(tree_dir, fname), encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    # Format: parent->child:user_id:timestamp
                    try:
                        arrow_part, rest = line.split("->", 1)
                        child_part, user_id, ts_str = rest.rsplit(":", 2)
                        parent_id = arrow_part.strip()
                        tweet_id = child_part.strip()
                        timestamp = int(float(ts_str.strip()))
                        edges.append((parent_id, tweet_id, user_id.strip(), timestamp))
                    except (ValueError, AttributeError):
                        continue
        except OSError:
            continue
        trees[cascade_id] = edges

    return trees


def load_twitter(dataset_root: str, event_id: str) -> list[dict]:
    """
    Load a Twitter15 or Twitter16 dataset from *dataset_root*.

    Args:
        dataset_root: Path to the dataset directory.
        event_id: Identifier string, e.g. "twitter15" or "twitter16".

    Returns:
        List of unified-schema dicts.
    """
    label_path = os.path.join(dataset_root, "label.txt")
    src_path = os.path.join(dataset_root, "source_tweets.txt")
    tree_dir = os.path.join(dataset_root, "tree")

    labels = _load_labels(label_path)
    source_tweets = _load_source_tweets(src_path)
    trees = _load_trees(tree_dir)  # empty dict if no tree/ dir

    rows: list[dict] = []

    if trees:
        # Full propagation-tree mode
        for cascade_id, edges in trees.items():
            label = labels.get(cascade_id, "unknown")
            # Sort by timestamp to find root (t=0)
            for parent_id, tweet_id, user_id, timestamp in edges:
                # The "ROOT" sentinel means this IS the root tweet
                is_root = parent_id.upper() == "ROOT"
                text = source_tweets.get(tweet_id, "")
                rows.append({
                    "tweet_id": tweet_id,
                    "user_id": user_id,
                    "timestamp": timestamp,
                    "text": text,
                    "parent_id": None if is_root else parent_id,
                    "cascade_id": cascade_id,
                    "event_id": event_id,
                    "label": label,
                })
    else:
        # Source-only mode: no tree/ directory available
        # Each source tweet is treated as a standalone root cascade.
        for tweet_id, text in source_tweets.items():
            label = labels.get(tweet_id, "unknown")
            rows.append({
                "tweet_id": tweet_id,
                "user_id": "",       # not available without tree file
                "timestamp": 0,
                "text": text,
                "parent_id": None,
                "cascade_id": tweet_id,
                "event_id": event_id,
                "label": label,
            })

    return rows
