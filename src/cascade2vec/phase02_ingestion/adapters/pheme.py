"""
PHEME adapter (pheme-rnr-dataset format).

Directory structure expected:
  <pheme_root>/
    <event>/                    e.g. charliehebdo, ferguson, ...
      rumours/
        <cascade_id>/
          source-tweet/
            <cascade_id>.json   ← root tweet
          reactions/
            <tweet_id>.json     ← reply tweets
      non-rumours/
        ...

Outputs a list of dicts conforming to the unified schema.
"""

import json
import os
from datetime import datetime, timezone
from typing import Iterator

from ..schema import SCHEMA_COLUMNS


# Twitter date format used across all tweet JSON files
_TW_DATE_FMT = "%a %b %d %H:%M:%S %z %Y"


def _parse_created_at(raw: str | None) -> datetime | None:
    if not raw:
        return None
    try:
        return datetime.strptime(raw, _TW_DATE_FMT)
    except ValueError:
        return None


def _tweet_to_row(
    tweet_json: dict,
    *,
    cascade_id: str,
    event_id: str,
    label: str,
    parent_id: str | None,
    root_dt: datetime | None,
) -> dict:
    """Map a raw tweet JSON dict to the unified schema row."""
    tweet_id = str(tweet_json.get("id_str") or tweet_json.get("id") or "")
    user_id = str(
        tweet_json.get("user", {}).get("id_str")
        or tweet_json.get("user", {}).get("id")
        or ""
    )
    text = tweet_json.get("text") or ""
    raw_dt = _parse_created_at(tweet_json.get("created_at"))

    # Normalize timestamp relative to cascade root
    if raw_dt and root_dt:
        timestamp = int((raw_dt - root_dt).total_seconds())
    elif raw_dt:
        timestamp = int(raw_dt.timestamp())
    else:
        timestamp = None

    return {
        "tweet_id": tweet_id,
        "user_id": user_id,
        "timestamp": timestamp,
        "text": text,
        "parent_id": parent_id,
        "cascade_id": cascade_id,
        "event_id": event_id,
        "label": label,
    }


def _load_json(path: str) -> dict | None:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _iter_cascade(
    cascade_dir: str, *, event_id: str, label: str
) -> Iterator[dict]:
    """Yield all unified-schema rows for one cascade directory."""
    cascade_id = os.path.basename(cascade_dir)

    # --- Source tweet ---
    src_dir = os.path.join(cascade_dir, "source-tweet")
    src_json_path = os.path.join(src_dir, f"{cascade_id}.json")
    src_data = _load_json(src_json_path)
    if src_data is None:
        return  # Skip malformed cascade

    root_dt = _parse_created_at(src_data.get("created_at"))
    root_row = _tweet_to_row(
        src_data,
        cascade_id=cascade_id,
        event_id=event_id,
        label=label,
        parent_id=None,
        root_dt=root_dt,
    )
    yield root_row

    # --- Reactions ---
    reactions_dir = os.path.join(cascade_dir, "reactions")
    if not os.path.isdir(reactions_dir):
        return

    for fname in os.listdir(reactions_dir):
        if not fname.endswith(".json") or "Zone" in fname:
            continue
        reaction_data = _load_json(os.path.join(reactions_dir, fname))
        if reaction_data is None:
            continue

        # parent = in_reply_to_status_id_str if present, else cascade root
        parent_id = str(
            reaction_data.get("in_reply_to_status_id_str")
            or reaction_data.get("in_reply_to_status_id")
            or cascade_id
        )
        yield _tweet_to_row(
            reaction_data,
            cascade_id=cascade_id,
            event_id=event_id,
            label=label,
            parent_id=parent_id,
            root_dt=root_dt,
        )


def load_pheme(pheme_root: str) -> list[dict]:
    """
    Load the full PHEME dataset from *pheme_root* into unified schema rows.

    Args:
        pheme_root: Path to the directory that contains event sub-directories
                    (e.g. 'data/raw/pheme-rnr-dataset').

    Returns:
        List of unified-schema dicts.
    """
    rows: list[dict] = []
    parse_failures: list[str] = []

    for event_id in os.listdir(pheme_root):
        event_dir = os.path.join(pheme_root, event_id)
        if not os.path.isdir(event_dir) or event_id.startswith("README"):
            continue

        for label_folder, label in [("rumours", "rumour"), ("non-rumours", "non-rumour")]:
            label_dir = os.path.join(event_dir, label_folder)
            if not os.path.isdir(label_dir):
                continue

            for cascade_id in os.listdir(label_dir):
                cascade_dir = os.path.join(label_dir, cascade_id)
                if not os.path.isdir(cascade_dir):
                    continue
                try:
                    rows.extend(
                        _iter_cascade(cascade_dir, event_id=event_id, label=label)
                    )
                except Exception as exc:
                    parse_failures.append(f"{cascade_dir}: {exc}")

    if parse_failures:
        print(f"[PHEMEAdapter] {len(parse_failures)} cascade(s) failed to parse:")
        for f in parse_failures[:10]:
            print(f"  {f}")

    return rows
