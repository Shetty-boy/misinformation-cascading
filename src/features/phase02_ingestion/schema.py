"""
Unified schema for CASCADE2VEC project.

All dataset adapters must map their raw format to this schema:

tweet_id   (str)  - Unique ID of the tweet/post
user_id    (str)  - Unique ID of the author
timestamp  (int)  - Seconds since cascade root tweet (t_root = 0)
text       (str)  - Text content of the tweet (may be empty string if missing)
parent_id  (str)  - tweet_id of the parent in cascade tree (None for root)
cascade_id (str)  - ID of the source/root tweet (thread identifier)
event_id   (str)  - Dataset-specific event label (e.g. "charliehebdo", "twitter15")
label      (str)  - "rumour" | "non-rumour" | "true" | "false" | "unverified" | "unknown"
"""

SCHEMA_COLUMNS = [
    "tweet_id",
    "user_id",
    "timestamp",
    "text",
    "parent_id",
    "cascade_id",
    "event_id",
    "label",
]
