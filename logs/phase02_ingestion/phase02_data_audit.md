# CASCADE2VEC — Data Audit Report

_Generated: 2026-08-05 09:18:09_

## Overview
| Metric | Value |
|--------|-------|
| Total rows (post-dedup) | 102,440 |
| Duplicates dropped | 772 |
| Parse failures (pheme) | 0 |

## Rows per Dataset
| event_id | rows | cascades |
|----------|------|---------|
| charliehebdo | 38,268 | 2,079 |
| ferguson | 23,403 | 1,143 |
| germanwings-crash | 4,489 | 469 |
| ottawashooting | 12,284 | 890 |
| sydneysiege | 23,996 | 1,221 |

## Class Balance
| label | count | % of total |
|-------|-------|-----------|
| non-rumour | 71,210 | 69.5% |
| rumour | 31,230 | 30.5% |

## Missing Fields
| column | missing | % missing |
|--------|---------|----------|
| tweet_id | 0 | 0.0% |
| user_id | 0 | 0.0% |
| timestamp | 0 | 0.0% |
| text | 0 | 0.0% |
| parent_id | 5,802 | 5.7% |
| cascade_id | 0 | 0.0% |
| event_id | 0 | 0.0% |
| label | 0 | 0.0% |

## Timestamp Sanity
- Rows with negative timestamp (replies before root): **0** (0.0%)
- Rows with null/unparseable timestamp: **0** (0.0%)
