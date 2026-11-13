# temporal_purge

core.chronological(rows) sorts records by integer t, then string id. core.split(rows, cutoff, gap) assigns whole entity groups: training only if max group timestamp < cutoff-gap; evaluation only if min group timestamp >= cutoff+gap; all other groups are purged. Gap is nonnegative, invalid gap raises ValueError. Input rows have id, entity, t. solve returns {train:[ids], eval:[ids], purged:[ids]} each in chronological order. No input mutation.

Only core.py and pipeline.py may be modified. Tests, contracts and fixtures are protected. A passing visible test is not sufficient for acceptance.
