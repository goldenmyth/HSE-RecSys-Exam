"""Candidate generation and train-history features for the supervised reranker.

This module does not fit a model or read relevance labels when constructing
candidates/features. Forecast schedule input contains canonical exposure counts
from ``history_reranking.schedule_context``, never future viewing behaviour.
"""

from collections import Counter

import numpy as np
from scipy import sparse

from scripts.prepare_interactions import wall_seconds


# Additions to the runner's score/rank feature group A, in output-column order.
# The runner constructs cumulative A, A+B, A+B+C, A+B+C+D groups itself.
FEATURE_GROUPS = {
    "B": ["history_nitems", "log_total_watches", "watches_per_train_day",
          "item_distinct_days", "item_last_age_days", "item_day_count_regularity",
          "item_watch_rate"],
    "C": ["hour_alignment", "weekday_alignment", "weekday_hour_alignment",
          "hour_concentration", "weekday_concentration", "weekday_hour_concentration"],
    "D": ["channel_alignment", "channel_hour_alignment", "joint_alignment",
          "channel_concentration", "history_nchannels"],
}


def _pairs(pairs):
    values = np.asarray(pairs, dtype=np.int64)
    if values.ndim != 2 or np.any(values <= 0):
        raise ValueError("Candidate pairs must be a positive-ID users x items matrix")
    if any(len(set(row)) != len(row) for row in values):
        raise ValueError("Candidate IDs must be unique within each user")
    return values


def candidate_union(lists, mandatory, popularity, k=200):
    """Keep mandatory V3 Top5, PPR Top100, then interleave native sources.

    ``lists`` maps ppr/repeat/als/content to already-ranked per-user lists;
    ``popularity`` is one already-ranked global fallback list. Native scores and
    ground truth are neither consumed nor inferred. Duplicate IDs are skipped.
    A source turn consumes its next unseen item, in repeat/ALS/content/pop order.
    """
    if not isinstance(k, (int, np.integer)) or k < 5:
        raise ValueError("k must be an integer of at least five")
    rows = len(mandatory)
    sources = {name: lists.get(name, [[] for _ in range(rows)])
               for name in ("ppr", "repeat", "als", "content")}
    if any(len(source) != rows for source in sources.values()):
        raise ValueError("Native source lists must align with mandatory users")
    popularity = tuple(int(item) for item in popularity)
    result = np.empty((rows, k), dtype=np.int64)
    for number in range(rows):
        selected, seen = [], set()

        def add(item):
            item = int(item)
            if item <= 0:
                raise ValueError("Candidate IDs must be positive")
            if item not in seen and len(selected) < k:
                seen.add(item)
                selected.append(item)

        if len(mandatory[number]) > k:
            raise ValueError("Mandatory candidates do not fit within k")
        for item in mandatory[number]:
            add(item)
        for item in sources["ppr"][number][:100]:
            add(item)
        streams = [iter(sources[name][number]) for name in ("repeat", "als", "content")]
        streams.append(iter(popularity))
        while len(selected) < k and streams:
            active = []
            for stream in streams:
                for item in stream:
                    previous = len(selected)
                    add(item)
                    if len(selected) > previous:
                        active.append(stream)
                        break
                if len(selected) == k:
                    break
            streams = active
        if len(selected) != k:
            raise ValueError("Native sources and popularity contain fewer than k unique IDs")
        result[number] = selected
    return result


def candidate_diagnostics(pairs, users, truth):
    """Full-GT recall and optimistic MAP@5 bound; empty-GT users contribute zero.

    Recall denominators include *all* relevant items, even outside candidates.
    The oracle bound is min(5, hits) / min(5, |GT|), assuming perfect ranking.
    """
    pairs = _pairs(pairs)
    if len(pairs) != len(users):
        raise ValueError("User and candidate rows must align")
    hits, sizes = [], []
    for user, row in zip(users, pairs):
        relevant = set(truth.get(user, ()))
        sizes.append(len(relevant))
        hits.append(len(relevant.intersection(row)))
    hits, sizes = np.asarray(hits), np.asarray(sizes)
    nonempty = sizes > 0
    recalls = np.divide(hits, sizes, out=np.zeros(len(users)), where=nonempty)
    oracle = np.divide(np.minimum(5, hits), np.minimum(5, sizes),
                       out=np.zeros(len(users)), where=nonempty)
    return {"users": len(users), "candidates_per_user": int(pairs.shape[1]),
            "nonempty_gt_users": int(nonempty.sum()), "relevant_pairs": int(sizes.sum()),
            "relevant_candidate_pairs": int(hits.sum()),
            "macro_recall_at_200": float(recalls.mean()) if len(users) else 0.0,
            "micro_recall_at_200": float(hits.sum() / sizes.sum()) if sizes.sum() else 0.0,
            "nonempty_macro_recall_at_200": float(recalls[nonempty].mean()) if nonempty.any() else 0.0,
            "any_hit_fraction": float((hits > 0).mean()) if len(users) else 0.0,
            "oracle_map_at_5": float(oracle.mean()) if len(users) else 0.0}


def _shares(counter):
    total = sum(counter.values())
    return {key: value / total for key, value in counter.items()} if total else {}


def _aggregate(counter, key):
    result = Counter()
    for slot, count in counter.items():
        result[key(slot)] += count
    return result


def history_features(history, context, users, pairs, schedule, forecast_days):
    """Return float32 [users, candidates, features] and B/C/D column names.

    B uses observed counts, distinct watch-start dates, and last observation.
    Rates use the entire elapsed training window, not the user's first event;
    regularity is min(1, distinct_dates/count). Last-age is NaN for unseen pairs.
    Watch duration is unavailable in these aggregates and is not fabricated.

    C/D average a user's *sum-normalised training event shares* over canonical
    forecast airings. Calendar bins (weekday, 3-hour bin) come from the existing
    schedule/context helpers, which retain original calendar labels across the
    clock gap. Concentration is sum(shares**2), with no ID-valued features.
    Unknown profiles have zero C/D features. Global forecast frequency already
    belongs to runner group A; ``forecast_days`` validates the forecast window.
    """
    pairs = _pairs(pairs)
    if len(pairs) != len(users) or not np.isfinite(forecast_days) or forecast_days <= 0:
        raise ValueError("Users must align and forecast_days must be positive")
    counts = sparse.csr_matrix(history["counts"])
    last = sparse.csr_matrix(history["last"])
    days = sparse.csr_matrix(context["days"])
    if last.shape != counts.shape or days.shape != counts.shape:
        raise ValueError("Count/day/last matrices must align")
    train_days = (history["cutoff"] - wall_seconds(history["lower"])) / 86400
    if not np.isfinite(train_days) or train_days <= 0:
        raise ValueError("Training window must have positive elapsed duration")
    columns = {int(item): number for number, item in enumerate(history["items"])}
    names = [name for group in FEATURE_GROUPS.values() for name in group]
    result = np.zeros((*pairs.shape, len(names)), dtype=np.float32)
    result[:, :, names.index("item_last_age_days")] = np.nan
    # Exposure histograms are item-only: derive once, not once per user/item.
    future_by_item = {}
    for item in np.unique(pairs):
        exposure = schedule["items"].get(int(item))
        if not exposure or not exposure["airings"]:
            continue
        joint = exposure["joint"]
        future_by_item[int(item)] = (exposure["airings"], {
            "hour": _aggregate(joint, lambda key: key[2]),
            "weekday": _aggregate(joint, lambda key: key[1]),
            "weekday_hour": exposure["slot"], "channel": exposure["channel"],
            "channel_hour": _aggregate(joint, lambda key: (key[0], key[2])), "joint": joint})
    for position, user in enumerate(users):
        number = history["users"].get(user)
        if number is None:
            continue
        row = counts.getrow(number)
        observed = dict(zip(row.indices, row.data))
        total = float(row.sum())
        result[position, :, :3] = (row.nnz, np.log1p(total), total / train_days)
        day_row, last_row = days.getrow(number), last.getrow(number)
        distinct, recent = dict(zip(day_row.indices, day_row.data)), dict(zip(last_row.indices, last_row.data))
        for target, item in enumerate(pairs[position]):
            column = columns.get(int(item))
            count, dates = observed.get(column, 0), distinct.get(column, 0)
            result[position, target, 3] = dates
            if count:
                if column in recent:
                    result[position, target, 4] = (history["cutoff"] - recent[column]) / 86400
                result[position, target, 5:7] = (min(1, dates / count), count / train_days)
        if total <= 0:
            continue
        context_number = context["users"].get(user)
        if context_number is None:
            continue
        joint = context["joint"][context_number]
        distributions = {
            "hour": _shares(context["hour"][context_number]),
            "weekday": _shares(context["weekday"][context_number]),
            "weekday_hour": _shares(_aggregate(joint, lambda key: key[1:])),
            "channel": _shares(context["channel"][context_number]),
            "channel_hour": _shares(_aggregate(joint, lambda key: (key[0], key[2]))),
            "joint": _shares(joint),
        }
        concentrations = {key: sum(value * value for value in values.values())
                          for key, values in distributions.items()}
        result[position, :, 10:13] = [concentrations[key] for key in ("hour", "weekday", "weekday_hour")]
        result[position, :, 16:18] = (concentrations["channel"], len(distributions["channel"]))
        for target, item in enumerate(pairs[position]):
            exposure = future_by_item.get(int(item))
            if exposure is None:
                continue
            airing_count, future = exposure
            for feature, index in (("hour", 7), ("weekday", 8), ("weekday_hour", 9),
                                   ("channel", 13), ("channel_hour", 14), ("joint", 15)):
                result[position, target, index] = sum(count * distributions[feature].get(key, 0)
                    for key, count in future[feature].items()) / airing_count
    return result, names


def rank_candidate_predictions(probabilities, pairs, users_with_history, fallback):
    """Rank five finite native probabilities, numeric-ID ties; cold = fallback.

    ``users_with_history`` is a boolean mask aligned with candidate rows. Warm
    rows need at least five finite probabilities; no silent popularity padding.
    """
    pairs = _pairs(pairs)
    values = np.asarray(probabilities, dtype=np.float64)
    warm = np.asarray(users_with_history, dtype=bool)
    fallback = [int(item) for item in fallback]
    if values.shape != pairs.shape or warm.shape != (len(pairs),) or pairs.shape[1] < 5:
        raise ValueError("Probability/candidate/history shapes must align with at least five candidates")
    if len(fallback) < 5 or len(set(fallback)) != len(fallback) or any(item <= 0 for item in fallback):
        raise ValueError("Fallback needs at least five distinct positive IDs")
    predictions = []
    for row, candidates, known in zip(values, pairs, warm):
        if not known:
            if not set(fallback[:5]).issubset(candidates):
                raise ValueError("Cold fallback must belong to the user's candidate pool")
            predictions.append(fallback[:5].copy())
            continue
        finite = np.flatnonzero(np.isfinite(row))
        if len(finite) < 5:
            raise ValueError("Warm users require at least five finite native probabilities")
        order = finite[np.lexsort((candidates[finite], -row[finite]))[:5]]
        predictions.append([int(item) for item in candidates[order]])
    return predictions
