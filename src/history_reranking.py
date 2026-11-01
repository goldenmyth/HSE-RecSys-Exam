"""Train-only seed-веса и ограниченное ранжирование по каналу/календарю эфиров."""

import csv
import time
from array import array
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import sparse

from scripts.prepare_interactions import build_channel_index, iso, iter_intersections, wall_seconds


def load_train_context(airings_csv, history):
    """Прочитать confirmed-airing CSV с непрерывными группами user_id.

    Qualification >50% выполняет исходный session matcher. Здесь проверяются
    границы и уникальность его событий; каналы из item-агрегатов не угадываются.
    """
    started = time.perf_counter()
    lower = wall_seconds(history.get("lower", "2020-03-09 00:00:00"))
    upper, period = history["cutoff"], history.get("period", "train")
    counts = sparse.csr_matrix(history["counts"], dtype=np.float64, copy=True)
    counts.sum_duplicates()
    counts.eliminate_zeros()
    counts.sort_indices()
    users = history["users"]
    item_index = {int(item): number for number, item in enumerate(history["items"])}
    preferences = {name: [Counter() for _ in range(counts.shape[0])]
                   for name in ("channel", "weekday", "hour", "joint")}
    rows, columns, days, recency = array("i"), array("i"), array("f"), array("d")
    current, current_days, events, seen_users = None, defaultdict(set), set(), set()
    current_decay = defaultdict(lambda: [0.0, 0])
    stats = Counter()

    def flush():
        if current is not None:
            for item, dates in current_days.items():
                rows.append(users[current])
                columns.append(item)
                days.append(len(dates))
                total_decay, event_count = current_decay[item]
                recency.append(total_decay / event_count)

    with Path(airings_csv).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"user_id", "tv_show_id", "channel_id", "airing_start", "airing_end",
                    "watch_start", "watch_end", "period"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("В confirmed-airing CSV отсутствуют обязательные столбцы")
        for row in reader:
            user, item, channel = row["user_id"], int(row["tv_show_id"]), int(row["channel_id"])
            a_start, a_end = wall_seconds(row["airing_start"]), wall_seconds(row["airing_end"])
            start, end = wall_seconds(row["watch_start"]), wall_seconds(row["watch_end"])
            if row["period"] != period or not lower <= start < end <= upper or not a_start <= start < end <= a_end:
                raise ValueError("Событие вышло за training period или интервал эфира")
            if user not in users or item not in item_index or channel <= 0:
                raise ValueError("Событие не соответствует пользователям/программам training history")
            user_row, item_column = users[user], item_index[item]
            if counts[user_row, item_column] <= 0:
                raise ValueError("Confirmed airing отсутствует в training counts")
            if user != current:
                if user in seen_users:
                    raise ValueError("Ожидались непрерывные группы user_id в confirmed-airing CSV")
                flush()
                current, current_days, events = user, defaultdict(set), set()
                current_decay = defaultdict(lambda: [0.0, 0])
                seen_users.add(user)
            key = (item, channel, a_start, a_end)
            if key in events:
                raise ValueError("Повтор confirmed user/channel/canonical airing")
            events.add(key)
            moment = datetime.fromisoformat(row["watch_start"])
            weekday, hour = moment.weekday(), moment.hour // 3
            current_days[item_column].add(moment.date().isoformat())
            current_decay[item_column][0] += 2.0 ** (-(upper - end) / (30 * 86400))
            current_decay[item_column][1] += 1
            preferences["channel"][user_row][channel] += 1
            preferences["weekday"][user_row][weekday] += 1
            preferences["hour"][user_row][hour] += 1
            preferences["joint"][user_row][(channel, weekday, hour)] += 1
            stats["confirmed_airing_events"] += 1
    flush()
    day_matrix = sparse.csr_matrix((days, (rows, columns)), shape=counts.shape, dtype=np.float64)
    day_matrix.sort_indices()
    recency_matrix = sparse.csr_matrix((recency, (rows, columns)), shape=counts.shape, dtype=np.float64)
    recency_matrix.sort_indices()
    if day_matrix.nnz != counts.nnz or not np.array_equal(day_matrix.indptr, counts.indptr) or not np.array_equal(day_matrix.indices, counts.indices):
        raise ValueError("Confirmed airing events не покрывают ровно те же user/item пары, что training counts")
    return {"days": day_matrix, "recency_30": recency_matrix, "users": users, **preferences, "audit": {
        **dict(stats), "users_with_events": len(seen_users), "user_item_pairs": int(day_matrix.nnz),
        "distinct_user_item_days": int(day_matrix.sum()),
        "period": period, "start_inclusive": iso(lower), "end_exclusive": iso(upper),
        "day_semantics": "watch_start calendar date, not normalised-second day number",
        "preference_unit": "one confirmed user/channel/canonical airing, no source-row repetition",
        "recency_policy": "mean per-event exp2(-(cutoff-watch_end)/(30 days)); no count-wide last-time substitution",
        "hour_bin_hours": 3, "elapsed_seconds": time.perf_counter() - started,
    }}


def restart_weights(history, context, mode="log_count", half_life_days=30):
    """Готовые CSR seed-веса; передать в PPR через restart_weights=, без второго log."""
    if mode not in ("log_count", "recency", "regularity"):
        raise ValueError("Неизвестный режим seed-весов")
    counts = sparse.csr_matrix(history["counts"], dtype=np.float64, copy=True)
    counts.sort_indices()
    if not np.all(np.isfinite(counts.data)) or np.any(counts.data <= 0):
        raise ValueError("Training counts должны быть конечными и положительными")
    np.log1p(counts.data, out=counts.data)
    if mode == "recency":
        if half_life_days != 30:
            raise ValueError("Mean event recency рассчитана для фиксированного half_life_days=30")
        decay = sparse.csr_matrix(context["recency_30"], dtype=np.float64, copy=True)
        decay.sort_indices()
        if decay.shape != counts.shape or not np.array_equal(decay.indptr, counts.indptr) or not np.array_equal(decay.indices, counts.indices):
            raise ValueError("Event recency должна быть выровнена с training counts")
        if not np.all(np.isfinite(decay.data)) or np.any(decay.data <= 0) or np.any(decay.data > 1):
            raise ValueError("Средний event decay должен находиться в (0,1]")
        counts.data *= decay.data
    elif mode == "regularity":
        days = sparse.csr_matrix(context["days"], dtype=np.float64, copy=True)
        days.sort_indices()
        if days.shape != counts.shape or not np.array_equal(days.indptr, counts.indptr) or not np.array_equal(days.indices, counts.indices):
            raise ValueError("Distinct days должны быть выровнены с training counts")
        if not np.all(np.isfinite(days.data)) or np.any(days.data <= 0):
            raise ValueError("Distinct days должны быть конечными и положительными")
        counts.data = np.sqrt(counts.data * np.log1p(days.data))
    counts.eliminate_zeros()
    return counts


def schedule_context(paths, start, end, candidates):
    """Forecast exposure только из ID, канала, времени и длительности расписания.

    Целиком исключаем из признаков эфир с неоднозначной частью внутри forecast
    окна. Это не меняет frozen candidate pool и не создаёт поведенческих данных.
    """
    started = time.perf_counter()
    lower, upper = wall_seconds(start), wall_seconds(end)
    if lower >= upper:
        raise ValueError("Forecast interval должен быть непустым")
    pool = {int(item) for item in candidates}
    if any(item <= 0 for item in pool):
        raise ValueError("Candidate IDs должны быть положительными")
    records = defaultdict(lambda: (array("q"), array("q"), array("q")))
    stats = Counter()
    for path in paths:
        with Path(path).open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                stats["source_schedule_rows"] += 1
                item, duration = int(row["tv_show_id"]), int(row["duration"])
                if item < 0 or duration <= 0:
                    continue
                left = wall_seconds(row["start_time"], schedule=True)
                right = left + duration
                if left >= upper or right <= lower:
                    continue
                channel = int(row["channel_id"])
                for target, value in zip(records[channel], (left, right, item)):
                    target.append(value)
                stats["forecast_source_rows_including_zero"] += 1
    exposures = {}
    for channel in list(records):
        index, _ = build_channel_index(zip(*records.pop(channel)))
        covered, airing_items = Counter(), {}
        for item, airing, left, right in iter_intersections(index, lower, upper):
            if item < 0:
                stats["ambiguous_forecast_index_seconds"] += right - left
            elif item > 0 and item in pool:
                covered[airing] += right - left
                airing_items[airing] = item
        for airing, known_seconds in covered.items():
            left = max(lower, index["airing_starts"][airing])
            right = min(upper, index["airing_ends"][airing])
            if known_seconds != right - left:
                stats["excluded_partly_ambiguous_positive_airings"] += 1
                continue
            item = airing_items[airing]
            exposure = exposures.setdefault(item, {"airings": 0, "days": set(), "channel": Counter(),
                                                    "slot": Counter(), "joint": Counter()})
            moment = datetime.fromisoformat(iso(left))
            weekday, hour = moment.weekday(), moment.hour // 3
            exposure["airings"] += 1
            exposure["days"].add(moment.date().isoformat())
            exposure["channel"][channel] += 1
            exposure["slot"][(weekday, hour)] += 1
            exposure["joint"][(channel, weekday, hour)] += 1
            stats["canonical_forecast_airings"] += 1
    for exposure in exposures.values():
        exposure["days"] = len(exposure["days"])
    return {"items": exposures, "audit": {
        **dict(stats), "candidate_count": len(pool), "candidates_with_exposure": len(exposures),
        "start_inclusive": start, "end_exclusive": end,
        "source_attributes_used": ["tv_show_id", "channel_id", "start_time", "duration"],
        "exposure_unit": "one fully unambiguous canonical channel/airing intersecting forecast window",
        "calendar_slot": "weekday and three-hour bin of max(airing_start, forecast_start), in original calendar labels",
        "preference_normalisation": "train event count / user's maximum event count; mean over canonical forecast airings",
        "candidate_pool_unchanged": True, "elapsed_seconds": time.perf_counter() - started,
    }}


def schedule_affinity(context, schedule, users, candidates, mode):
    """Ограниченные [0,1] календарные признаки; нет истории/экспозиции — ноль."""
    if mode not in ("channel", "joint", "availability"):
        raise ValueError("mode должен быть channel, joint или availability")
    ids = [int(item) for item in candidates]
    result = np.zeros((len(users), len(ids)), dtype=np.float64)
    exposures = schedule["items"]
    if mode == "availability":
        values = np.log1p([exposures.get(item, {}).get("airings", 0) for item in ids])
        if len(values) and values.max() > 0:
            result[:] = values / values.max()
        return result
    for position, user in enumerate(users):
        number = context["users"].get(user)
        if number is None:
            continue
        preferences = context[mode][number]
        maximum = max(preferences.values(), default=0)
        if not maximum:
            continue
        for column, item in enumerate(ids):
            exposure = exposures.get(item)
            if exposure:
                result[position, column] = sum(count * preferences.get(key, 0) for key, count in exposure[mode].items()) / (exposure["airings"] * maximum)
    return result


def bounded_rerank(scores, feature, candidates, strength=0.25, limit=100):
    """Только native Top-limit; s' = s*(1+strength*f), f∈[0,1].

    Это reranking неотрицательных PPR scores, без padding и score fusion.
    Неизвестный feature нейтрален; общий fallback применяется вызывающим кодом.
    """
    values, feature = np.asarray(scores, dtype=np.float64), np.asarray(feature, dtype=np.float64)
    ids = np.asarray(candidates, dtype=np.int64)
    if values.ndim != 2 or feature.shape != values.shape or values.shape[1] != len(ids):
        raise ValueError("scores и feature должны иметь форму users × candidates")
    if len(set(ids)) != len(ids) or np.any(ids <= 0) or not isinstance(limit, int) or limit <= 0:
        raise ValueError("Нужны уникальные положительные candidate IDs и положительный limit")
    if not np.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError("strength должна находиться в [0,1]")
    if np.any(np.isinf(values)) or np.any(values[np.isfinite(values)] < 0):
        raise ValueError("Для multiplicative reranking нужны конечные неотрицательные native scores")
    if np.any(np.isinf(feature)) or np.any((feature[np.isfinite(feature)] < 0) | (feature[np.isfinite(feature)] > 1)):
        raise ValueError("feature должна находиться в [0,1] либо быть NaN")
    result = np.full(values.shape, np.nan)
    for position, row in enumerate(values):
        known = np.flatnonzero(np.isfinite(row))
        chosen = known[np.lexsort((ids[known], -row[known]))[:limit]]
        factors = 1 + strength * np.nan_to_num(feature[position, chosen], nan=0)
        result[position, chosen] = row[chosen] * factors
    return result
