"""Взаимодействия по одному непрерывному фрагменту исходной сессии."""

from bisect import bisect_right
from collections import Counter, defaultdict

from scripts.prepare_interactions import iter_intersections, merge_intervals


def _channel_masks(rows, audit):
    """Объединять coverage можно только для обнаружения разных активных каналов."""
    coverage = defaultdict(list)
    for channel, start, end in rows:
        coverage[channel].append((start, end))
    events = []
    for channel, intervals in coverage.items():
        for start, end in merge_intervals(intervals):
            events.append((start, 1, channel))
            events.append((end, -1, channel))
            audit["unique_user_channel_covered_seconds"] += end - start
    events.sort()
    segments = defaultdict(list)
    active = set()
    previous = None
    position = 0
    while position < len(events):
        moment = events[position][0]
        if previous is not None and previous < moment:
            if len(active) == 1:
                channel = next(iter(active))
                if segments[channel] and segments[channel][-1][1] == previous:
                    segments[channel][-1] = (segments[channel][-1][0], moment)
                else:
                    segments[channel].append((previous, moment))
            elif len(active) > 1:
                audit["cross_channel_ambiguous_user_seconds"] += moment - previous
                audit["cross_channel_ambiguous_user_channel_seconds"] += (moment - previous) * len(active)
        while position < len(events) and events[position][0] == moment:
            _, direction, channel = events[position]
            if direction == 1:
                active.add(channel)
            else:
                active.remove(channel)
            position += 1
        previous = moment
    return {
        channel: ([start for start, _ in intervals], [end for _, end in intervals])
        for channel, intervals in segments.items()
    }


def _session_pieces(start, end, mask):
    if mask is None:
        return
    starts, ends = mask
    position = bisect_right(ends, start)
    while position < len(starts) and starts[position] < end:
        left, right = max(start, starts[position]), min(end, ends[position])
        if left < right:
            yield left, right
        position += 1


def _labelled_pieces(index, start, end, audit):
    """Сшивать только касающиеся сегменты одного эфира внутри одной сессии."""
    pending = None
    covered = 0
    for item, airing, left, right in iter_intersections(index, start, end):
        covered += right - left
        if item <= 0:
            key = "unlabelled_overlap_seconds" if item == 0 else "ambiguous_schedule_overlap_seconds"
            audit[key] += right - left
            if pending is not None:
                yield pending
                pending = None
        elif pending is not None and pending[:2] == (item, airing) and pending[3] == left:
            pending = (item, airing, pending[2], right)
        else:
            if pending is not None:
                yield pending
            pending = (item, airing, left, right)
    if pending is not None:
        yield pending
    audit["outside_schedule_piece_seconds"] += end - start - covered


def _new_audit(thresholds):
    audit = Counter({key: 0 for key in (
        "source_session_rows", "source_session_seconds", "exact_duplicate_session_rows",
        "deduplicated_session_rows", "deduplicated_session_seconds", "continuous_pieces",
        "row_weighted_cross_channel_ambiguous_seconds", "sessions_with_cross_channel_overlap",
        "fully_ambiguous_session_rows", "sessions_split_by_cross_channel_overlap",
        "unique_user_channel_covered_seconds", "cross_channel_ambiguous_user_seconds",
        "cross_channel_ambiguous_user_channel_seconds", "absent_channel_session_rows",
        "cross_channel_mask_evaluated_users",
        "absent_channel_piece_seconds", "outside_schedule_piece_seconds",
        "unlabelled_overlap_seconds", "ambiguous_schedule_overlap_seconds", "matched_labelled_pieces",
    )})
    for percent in thresholds:
        for key in ("qualifying_pieces", "qualified_user_airings", "total_unique_seconds", "watch_count", "user_programme_pairs"):
            audit[f"{key}_{percent}"] = 0
    return audit


def match_sessions(channel_views, index_by_channel, periods, thresholds=(50, 80), exclude_cross_channel=True, qualified_events=None):
    """Не суммировать сессии или разделённые куски для прохождения порога.

    Exact dedup использует исходные (channel, start, end). Перед построением
    межканальной маски и оценкой каждая исходная строка обрезается по периоду.
    Порог строгий, знаменатель — полная длительность canonical airing.
    Union секунд и дедупликация эфиров выполняются только после qualification.
    При exclude_cross_channel=False исходные строки всё равно независимы.
    Межканальные coverage-счётчики вычисляются только при включённой маске;
    cross_channel_mask_evaluated_users отмечает периоды с такой проверкой.
    Optional sink получает один tuple на period/threshold/channel/airing:
    (period, threshold, item, channel, full_start, full_end, watch_start, watch_end).
    """
    thresholds = tuple(dict.fromkeys(thresholds))
    if not thresholds or any(not isinstance(percent, int) or not 0 < percent <= 100 for percent in thresholds):
        raise ValueError("Порог должен быть целым процентом от 1 до 100")
    periods = tuple(periods)
    if len({name for name, _, _ in periods}) != len(periods) or any(lower >= upper for _, lower, upper in periods):
        raise ValueError("Нужны различные имена и непустые периоды")
    result = {name: {percent: {} for percent in thresholds} for name, _, _ in periods}
    audit = {name: _new_audit(thresholds) for name, _, _ in periods}
    period_rows = {name: [] for name, _, _ in periods}
    for channel, intervals in channel_views.items():
        seen = set()
        for start, end in intervals:
            if end <= start:
                continue
            duplicate = (start, end) in seen
            seen.add((start, end))
            for name, lower, upper in periods:
                left, right = max(start, lower), min(end, upper)
                if left >= right:
                    continue
                stats = audit[name]
                stats["source_session_rows"] += 1
                stats["source_session_seconds"] += right - left
                if duplicate:
                    stats["exact_duplicate_session_rows"] += 1
                    continue
                stats["deduplicated_session_rows"] += 1
                stats["deduplicated_session_seconds"] += right - left
                period_rows[name].append((channel, left, right))

    for name, lower, upper in periods:
        rows, stats = period_rows[name], audit[name]
        stats["cross_channel_mask_evaluated_users"] = int(exclude_cross_channel and bool(rows))
        masks = _channel_masks(rows, stats) if exclude_cross_channel else None
        qualified = {percent: defaultdict(list) for percent in thresholds}
        for channel, start, end in rows:
            pieces = list(_session_pieces(start, end, masks.get(channel))) if masks is not None else [(start, end)]
            removed = end - start - sum(right - left for left, right in pieces)
            stats["row_weighted_cross_channel_ambiguous_seconds"] += removed
            stats["sessions_with_cross_channel_overlap"] += removed > 0
            stats["fully_ambiguous_session_rows"] += not pieces
            stats["sessions_split_by_cross_channel_overlap"] += len(pieces) > 1
            stats["continuous_pieces"] += len(pieces)
            index = index_by_channel.get(channel)
            if index is None:
                stats["absent_channel_session_rows"] += 1
                stats["absent_channel_piece_seconds"] += sum(right - left for left, right in pieces)
                continue
            for left, right in pieces:
                for item, airing, overlap_start, overlap_end in _labelled_pieces(index, left, right, stats):
                    stats["matched_labelled_pieces"] += 1
                    duration = index["airing_ends"][airing] - index["airing_starts"][airing]
                    seconds = overlap_end - overlap_start
                    for percent in thresholds:
                        if seconds * 100 > percent * duration:
                            qualified[percent][(channel, airing, item)].append((overlap_start, overlap_end))
                            stats[f"qualifying_pieces_{percent}"] += 1
        for percent in thresholds:
            watched, airings = defaultdict(list), defaultdict(list)
            stats[f"qualified_user_airings_{percent}"] = len(qualified[percent])
            for (channel, airing, item), intervals in qualified[percent].items():
                watched[item].extend(intervals)
                index = index_by_channel[channel]
                if qualified_events is not None:
                    qualified_events.append((
                        name, percent, item, channel,
                        index["airing_starts"][airing], index["airing_ends"][airing],
                        min(start for start, _ in intervals), max(end for _, end in intervals),
                    ))
                airings[item].append((max(lower, index["airing_starts"][airing]), min(upper, index["airing_ends"][airing])))
            for item, intervals in watched.items():
                intervals = merge_intervals(intervals)
                seconds = sum(end - start for start, end in intervals)
                count = len(merge_intervals(airings[item], touching=False))
                result[name][percent][item] = (count, seconds, intervals[-1][1])
                stats[f"total_unique_seconds_{percent}"] += seconds
                stats[f"watch_count_{percent}"] += count
            stats[f"user_programme_pairs_{percent}"] = len(result[name][percent])
    return result, {name: dict(stats) for name, stats in audit.items()}
