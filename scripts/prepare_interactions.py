"""Полный EDA и временные взаимодействия sweet.tv; только стандартная библиотека.

Запуск: .venv/bin/python scripts/prepare_interactions.py
Времена — календарная шкала с проверенной нормализацией часового скачка;
абсолютная timezone не предполагается.
"""

import bisect
import csv
import hashlib
import heapq
import itertools
import json
import os
import platform
import resource
import struct
import subprocess
import sys
import tempfile
import time
from array import array
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUTPUT = DATA / "processed"
RESULT = ROOT / "results" / "data_preparation.json"
EPOCH = datetime(1970, 1, 1)
# Inferred from the training data, not from a presumed timezone: no timestamps
# in 03:xx; 117 view deltas and 394 adjacent-airing deltas identify a +1 h jump.
_gap = datetime(2020, 3, 29, 3) - EPOCH
CLOCK_GAP_START = _gap.days * 86400 + _gap.seconds
CLOCK_GAP_END = CLOCK_GAP_START + 3600
VIEW_COLUMNS = ["user_id", "vsetv_id", "start_time", "stop_time", "duraton"]
SCHEDULE_COLUMNS = [
    "channel_id", "channel_title", "start_time", "duration", "tv_show_title",
    "tv_show_id", "tv_show_category", "tv_show_genre_1", "tv_show_genre_2",
    "tv_show_genre_3", "year_of_production", "director", "actors",
]
META_COLUMNS = SCHEDULE_COLUMNS[4:5] + SCHEDULE_COLUMNS[6:]
SCHEDULE_FILES = (
    "export_arh_11-20-final.csv", "export_arh_21-30-final.csv",
    "export_arh_31-42-final.csv",
)
BUCKETS = 32
RECORD = struct.Struct("<IIqqq")


def wall_seconds(value, schedule=False):
    if schedule:
        value = value[6:10] + "-" + value[3:5] + "-" + value[:2] + value[10:]
    moment = datetime.fromisoformat(value)
    if moment.tzinfo is not None:
        raise ValueError("Ожидалась дата без timezone")
    delta = moment - EPOCH
    seconds = delta.days * 86400 + delta.seconds
    if CLOCK_GAP_START <= seconds < CLOCK_GAP_END:
        raise ValueError(f"Дата попала в эмпирически отсутствующий календарный час: {value}")
    return seconds - 3600 if seconds >= CLOCK_GAP_END else seconds


def calendar_seconds(value):
    return value + 3600 if value >= CLOCK_GAP_START else value


def iso(value):
    return (EPOCH + timedelta(seconds=calendar_seconds(value))).isoformat(sep=" ") if value is not None else None


PERIODS = [
    ("train", wall_seconds("2020-03-09 00:00:00"), wall_seconds("2020-05-18 00:00:00")),
    ("validation", wall_seconds("2020-05-18 00:00:00"), wall_seconds("2020-07-27 00:00:00")),
]
TEST_PERIOD = (wall_seconds("2020-07-27 00:00:00"), wall_seconds("2020-10-19 00:00:00"))


def period_slices(start, end, periods=PERIODS):
    for name, lower, upper in periods:
        left, right = max(start, lower), min(end, upper)
        if left < right:
            yield name, left, right


def merge_intervals(intervals, touching=True):
    merged = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if merged and (start <= merged[-1][1] if touching else start < merged[-1][1]):
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def build_channel_index(records):
    """Sweep-line: один непересекающийся сегмент на момент времени.

    Перекрывающиеся эфиры одного item объединяются, соседние остаются разными.
    Любые разные активные ID, включая 0 + ненулевой ID, считаются неоднозначными.
    """
    records = list(records)
    qa = Counter(source_rows=len(records))
    valid = sorted((item, start, end) for start, end, item in records if end > start and item >= 0)
    qa["invalid_intervals"] = len(records) - len(valid)
    canonical = []
    previous = None
    for item, start, end in valid:
        if previous == (item, start, end):
            qa["duplicate_interval_item_rows"] += 1
        previous = item, start, end
        if canonical and canonical[-1][2] == item and start < canonical[-1][1]:
            left, right, _ = canonical[-1]
            canonical[-1] = (left, max(right, end), item)
            qa["merged_overlapping_same_item_rows"] += 1
        else:
            canonical.append((start, end, item))

    events = []
    for airing, (start, end, _) in enumerate(canonical):
        events.append((start, 1, airing))
        events.append((end, -1, airing))
    events.sort()
    index = {
        "starts": array("q"), "ends": array("q"), "items": array("q"),
        "airings": array("i"),
        "airing_starts": array("q", (row[0] for row in canonical)),
        "airing_ends": array("q", (row[1] for row in canonical)),
    }
    active = set()
    previous_time = None
    position = 0
    while position < len(events):
        moment = events[position][0]
        if previous_time is not None and moment > previous_time and active:
            if len(active) == 1:
                airing = next(iter(active))
                item = canonical[airing][2]
            else:
                airing, item = -1, -1
            index["starts"].append(previous_time)
            index["ends"].append(moment)
            index["items"].append(item)
            index["airings"].append(airing)
            kind = "ambiguous" if item < 0 else "unlabelled" if item == 0 else "labelled"
            qa[kind + "_seconds"] += moment - previous_time
            if item < 0:
                qa["ambiguous_segments"] += 1
                ids = {canonical[key][2] for key in active}
                if 0 in ids:
                    qa["ambiguous_with_zero_seconds"] += moment - previous_time
        while position < len(events) and events[position][0] == moment:
            _, direction, airing = events[position]
            if direction == 1:
                active.add(airing)
            else:
                active.remove(airing)
            position += 1
        previous_time = moment
    qa["canonical_airings"] = len(canonical)
    qa["segments"] = len(index["starts"])
    return index, dict(qa)


def iter_intersections(index, start, end):
    position = bisect.bisect_right(index["ends"], start)
    while position < len(index["starts"]) and index["starts"][position] < end:
        left, right = max(start, index["starts"][position]), min(end, index["ends"][position])
        if left < right:
            yield index["items"][position], index["airings"][position], left, right
        position += 1


THRESHOLD_RULES = (
    "seconds_30", "seconds_60", "seconds_120", "seconds_300",
    "fraction_0.10", "fraction_0.25", "fraction_0.50", "seconds_60_and_fraction_0.10",
)


def match_user(channel_views, index_by_channel, periods=PERIODS, threshold=60, overlap_histograms=None):
    """Порог применяется к уникальным секундам user × channel × canonical airing.

    Секунды одного item на разных каналах также объединяются. watch_count считает
    квалифицировавшиеся эфиры, объединяя строго перекрывающиеся эфиры того же item
    между каналами. Последний момент — исключённая правая граница просмотра.
    """
    overlaps = defaultdict(list)
    diagnostics = {name: Counter() for name, _, _ in periods}
    for channel, intervals in channel_views.items():
        index = index_by_channel.get(channel)
        for start, end in merge_intervals(intervals):
            for period, left, right in period_slices(start, end, periods):
                diag = diagnostics[period]
                diag["unique_user_channel_view_seconds"] += right - left
                if index is None:
                    diag["absent_channel_seconds"] += right - left
                    continue
                covered = 0
                for item, airing, overlap_start, overlap_end in iter_intersections(index, left, right):
                    seconds = overlap_end - overlap_start
                    covered += seconds
                    kind = "ambiguous" if item < 0 else "unlabelled" if item == 0 else "labelled"
                    diag[kind + "_overlap_seconds"] += seconds
                    if item > 0:
                        overlaps[(period, channel, airing, item)].append((overlap_start, overlap_end))
                diag["outside_schedule_seconds"] += right - left - covered

    accepted = defaultdict(lambda: defaultdict(lambda: [[], []]))
    for (period, channel, airing, item), intervals in overlaps.items():
        intervals = merge_intervals(intervals)
        seconds = sum(end - start for start, end in intervals)
        index = index_by_channel[channel]
        airing_start, airing_end = index["airing_starts"][airing], index["airing_ends"][airing]
        duration = airing_end - airing_start
        diag = diagnostics[period]
        diag["matched_labelled_user_airings"] += 1
        if overlap_histograms is not None and period in overlap_histograms:
            overlap_histograms[period][seconds] += 1
        for cut in (30, 60, 120, 300):
            if seconds >= cut:
                diag[f"seconds_{cut}"] += 1
        for fraction in (0.10, 0.25, 0.50):
            if seconds >= fraction * duration:
                diag[f"fraction_{fraction:.2f}"] += 1
        if seconds >= 60 and seconds >= 0.10 * duration:
            diag["seconds_60_and_fraction_0.10"] += 1
        diag["overlap_seconds_sum_before_threshold"] += seconds
        if seconds >= threshold:
            diag["accepted_user_airings"] += 1
            accepted[period][item][0].extend(intervals)
            # Clipping prevents accepted-airing counts from linking periods.
            bounds = next((lower, upper) for name, lower, upper in periods if name == period)
            accepted[period][item][1].append((max(bounds[0], airing_start), min(bounds[1], airing_end)))

    interactions = {name: {} for name, _, _ in periods}
    for period, items in accepted.items():
        for item, (intervals, airings) in items.items():
            intervals = merge_intervals(intervals)
            seconds = sum(end - start for start, end in intervals)
            count = len(merge_intervals(airings, touching=False))
            interactions[period][item] = (count, seconds, max(end for _, end in intervals))
    return interactions, diagnostics


def quantiles(histogram):
    """Точные линейно интерполированные квантили по частотам, без выборки."""
    count = sum(histogram.values())
    if not count:
        return {"count": 0}
    values = sorted(histogram.items())
    ranks = []
    for probability in (0, 0.01, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1):
        rank = probability * (count - 1)
        ranks.append((probability, int(rank), min(int(rank) + 1, count - 1), rank % 1))
    needed = {rank for _, lower, upper, _ in ranks for rank in (lower, upper)}
    selected, cumulative = {}, 0
    for value, frequency in values:
        for rank in needed:
            if cumulative <= rank < cumulative + frequency:
                selected[rank] = value
        cumulative += frequency
    result = {"count": count, "mean": sum(key * value for key, value in values) / count}
    for probability, lower, upper, fraction in ranks:
        result[f"p{probability * 100:g}"] = selected[lower] + fraction * (selected[upper] - selected[lower])
    return result


def fingerprints(row):
    # Length-prefixing makes the encoding unambiguous even for embedded separators.
    packed = bytearray()
    for field in row:
        encoded = field.encode("utf-8")
        packed.extend(struct.pack("<I", len(encoded)))
        packed.extend(encoded)
    return hashlib.blake2b(packed, digest_size=16).digest()


def exact_duplicate_count(path, repeated):
    """Подтверждаем кандидатов hash-дубликатов сравнением полных CSV-строк."""
    if not repeated:
        return 0
    counts = Counter()
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        next(reader)
        for row in reader:
            if fingerprints(row) in repeated:
                counts[tuple(row)] += 1
    return sum(count - 1 for count in counts.values())


def new_metadata():
    return defaultdict(lambda: [set() for _ in META_COLUMNS])


def metadata_summary(metadata):
    result = {}
    for position, column in enumerate(META_COLUMNS):
        conflicts = [(item, values[position]) for item, values in metadata.items() if len(values[position]) > 1]
        result[column] = {
            "ids_with_multiple_nonempty_values": len(conflicts),
            "examples": [{"tv_show_id": item, "values": sorted(values)[:5]} for item, values in sorted(conflicts)[:3]],
        }
    titles = defaultdict(set)
    for item, attributes in metadata.items():
        for title in attributes[0]:
            titles[title].add(item)
    result["same_title_multiple_ids"] = {
        "titles": sum(len(ids) > 1 for ids in titles.values()),
        "examples": [{"title": title, "ids": sorted(ids)} for title, ids in sorted(titles.items()) if len(ids) > 1][:5],
    }
    return result


def read_schedule(path, global_metadata, candidates):
    started = time.perf_counter()
    records = defaultdict(lambda: (array("q"), array("q"), array("q")))
    stats = Counter()
    empty, labelled_empty, whitespace = Counter(), Counter(), Counter()
    examples = {column: [] for column in SCHEDULE_COLUMNS}
    categories, duration_hist, labelled_duration_hist, item_ids, channel_titles = Counter(), Counter(), Counter(), set(), defaultdict(set)
    metadata = new_metadata()
    seen, repeated = set(), set()
    minimum, maximum, maximum_end = None, None, None
    start_hours = Counter()
    year_forms = Counter()
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        if header != SCHEDULE_COLUMNS:
            raise ValueError(f"Неожиданные столбцы {path.name}: {header}")
        for row in reader:
            if len(row) != len(header):
                raise ValueError(f"Неверное число полей в {path.name}, строка {reader.line_num}")
            stats["rows"] += 1
            digest = fingerprints(row)
            if digest in seen:
                repeated.add(digest)
            seen.add(digest)
            channel, start, duration, item = int(row[0]), wall_seconds(row[2], True), int(row[3]), int(row[5])
            end = start + duration
            if str(channel) != row[0] or str(item) != row[5]:
                stats["noncanonical_integer_identifiers"] += 1
            minimum = start if minimum is None else min(minimum, start)
            maximum = start if maximum is None else max(maximum, start)
            maximum_end = end if maximum_end is None else max(maximum_end, end)
            start_hours[iso(start)[:13]] += 1
            duration_hist[duration] += 1
            if item > 0:
                labelled_duration_hist[duration] += 1
            categories[row[6]] += 1
            channel_titles[channel].add(row[1])
            item_ids.add(item)
            stats["labelled_rows" if item > 0 else "zero_id_rows" if item == 0 else "negative_id_rows"] += 1
            stats["zero_duration_rows"] += duration == 0
            stats["negative_duration_rows"] += duration < 0
            stats["duration_over_24h_rows"] += duration > 86400
            stats["cross_calendar_day_rows"] += calendar_seconds(start) // 86400 != calendar_seconds(end - 1) // 86400 if duration > 0 else False
            for column, value in zip(header, row):
                empty[column] += value == ""
                labelled_empty[column] += item > 0 and value == ""
                whitespace[column] += value != "" and not value.strip()
                if value and len(examples[column]) < 3 and value not in examples[column]:
                    examples[column].append(value)
            year_forms["missing" if not row[10] else "integer_year" if row[10].isdigit() else "range_or_text"] += 1
            if item > 0:
                for position, column in enumerate(META_COLUMNS):
                    value = row[header.index(column)].strip()
                    if value:
                        metadata[item][position].add(value)
                        global_metadata[item][position].add(value)
                if duration > 0:
                    for period, lower, upper in PERIODS + [("test", *TEST_PERIOD)]:
                        if start < upper and end > lower:
                            candidates[period].add(item)
            records[channel][0].append(start)
            records[channel][1].append(end)
            records[channel][2].append(item)
    del seen
    stats["exact_duplicate_rows"] = exact_duplicate_count(path, repeated)
    overlapping_pairs = 0
    gap_naive_deltas, gap_normalised_deltas = Counter(), Counter()
    for starts, ends, _ in records.values():
        active = []
        ordered = sorted(zip(starts, ends))
        for (start, end), (next_start, _) in zip(ordered, ordered[1:]):
            # Sparse channels may have genuine multi-day gaps. Only an airing
            # physically reaching the clock jump can establish its size.
            if start < CLOCK_GAP_START <= end and next_start >= CLOCK_GAP_START:
                gap_naive_deltas[calendar_seconds(next_start) - calendar_seconds(start) - (end - start)] += 1
                gap_normalised_deltas[next_start - end] += 1
        for start, end in ordered:
            if end <= start:
                continue
            while active and active[0] <= start:
                heapq.heappop(active)
            overlapping_pairs += len(active)
            heapq.heappush(active, end)
    stats["overlapping_schedule_row_pairs"] = overlapping_pairs
    result = dict(stats)
    result.update({
        "columns": header,
        "logical_types": {column: "int" if column in ("channel_id", "duration", "tv_show_id") else "naive datetime, DD.MM.YYYY HH:MM:SS" if column == "start_time" else "str" for column in header},
        "empty_fields": dict(empty), "empty_fields_labelled_rows": dict(labelled_empty),
        "whitespace_only_fields": dict(whitespace), "examples": examples,
        "channels": len(records), "unique_ids_including_zero": len(item_ids),
        "unique_nonzero_programmes": len(item_ids - {0}),
        "zero_id_row_fraction": stats["zero_id_rows"] / stats["rows"],
        "start_min": iso(minimum), "start_max": iso(maximum), "end_max": iso(maximum_end),
        "duration_seconds": quantiles(duration_hist), "categories": dict(categories),
        "labelled_airing_duration_seconds": quantiles(labelled_duration_hist),
        "labelled_airings_shorter_than_60_seconds": sum(count for duration, count in labelled_duration_hist.items() if 0 < duration < 60),
        "year_value_formats": dict(year_forms),
        "clock_gap_crossing_adjacent_airings_naive_delta_seconds": dict(gap_naive_deltas),
        "clock_gap_crossing_adjacent_airings_normalised_delta_seconds": dict(gap_normalised_deltas),
        "metadata_consistency": metadata_summary(metadata),
        "channels_with_multiple_titles": sum(len(titles) > 1 for titles in channel_titles.values()),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    })
    return result, records, item_ids - {0}, start_hours


def read_views(work):
    started = time.perf_counter()
    path = DATA / "dataset11-30.csv"
    streams = [(work / f"views_{number:02}.bin").open("wb") for number in range(BUCKETS)]
    user_to_number, users = {}, []
    stats, channels, empty = Counter(), Counter(), Counter()
    raw_duration, timestamp_duration, elapsed_duration, mismatch_delta, start_hours = Counter(), Counter(), Counter(), Counter(), Counter()
    period_stats = {name: Counter() for name, _, _ in PERIODS}
    period_durations = {name: Counter() for name, _, _ in PERIODS}
    period_users = {name: set() for name, _, _ in PERIODS}
    minimum, maximum_start, maximum_end, previous_start = None, None, None, None
    mismatch_examples = []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        if header != VIEW_COLUMNS:
            raise ValueError(f"Неожиданные столбцы журнала: {header}")
        for row in reader:
            if len(row) != 5:
                raise ValueError(f"Неверное число полей журнала, строка {reader.line_num}")
            stats["rows"] += 1
            for column, value in zip(header, row):
                empty[column] += value == ""
            user = row[0]  # Никогда не преобразуется через float или int.
            if user not in user_to_number:
                user_to_number[user] = len(users)
                users.append(user)
            user_number = user_to_number[user]
            channel, start, end, duration = int(row[1]), wall_seconds(row[2]), wall_seconds(row[3]), int(row[4])
            if str(channel) != row[1]:
                stats["noncanonical_channel_identifiers"] += 1
            channels[channel] += 1
            raw_duration[duration] += 1
            naive_duration = calendar_seconds(end) - calendar_seconds(start)
            timestamp_duration[naive_duration] += 1
            elapsed_duration[end - start] += 1
            start_hours[row[2][:13]] += 1
            minimum = start if minimum is None else min(minimum, start)
            maximum_start = start if maximum_start is None else max(maximum_start, start)
            maximum_end = end if maximum_end is None else max(maximum_end, end)
            stats["nonmonotonic_start_rows"] += previous_start is not None and start < previous_start
            previous_start = start
            stats["zero_intervals"] += end == start
            stats["negative_intervals"] += end < start
            stats["zero_recorded_duration"] += duration == 0
            stats["negative_recorded_duration"] += duration < 0
            stats["intervals_over_24h"] += end - start > 86400
            stats["intervals_over_48h"] += end - start > 172800
            stats["cross_calendar_day_rows"] += calendar_seconds(start) // 86400 != calendar_seconds(end - 1) // 86400 if end > start else False
            stats["duration_mismatches_after_clock_normalisation"] += duration != end - start
            if duration != naive_duration:
                stats["duration_timestamp_mismatches"] += 1
                mismatch_delta[duration - naive_duration] += 1
                if len(mismatch_examples) < 5:
                    mismatch_examples.append({"start": row[2], "stop": row[3], "duraton": duration, "naive_calendar_seconds": naive_duration, "normalised_elapsed_seconds": end - start})
            if start < PERIODS[0][1] or end > PERIODS[-1][2]:
                stats["rows_extending_outside_history_window"] += 1
            if start < PERIODS[0][2] < end:
                stats["sessions_crossing_train_validation_cutoff"] += 1
            slices = list(period_slices(start, end))
            for period, left, right in slices:
                period_stats[period]["source_rows"] += 1
                period_stats[period]["raw_view_seconds"] += right - left
                period_durations[period][right - left] += 1
                period_users[period].add(user_number)
            if end > start and slices:
                streams[user_number % BUCKETS].write(RECORD.pack(user_number, channel, start, end, duration))
    for stream in streams:
        stream.close()
    result = dict(stats)
    result.update({
        "columns": header,
        "logical_types": {"user_id": "str (exact CSV value)", "vsetv_id": "int", "start_time": "naive datetime, YYYY-MM-DD HH:MM:SS", "stop_time": "naive datetime, YYYY-MM-DD HH:MM:SS", "duraton": "int, seconds"},
        "users": len(users), "channels": len(channels), "empty_fields": dict(empty),
        "user_id_lengths": dict(Counter(map(len, users))),
        "start_min": iso(minimum), "start_max": iso(maximum_start), "stop_max": iso(maximum_end),
        "recorded_duration_seconds": quantiles(raw_duration),
        "timestamp_interval_seconds": quantiles(timestamp_duration),
        "clock_normalised_interval_seconds": quantiles(elapsed_duration),
        "duration_mismatch_deltas_seconds": dict(mismatch_delta), "duration_mismatch_examples": mismatch_examples,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    })
    return result, users, channels, period_stats, period_durations, period_users, start_hours


def read_submission(users):
    path = DATA / "submission.csv"
    submission_users, rows, lengths, malformed = [], 0, Counter(), 0
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        if header != ["user_id", "tv_show_id"]:
            raise ValueError(f"Неожиданный submission header: {header}")
        for row in reader:
            rows += 1
            submission_users.append(row[0])
            ids = row[1].split()
            lengths[len(ids)] += 1
            malformed += len(ids) != 5 or not all(value.isdigit() for value in ids)
    unique = set(submission_users)
    log_users = set(users)
    return {
        "rows": rows, "columns": header, "unique_users": len(unique),
        "duplicate_user_rows": rows - len(unique), "programme_ids_per_row": dict(lengths),
        "malformed_rows": malformed,
        "users_present_in_view_log": len(unique & log_users),
        "users_absent_from_view_log": len(unique - log_users),
        "view_log_users_absent_from_submission": len(log_users - unique),
        "user_id_type": "str, exact CSV value; no numeric conversion",
    }, unique


def source_coverage(rows, indices, period_stats, coverage_details):
    for _, channel, start, end, _ in rows:
        for period, left, right in period_slices(start, end):
            stats = period_stats[period]
            index = indices.get(channel)
            if index is None:
                stats["source_rows_absent_channel"] += 1
                continue
            stats["source_rows_channel_present"] += 1
            kinds = set()
            for item, _, _, _ in iter_intersections(index, left, right):
                kinds.add("ambiguous" if item < 0 else "unlabelled" if item == 0 else "labelled")
            stats["source_rows_with_schedule_overlap"] += bool(kinds)
            stats["source_rows_without_schedule_overlap"] += not kinds
            if not kinds:
                detail = coverage_details[period]
                detail["no_overlap_by_channel"][channel] += 1
                detail["no_overlap_by_day"][iso(left)[:10]] += 1
                if not index["starts"]:
                    reason = "channel_has_no_valid_airings"
                elif right <= index["starts"][0]:
                    reason = "before_first_channel_airing"
                elif left >= index["ends"][-1]:
                    reason = "after_last_channel_airing"
                else:
                    reason = "internal_channel_schedule_gap"
                detail["no_overlap_reason"][reason] += 1
            for kind in kinds:
                stats["source_rows_with_" + kind + "_overlap"] += 1
            stats["source_rows_only_unlabelled"] += kinds == {"unlabelled"}


def process_buckets(work, users, indices, period_stats, period_users, submission_users, threshold):
    started = time.perf_counter()
    writers, streams = {}, {}
    fields = ["user_id", "tv_show_id", "watch_count", "total_watch_seconds", "last_watch_time", "period"]
    summaries = {name: Counter() for name, _, _ in PERIODS}
    eligible_users = {name: set() for name, _, _ in PERIODS}
    item_users = {name: Counter() for name, _, _ in PERIODS}
    item_watches = {name: Counter() for name, _, _ in PERIODS}
    user_interactions = {name: Counter() for name, _, _ in PERIODS}
    user_watches = {name: Counter() for name, _, _ in PERIODS}
    overlap_histograms = {"train": Counter()}
    coverage_details = {
        name: {key: Counter() for key in ("no_overlap_by_channel", "no_overlap_by_day", "no_overlap_reason")}
        for name, _, _ in PERIODS
    }
    for name, _, _ in PERIODS:
        stream = (OUTPUT / f"{name}_interactions.csv").open("w", encoding="utf-8", newline="")
        streams[name] = stream
        writers[name] = csv.writer(stream)
        writers[name].writerow(fields)
    duplicate_rows, duplicates_by_period, normalised_session_count = 0, Counter(), 0
    for bucket in range(BUCKETS):
        raw = (work / f"views_{bucket:02}.bin").read_bytes()
        rows = list(RECORD.iter_unpack(raw))
        del raw
        rows.sort()
        previous = None
        for row in rows:
            if row == previous:
                duplicate_rows += 1
                for name, _, _ in period_slices(row[2], row[3]):
                    duplicates_by_period[name] += 1
            previous = row
        source_coverage(rows, indices, period_stats, coverage_details)
        for user_number, group in itertools.groupby(rows, key=lambda row: row[0]):
            channel_views = defaultdict(list)
            for _, channel, start, end, _ in group:
                channel_views[channel].append((start, end))
            normalised_session_count += sum(len(merge_intervals(intervals)) for intervals in channel_views.values())
            interactions, diagnostics = match_user(channel_views, indices, threshold=threshold, overlap_histograms=overlap_histograms)
            for name, items in interactions.items():
                summaries[name].update(diagnostics[name])
                if not items:
                    continue
                eligible_users[name].add(user_number)
                user_interactions[name][len(items)] += 1
                user_watches[name][sum(values[0] for values in items.values())] += 1
                for item, (count, seconds, last) in sorted(items.items()):
                    writers[name].writerow((users[user_number], item, count, seconds, iso(last), name))
                    summaries[name]["user_programme_pairs"] += 1
                    summaries[name]["watch_count"] += count
                    summaries[name]["total_watch_seconds"] += seconds
                    item_users[name][item] += 1
                    item_watches[name][item] += count
        del rows
        if (bucket + 1) % 8 == 0:
            print(f"Обработано блоков пользователей: {bucket + 1}/{BUCKETS}", flush=True)
    for stream in streams.values():
        stream.close()
    result = {}
    for name, _, _ in PERIODS:
        active = eligible_users[name]
        population = period_users[name]
        all_distribution = user_interactions[name].copy()
        all_distribution[0] += len(population - active)
        active_ids = {users[number] for number in active}
        result[name] = {
            **dict(period_stats[name]), **dict(summaries[name]),
            "exact_duplicate_source_rows": duplicates_by_period[name],
            "source_users": len(population), "users_with_interactions": len(active),
            "users_without_interactions": len(population - active),
            "users_without_interactions_fraction": len(population - active) / len(population),
            "users_with_interactions_fraction": len(active) / len(population),
            "view_log_users_without_period_history": len(users) - len(active),
            "view_log_users_without_period_history_fraction": (len(users) - len(active)) / len(users),
            "submission_users_with_interactions": len(submission_users & active_ids),
            "submission_users_without_interactions": len(submission_users - active_ids),
            "unique_interacted_programmes": len(item_users[name]),
            "programme_count_per_active_user": quantiles(user_interactions[name]),
            "programme_count_per_source_user_including_zero": quantiles(all_distribution),
            "watch_count_per_active_user": quantiles(user_watches[name]),
            "user_count_per_interacted_programme": quantiles(Counter(item_users[name].values())),
            "watch_count_per_interacted_programme": quantiles(Counter(item_watches[name].values())),
            "no_schedule_overlap_diagnostics": {
                "reasons": dict(coverage_details[name]["no_overlap_reason"]),
                "by_day": dict(sorted(coverage_details[name]["no_overlap_by_day"].items())),
                "top_channels": [
                    {"channel_id": channel, "source_rows": count,
                     "first_airing_start": iso(indices[channel]["starts"][0]) if indices[channel]["starts"] else None,
                     "last_airing_end": iso(indices[channel]["ends"][-1]) if indices[channel]["ends"] else None}
                    for channel, count in coverage_details[name]["no_overlap_by_channel"].most_common(15)
                ],
            },
        }
        result[name]["overlapping_view_seconds_removed"] = result[name]["raw_view_seconds"] - result[name]["unique_user_channel_view_seconds"]
        denominator = period_stats[name]["source_rows"]
        for key in ("source_rows_channel_present", "source_rows_with_schedule_overlap", "source_rows_with_labelled_overlap"):
            result[name][key + "_fraction"] = period_stats[name][key] / denominator
    train_active, validation_active = eligible_users["train"], eligible_users["validation"]
    combined_ids = {users[number] for number in train_active | validation_active}
    cold_start = {
        "submission_users_total": len(submission_users),
        "submission_without_train_history": len(submission_users - {users[number] for number in train_active}),
        "submission_without_history_in_either_saved_period": len(submission_users - combined_ids),
        "validation_truth_users_without_train_history": len(validation_active - train_active),
        "validation_source_users_without_train_history": len(period_users["validation"] - train_active),
        "validation_truth_programmes_absent_from_train_interactions": len(item_users["validation"].keys() - item_users["train"].keys()),
    }
    analysis = {
        "scope": "train only; validation statistics are not used to select the threshold",
        "matched_user_airing_overlap_seconds": quantiles(overlap_histograms["train"]),
        "accepted_user_airings_by_rule": {rule: summaries["train"][rule] for rule in THRESHOLD_RULES},
        "unit": "sum of unique observed seconds for one user and one canonical airing within train",
    }
    return result, cold_start, analysis, {
        "exact_duplicate_view_rows": duplicate_rows,
        "normalised_user_channel_sessions": normalised_session_count,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }


def file_sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_outputs(period_summaries, candidates, known_users):
    checks = {}
    known_users = set(known_users)
    for name, lower, upper in PERIODS:
        # Output is grouped by user; checking only the current user's items is
        # sufficient and avoids keeping 1.5 million output keys in RAM.
        seen_users, current_items, current_user = set(), set(), None
        row_count, seconds, watches = 0, 0, 0
        path = OUTPUT / f"{name}_interactions.csv"
        with path.open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                item = int(row["tv_show_id"])
                user = row["user_id"]
                if user != current_user:
                    assert user not in seen_users, f"Повтор группы пользователя в {name}"
                    seen_users.add(user)
                    current_items.clear()
                    current_user = user
                if item in current_items or user not in known_users:
                    raise AssertionError(f"Повтор пары или неизвестный user_id в {name}: {(user, item)}")
                current_items.add(item)
                row_count += 1
                count, total, last = int(row["watch_count"]), int(row["total_watch_seconds"]), wall_seconds(row["last_watch_time"])
                assert item > 0 and item in candidates[name]
                assert count > 0 and total >= 60 and lower < last <= upper
                assert row["period"] == name
                seconds += total
                watches += count
        assert row_count == period_summaries[name]["user_programme_pairs"]
        assert seconds == period_summaries[name]["total_watch_seconds"]
        assert watches == period_summaries[name]["watch_count"]
        checks[name] = {
            "rows": row_count, "unique_user_programme_pairs": True,
            "known_string_user_ids": True, "positive_candidate_ids_only": True,
            "positive_counts_and_seconds": True, "last_watch_within_period": True,
            "aggregate_totals_reconciled": True,
            "sha256": file_sha256(path), "bytes": path.stat().st_size,
        }
    return checks


def main():
    started = time.perf_counter()
    cpu_started = time.process_time()
    stage_peak_mib = {}
    def note_peak(stage):
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        stage_peak_mib[stage] = round(peak / (1024**2 if sys.platform == "darwin" else 1024), 2)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    RESULT.parent.mkdir(exist_ok=True)
    test = subprocess.run(
        [sys.executable, "-B", "-m", "unittest", "discover", "-s", str(ROOT / "tests"), "-v"],
        cwd=ROOT, capture_output=True, text=True,
    )
    if test.returncode:
        raise RuntimeError(test.stdout + test.stderr)
    tests = {"passed": True, "command": ".venv/bin/python -B -m unittest discover -s tests -v", "output": test.stderr.strip()}
    source_names = ["dataset11-30.csv", *SCHEDULE_FILES, "submission.csv"]
    source_hashes = {name: {"bytes": (DATA / name).stat().st_size, "sha256": file_sha256(DATA / name)} for name in source_names}
    candidates = {"train": set(), "validation": set(), "test": set()}
    global_metadata = new_metadata()
    history = defaultdict(lambda: (array("q"), array("q"), array("q")))
    schedule_stats, programme_sets, schedule_hours, schedule_bounds = {}, [], Counter(), []
    for number, name in enumerate(SCHEDULE_FILES):
        print(f"Полный EDA расписания: {name}", flush=True)
        stats, records, items, hours = read_schedule(DATA / name, global_metadata, candidates)
        schedule_stats[name] = stats
        programme_sets.append(items)
        schedule_hours.update(hours)
        schedule_bounds.append({channel: (min(values[0]), max(values[1])) for channel, values in records.items()})
        if number < 2:
            for channel, values in records.items():
                for target, incoming in zip(history[channel], values):
                    target.extend(incoming)
        del records
        note_peak("schedule_" + str(number + 1))
    metadata = metadata_summary(global_metadata)
    del global_metadata
    index_started = time.perf_counter()
    indices, index_qa = {}, Counter()
    for channel in list(history):
        values = history.pop(channel)
        index, qa = build_channel_index(zip(*values))
        indices[channel] = index
        index_qa.update(qa)
    del history
    index_elapsed = time.perf_counter() - index_started
    note_peak("history_index")
    boundary_checks = []
    for before, after in zip(schedule_bounds, schedule_bounds[1:]):
        common = before.keys() & after.keys()
        boundary_checks.append({
            "common_channels": len(common),
            "touching_channels": sum(before[channel][1] == after[channel][0] for channel in common),
            "overlapping_channels": sum(before[channel][1] > after[channel][0] for channel in common),
            "gap_channels": sum(before[channel][1] < after[channel][0] for channel in common),
        })
    print("Индекс истории построен; полный EDA и разбиение журнала на компактные блоки", flush=True)
    with tempfile.TemporaryDirectory(prefix=".prepare-", dir=OUTPUT) as temporary:
        work = Path(temporary)
        views, users, channels, period_stats, duration_histograms, period_users, view_hours = read_views(work)
        note_peak("view_log_and_spool")
        gap_evidence = schedule_stats[SCHEDULE_FILES[0]]
        assert set(gap_evidence["clock_gap_crossing_adjacent_airings_naive_delta_seconds"]) == {3600}, gap_evidence["clock_gap_crossing_adjacent_airings_naive_delta_seconds"]
        assert set(gap_evidence["clock_gap_crossing_adjacent_airings_normalised_delta_seconds"]) == {0}, gap_evidence["clock_gap_crossing_adjacent_airings_normalised_delta_seconds"]
        assert views["duration_mismatches_after_clock_normalisation"] == 0
        spool_bytes = sum(path.stat().st_size for path in work.iterdir())
        submission, submission_users = read_submission(users)
        training_distribution = quantiles(duration_histograms["train"])
        # The one-minute rule is fixed from train duration evidence, before matching
        # any validation ground truth. All alternatives are diagnostics, not tuning.
        threshold = 60
        if training_distribution.get("p50", 0) < threshold:
            raise ValueError("Медиана train ниже выбранного порога; требуется пересмотр правила")
        print("Train duration evidence:", json.dumps(training_distribution, ensure_ascii=False), flush=True)
        print("Правило зафиксировано до построения validation: >= 60 уникальных секунд на эфир", flush=True)
        summaries, cold_start, threshold_analysis, processing = process_buckets(
            work, users, indices, period_stats, period_users, submission_users, threshold,
        )
        note_peak("interaction_aggregation")
    output_checks = verify_outputs(summaries, candidates, users)
    note_peak("output_validation")
    for name, values in summaries.items():
        covered_seconds = sum(values.get(key, 0) for key in (
            "labelled_overlap_seconds", "unlabelled_overlap_seconds", "ambiguous_overlap_seconds",
            "absent_channel_seconds", "outside_schedule_seconds",
        ))
        assert covered_seconds == values["unique_user_channel_view_seconds"], name
    for name in source_names:
        assert file_sha256(DATA / name) == source_hashes[name]["sha256"], name
    absent = sorted(channels.keys() - indices.keys())
    channel_check = {
        "view_log_channels": len(channels), "history_schedule_channels": len(indices),
        "common_channels": len(channels.keys() & indices.keys()),
        "absent_view_channels": absent,
        "view_rows_on_absent_channels": sum(channels[channel] for channel in absent),
        "view_rows_on_present_channels": sum(count for channel, count in channels.items() if channel in indices),
        "view_rows_on_absent_channels_fraction": sum(channels[channel] for channel in absent) / views["rows"],
        "join_rule": "int(vsetv_id) == int(channel_id); no remapping",
    }
    for name in summaries:
        summaries[name]["candidate_programmes"] = len(candidates[name])
        summaries[name]["interacted_candidate_fraction"] = summaries[name]["unique_interacted_programmes"] / len(candidates[name])
    expected = {"dataset11-30.csv": 5821987, SCHEDULE_FILES[0]: 1074235, SCHEDULE_FILES[1]: 1074394, SCHEDULE_FILES[2]: 1295378}
    observed = {"dataset11-30.csv": views["rows"], **{name: stats["rows"] for name, stats in schedule_stats.items()}}
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_bytes = peak if sys.platform == "darwin" else peak * 1024
    result = {
        "schema_version": 1,
        "parameters": {
            "implementation": "Python standard library only; no DuckDB, pandas, numpy or ML dependencies",
            "input_root": "data/", "output_root": "data/processed/", "user_id_type": "str",
            "join": "channel equality and half-open interval overlap",
            "index": "per-channel sweep-line canonicalisation; disjoint sorted arrays; bisect then output-sensitive scan",
            "index_complexity": "O(S log S) preprocessing; O(log S_channel + number_of_intersected_segments) per view",
            "source_of_view_duration": "stop_time - start_time after the empirically verified clock-gap normalisation; original duraton is retained for EDA",
            "time_coordinate": "observed calendar coordinates with the missing 2020-03-29 03:00–04:00 hour removed; both sources use the same transform; no absolute timezone assigned",
            "clock_gap_normalisation": "f(t)=naive_calendar_seconds(t)-3600 for t>=2020-03-29 04:00; identity earlier; timestamps in the missing hour are rejected",
            "seconds_threshold": threshold, "threshold_applies_to": "unique seconds per user and canonical airing, separately in each period",
            "threshold_rationale": f"Train median clipped session is {training_distribution['p50']:g} s. The original log has a minimum session duration of 61 s; the 60 s cut removes brief incidental programme overlaps, rather than claiming to remove source sessions below one minute. No labelled training airing is shorter than 60 s. Unlike a fraction-only cut, the rule does not penalise long programmes proportionally. Chosen using train only; no MAP or validation-based tuning.",
            "zero_id_policy": "coverage only; never an item",
            "conflicting_schedule_policy": "all conflicting simultaneous item IDs, including zero+labelled, are ambiguous; excluded from item seconds",
            "view_duplicates_policy": "union overlapping/touching sessions per user/channel; do not count seconds twice",
            "schedule_duplicates_policy": "union strictly overlapping same-item airings on one channel; adjacent airings remain distinct",
            "cross_channel_same_item_policy": "union watched seconds; merge strictly overlapping accepted airing spans for watch_count",
            "last_watch_time_semantics": "exclusive end of the last accepted observed interval, not a programme end prediction",
            "period_boundaries": {name: {"start_inclusive": iso(lower), "end_exclusive": iso(upper)} for name, lower, upper in PERIODS + [("test", *TEST_PERIOD)]},
            "split_policy": "clip views and overlaps at calendar cutoffs; train/validation use union of the first two schedule files",
            "candidate_policy": "positive-duration nonzero labelled airings intersecting the calendar period across all schedule files",
            "user_buckets": BUCKETS, "temporary_record_bytes": RECORD.size,
            "deterministic": True, "randomness": "none",
        },
        "inputs": source_hashes,
        "eda": {
            "view_log": views, "schedules": schedule_stats, "submission_template": submission,
            "programme_id_intersections": {
                "train_file_and_validation_file": len(programme_sets[0] & programme_sets[1]),
                "train_file_and_test_file": len(programme_sets[0] & programme_sets[2]),
                "validation_file_and_test_file": len(programme_sets[1] & programme_sets[2]),
                "all_three_files": len(set.intersection(*programme_sets)),
                "all_files_union": len(set.union(*programme_sets)),
                "validation_file_not_in_train_file": len(programme_sets[1] - programme_sets[0]),
                "test_file_not_in_history_files": len(programme_sets[2] - (programme_sets[0] | programme_sets[1])),
            },
            "candidate_intersections_by_calendar_period": {
                "train_and_validation": len(candidates["train"] & candidates["validation"]),
                "train_and_test": len(candidates["train"] & candidates["test"]),
                "validation_and_test": len(candidates["validation"] & candidates["test"]),
                "all_three": len(set.intersection(*candidates.values())),
                "validation_not_in_train": len(candidates["validation"] - candidates["train"]),
                "test_not_in_history": len(candidates["test"] - candidates["train"] - candidates["validation"]),
            },
            "global_metadata_consistency": metadata,
            "channel_correspondence": channel_check,
            "schedule_file_boundaries_per_channel": boundary_checks,
            "history_schedule_index": dict(index_qa),
            "timezone_evidence": {
                "absolute_timezone": "unknown; CSV timestamps contain no offsets or zone name",
                "no_global_shift_applied": True,
                "evidence_scope": "matching missing hour, all 117 view duration discrepancies and all 394 crossing adjacent-airing gaps; not proof of a named timezone or content-level clock synchronisation",
                "empirical_clock_gap_normalisation_applied_to_both_sources": True,
                "duration_mismatches_before": views.get("duration_timestamp_mismatches", 0),
                "duration_mismatches_after": views.get("duration_mismatches_after_clock_normalisation", 0),
                "crossing_schedule_adjacencies": sum(gap_evidence["clock_gap_crossing_adjacent_airings_naive_delta_seconds"].values()),
                "naive_schedule_adjacency_gaps_seconds": gap_evidence["clock_gap_crossing_adjacent_airings_naive_delta_seconds"],
                "normalised_schedule_adjacency_gaps_seconds": gap_evidence["clock_gap_crossing_adjacent_airings_normalised_delta_seconds"],
                "2020_03_29_view_starts_by_hour": {f"{hour:02}": view_hours[f"2020-03-29 {hour:02}"] for hour in range(7)},
                "2020_03_29_schedule_starts_by_hour": {f"{hour:02}": schedule_hours[f"2020-03-29 {hour:02}"] for hour in range(7)},
            },
            "comparison_with_exam": {
                "expected_rows": expected, "observed_rows": observed, "row_counts_match": expected == observed,
                "programme_counts_in_exam_include_zero": True,
                "duraton_not_always_equal_to_timestamp_difference": views.get("duration_timestamp_mismatches", 0),
            },
        },
        "threshold_analysis_train_only": {**threshold_analysis, "source_session_seconds": training_distribution},
        "interactions": summaries, "cold_start": cold_start,
        "candidate_programmes_by_calendar_period": {name: len(items) for name, items in candidates.items()},
        "checks": {"unit_tests": tests, "output_validation": output_checks, "input_hashes_unchanged": True, "no_test_viewing_used": True, "unique_view_seconds_fully_partitioned_without_double_counting": True, "clock_gap_evidence_verified": True},
        "runtime": {
            "wall_seconds": round(time.perf_counter() - started, 3),
            "cpu_seconds": round(time.process_time() - cpu_started, 3),
            "peak_rss_bytes": peak_bytes, "peak_rss_mib": round(peak_bytes / 1024**2, 2),
            "peak_rss_method": "resource.getrusage(RUSAGE_SELF).ru_maxrss; macOS bytes, Linux KiB converted to bytes",
            "cumulative_peak_rss_mib_by_stage": stage_peak_mib,
            "schedule_index_seconds": round(index_elapsed, 3), "bucket_processing": processing,
            "temporary_binary_spool_bytes": spool_bytes,
            "python": platform.python_version(), "platform": platform.platform(),
            "logical_cpus": os.cpu_count(), "additional_dependencies": [],
        },
        "limitations": [
            "Absolute timezone cannot be established from these CSVs; no timezone name or UTC conversion is asserted. The one-hour spring clock jump is normalised using training-data evidence in both sources.",
            "Most airtime has tv_show_id=0; absence of a labelled interaction is not proof of no watching.",
            "Different titles/metadata under one programme ID may represent episodes or data inconsistency; IDs are not merged by title.",
            "A qualifying overlap is a proxy for watching, not evidence of enjoyment.",
            "Final training on weeks11–30 should recompute airing thresholds over the combined window; simply adding two split aggregates may miss boundary fragments.",
            "Resource measurements cover this full profiling/preparation process and are not comparable to a bare CSV read benchmark.",
        ],
    }
    temporary_result = RESULT.with_suffix(".json.tmp")
    temporary_result.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary_result.replace(RESULT)
    print("Результат:", RESULT.relative_to(ROOT), flush=True)
    print(json.dumps({"interactions": {name: {key: values[key] for key in ("user_programme_pairs", "users_with_interactions", "unique_interacted_programmes")} for name, values in summaries.items()}, "cold_start": cold_start, "runtime": result["runtime"]}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
