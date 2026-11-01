"""Пересобрать единое окно истории и подготовить submission; ничего не отправлять."""

import csv
import heapq
import itertools
import json
import platform
import resource
import subprocess
import sys
import tempfile
import time
from array import array
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_interactions import (
    BUCKETS, RECORD, SCHEDULE_FILES, build_channel_index, file_sha256, iso,
    match_user, read_views, wall_seconds,
)
from src.popularity import fit_popularity, top_popular

FINAL_START, FINAL_END = "2020-03-09 00:00:00", "2020-07-27 00:00:00"
TEST_START, TEST_END = FINAL_END, "2020-10-19 00:00:00"
FINAL_PERIODS = [("final_train", wall_seconds(FINAL_START), wall_seconds(FINAL_END))]
FINAL_INTERACTIONS = ROOT / "data/processed/final_interactions.csv"


def load_schedules(history_paths, test_path, test_start_iso, test_end_iso):
    """History index — прежние два EPG; test pool включает граничные эфиры."""
    history_paths = [Path(path) for path in history_paths]
    lower, upper = wall_seconds(test_start_iso), wall_seconds(test_end_iso)
    records = defaultdict(lambda: (array("q"), array("q"), array("q")))
    candidates, titles, per_source = set(), defaultdict(Counter), {}
    for number, path in enumerate([*history_paths, Path(test_path)]):
        source_candidates = set()
        with path.open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                start, duration, item = wall_seconds(row["start_time"], True), int(row["duration"]), int(row["tv_show_id"])
                end = start + duration
                if number < len(history_paths):
                    channel = int(row["channel_id"])
                    for values, value in zip(records[channel], (start, end, item)):
                        values.append(value)
                if item > 0 and duration > 0 and start < upper and end > lower:
                    candidates.add(item)
                    source_candidates.add(item)
                    if row["tv_show_title"].strip():
                        titles[item][row["tv_show_title"].strip()] += 1
        per_source[path.name] = len(source_candidates)
    indices, qa = {}, Counter()
    for channel in list(records):
        values = records.pop(channel)
        indices[channel], audit = build_channel_index(zip(*values))
        qa.update(audit)
    names = {item: min(counts, key=lambda title: (-counts[title], title)) for item, counts in titles.items()}
    return indices, candidates, names, {"history_index": dict(qa), "test_candidate_count_by_source": per_source,
        "title_policy": "most frequent nonempty title among test-overlapping airings; lexical tie break; display only"}


def create_submission(template_path, output_path, top5, candidates, expected_rows=1260):
    template_path, output_path = Path(template_path), Path(output_path)
    if template_path.resolve() == output_path.resolve():
        raise ValueError("Нельзя перезаписывать исходный шаблон")
    if len(top5) != 5 or len(set(top5)) != 5 or any(not isinstance(item, int) or item <= 0 for item in top5) or not set(top5) <= set(candidates):
        raise ValueError("Top-5 должен содержать пять различных положительных кандидатов")
    with template_path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.reader(stream))
    if not rows or rows[0] != ["user_id", "tv_show_id"]:
        raise ValueError("Неожиданные столбцы шаблона")
    users = [row[0] for row in rows[1:] if len(row) == 2]
    if len(rows) - 1 != expected_rows or len(users) != expected_rows or len(set(users)) != expected_rows or any(not user for user in users):
        raise ValueError("Некорректное число строк или пользователей шаблона")
    prediction = " ".join(map(str, top5))
    temporary = output_path.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(rows[0])
        writer.writerows((user, prediction) for user in users)
    with temporary.open(encoding="utf-8", newline="") as stream:
        written = list(csv.reader(stream))
    if written != [rows[0], *[[user, prediction] for user in users]]:
        raise AssertionError("Submission не соответствует шаблону")
    temporary.replace(output_path)
    return {"rows": len(users), "columns": rows[0], "users_and_order_match_template": True,
        "five_distinct_positive_candidate_ids_per_row": True, "no_empty_values_or_extra_columns": True,
        "user_ids_preserved_as_strings": True, "sha256": file_sha256(output_path), "bytes": output_path.stat().st_size}


def compare_split_union(final_path, split_paths, users):
    """Только диагностика сохранённых результатов; final fit от них не зависит."""
    positions = {user: position for position, user in enumerate(users)}
    def keys(path):
        previous = None
        with Path(path).open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                position = positions[row["user_id"]]
                key = position % BUCKETS, position, int(row["tv_show_id"])
                if previous is not None and key <= previous:
                    raise AssertionError("CSV не упорядочен по группам исходных пользователей")
                previous = key
                yield key
    old = (key for key, _ in itertools.groupby(heapq.merge(*(keys(path) for path in split_paths))))
    final = keys(final_path)
    left, right = next(old, None), next(final, None)
    common, old_only, new_only, examples = 0, 0, 0, []
    while left is not None or right is not None:
        if right is None or left is not None and left < right:
            old_only += 1
            left = next(old, None)
        elif left is None or right < left:
            new_only += 1
            if len(examples) < 5:
                examples.append({"user_id": users[right[1]], "tv_show_id": right[2]})
            right = next(final, None)
        else:
            common += 1
            left, right = next(old, None), next(final, None)
    if old_only:
        raise AssertionError("При объединении окна потеряны старые user/item пары")
    return {"split_union_user_programme_pairs": common + old_only, "final_user_programme_pairs": common + new_only,
            "additional_pairs_after_unsplit_threshold": new_only, "lost_pairs": old_only, "additional_pair_examples": examples}


def main():
    started, cpu_started = time.perf_counter(), time.process_time()
    preparation = json.loads((ROOT / "results/data_preparation.json").read_text())
    baseline = json.loads((ROOT / "results/popularity_baseline.json").read_text())
    comparison = json.loads((ROOT / "results/model_comparison.json").read_text())
    if comparison["selected_strategy"] != "popularity" or baseline["selected_baseline"] != "unique_users":
        raise ValueError("Выбранная стратегия не совпадает с завершёнными экспериментами")
    protected_paths = [*preparation["inputs"], "processed/train_interactions.csv", "processed/validation_interactions.csv"]
    protected = {"data/" + path: file_sha256(ROOT / "data" / path) for path in protected_paths}
    protected.update({str(path.relative_to(ROOT)): file_sha256(path) for path in (ROOT / "results").iterdir() if path.name != "final_submission.json"})
    protected["scripts/prepare_interactions.py"] = file_sha256(ROOT / "scripts/prepare_interactions.py")
    for name, metadata in preparation["inputs"].items():
        if protected["data/" + name] != metadata["sha256"]:
            raise ValueError("Исходный CSV изменился: " + name)
    for period in ("train", "validation"):
        if protected[f"data/processed/{period}_interactions.csv"] != preparation["checks"]["output_validation"][period]["sha256"]:
            raise ValueError("Изменилось подготовленное разделение")
    tests = subprocess.run([sys.executable, "-B", "-m", "unittest", "discover", "-s", str(ROOT / "tests"), "-v"], cwd=ROOT, capture_output=True, text=True)
    if tests.returncode:
        raise RuntimeError(tests.stdout + tests.stderr)
    print("Построение прежнего history index и календарного test pool", flush=True)
    indices, candidates, titles, schedule_audit = load_schedules(
        [ROOT / "data" / name for name in SCHEDULE_FILES[:2]], ROOT / "data" / SCHEDULE_FILES[2], TEST_START, TEST_END)
    if len(candidates) != preparation["candidate_programmes_by_calendar_period"]["test"]:
        raise AssertionError("Test pool отличается от календарного EDA")
    summaries, active_users, source_users, item_counts = Counter(), set(), set(), Counter()
    temporary_output = FINAL_INTERACTIONS.with_suffix(".csv.tmp")
    with tempfile.TemporaryDirectory(prefix=".final-", dir=FINAL_INTERACTIONS.parent) as directory:
        work = Path(directory)
        view_stats, users, *_ = read_views(work)  # Packed rows retain original, unsplit start/stop.
        if view_stats.get("duration_mismatches_after_clock_normalisation", 0):
            raise AssertionError("Нормализация времени отличается от проверенной подготовки")
        with temporary_output.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["user_id", "tv_show_id", "watch_count", "total_watch_seconds", "last_watch_time", "period"])
            for bucket in range(BUCKETS):
                rows = sorted(RECORD.iter_unpack((work / f"views_{bucket:02}.bin").read_bytes()))
                summaries["source_view_rows_entering_final_window"] += len(rows)
                for number, group in itertools.groupby(rows, key=lambda row: row[0]):
                    source_users.add(number)
                    channel_views = defaultdict(list)
                    for _, channel, start, end, _ in group:
                        channel_views[channel].append((start, end))
                    interactions, diagnostics = match_user(channel_views, indices, periods=FINAL_PERIODS, threshold=60)
                    summaries.update(diagnostics["final_train"])
                    if interactions["final_train"]:
                        active_users.add(number)
                    for item, (count, seconds, last) in sorted(interactions["final_train"].items()):
                        if item <= 0 or count <= 0 or seconds < 60 or not FINAL_PERIODS[0][1] < last <= FINAL_PERIODS[0][2]:
                            raise AssertionError("Некорректное финальное взаимодействие")
                        writer.writerow([users[number], item, count, seconds, iso(last), "final_train"])
                        summaries["user_programme_pairs"] += 1
                        summaries["watch_count"] += count
                        summaries["total_watch_seconds"] += seconds
                        item_counts[item] += 1
                if (bucket + 1) % 8 == 0:
                    print(f"Final user blocks: {bucket + 1}/{BUCKETS}", flush=True)
    temporary_output.replace(FINAL_INTERACTIONS)
    scores, activity = fit_popularity(FINAL_INTERACTIONS, expected_period="final_train")
    if scores["unique_users"] != item_counts or len(activity) != len(active_users) or sum(activity.values()) != summaries["user_programme_pairs"]:
        raise AssertionError("Финальная popularity не согласуется с user/item взаимодействиями")
    top5 = top_popular(scores["unique_users"], candidates)
    union_check = compare_split_union(FINAL_INTERACTIONS,
        [ROOT / "data/processed/train_interactions.csv", ROOT / "data/processed/validation_interactions.csv"], users)
    submission = create_submission(ROOT / "data/submission.csv", ROOT / "submission.csv", top5, candidates)
    for path, expected in protected.items():
        if file_sha256(ROOT / path) != expected:
            raise AssertionError("Изменился защищённый файл: " + path)
    seconds_partition = sum(summaries[key] for key in ("labelled_overlap_seconds", "unlabelled_overlap_seconds", "ambiguous_overlap_seconds", "absent_channel_seconds", "outside_schedule_seconds"))
    if seconds_partition != summaries["unique_user_channel_view_seconds"]:
        raise AssertionError("Нарушен баланс уникальных секунд")
    old_top5 = baseline["variants"]["unique_users"]["top5"]
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result = {
        "final_train": {"start_inclusive": FINAL_START, "end_exclusive": FINAL_END, **dict(summaries),
                        "source_users": len(source_users), "users_with_interactions": len(active_users), "unique_programmes": len(item_counts),
                        "users_without_interactions": len(source_users - active_users), "sha256": file_sha256(FINAL_INTERACTIONS),
                        "path": "data/processed/final_interactions.csv", "bytes": FINAL_INTERACTIONS.stat().st_size},
        "test": {"start_inclusive": TEST_START, "end_exclusive": TEST_END, "candidate_count": len(candidates)},
        "parameters": {"algorithm": "global popularity by distinct final-train users", "threshold_seconds": 60,
                       "period_label": "final_train", "view_rule": preparation["parameters"]["threshold_applies_to"].replace("separately in each period", "on the full unsplit final-train window"),
                       "interval_rule": "same verified channel and half-open overlap; previous match_user/build_channel_index reused",
                       "time_normalisation": preparation["parameters"]["clock_gap_normalisation"],
                       "same_zero_id_and_conflict_and_duplicate_policies": True, "test_views_used": False,
                       "ranking": "unique users descending, numeric tv_show_id ascending", "personalisation": False},
        "top5": [{"tv_show_id": item, "title": titles.get(item), "final_unique_users": scores["unique_users"][item]} for item in top5],
        "validation_top5": old_top5, "top5_comparison": {"identical_order": top5 == old_top5,
            "added": sorted(set(top5) - set(old_top5)), "removed": sorted(set(old_top5) - set(top5))},
        "combined_window_verification": union_check, "schedule_audit": schedule_audit,
        "submission": {"path": "submission.csv", **submission},
        "checks": {"unit_tests_passed": True, "unit_test_output": tests.stderr.strip(), "old_results_and_inputs_unchanged": True,
                   "popularity_reconciled": True, "seconds_partition_reconciled": True, "no_split_pairs_lost": True,
                   "full_window_rebuilt_from_raw_views": True},
        "protected_sha256": protected,
        "runtime": {"wall_seconds": time.perf_counter() - started, "cpu_seconds": time.process_time() - cpu_started,
                    "peak_rss_mib": peak / (1024**2 if sys.platform == "darwin" else 1024), "python": platform.python_version()},
        "kaggle_submitted": False, "kaggle_private_score": None,
    }
    result_path = ROOT / "results/final_submission.json"
    temporary = result_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(result_path)
    print(json.dumps({"final_train": {key: result["final_train"][key] for key in ("user_programme_pairs", "users_with_interactions", "unique_programmes", "watch_count")},
                      "top5": result["top5"], "submission": submission, "runtime": result["runtime"]}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
