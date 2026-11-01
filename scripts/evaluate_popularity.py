"""Воспроизводимый offline MAP@5 и три popularity baseline, без новых зависимостей."""

import csv
import hashlib
import json
import platform
import resource
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.prepare_interactions import wall_seconds
from src.metrics import mean_average_precision_at_k
from src.popularity import VARIANTS, fit_popularity, top_popular

K = 5
RESULT = ROOT / "results/popularity_baseline.json"
ACTIVITY_BANDS = ("cold_0", "light_1_50", "medium_51_200", "heavy_201_plus")


def file_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def activity_band(count):
    return ACTIVITY_BANDS[0 if count == 0 else 1 if count <= 50 else 2 if count <= 200 else 3]


def validation_candidates(paths, start, end):
    lower, upper = wall_seconds(start), wall_seconds(end)
    candidates = set()
    for path in paths:
        with path.open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                item = int(row["tv_show_id"])
                if item <= 0 or int(row["duration"]) <= 0:
                    continue
                airing_start = wall_seconds(row["start_time"], schedule=True)
                if airing_start < upper and airing_start + int(row["duration"]) > lower:
                    candidates.add(item)
    return candidates


def validation_users(path, start, end):
    # ISO calendar labels are sortable. The validation window is after the
    # verified spring clock jump, so ordering gives the same half-open overlap.
    users = set()
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["start_time"] < end and row["stop_time"] > start and row["start_time"] < row["stop_time"]:
                users.add(row["user_id"])
    return users


def load_ground_truth(path, start, end):
    truth = defaultdict(set)
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            item = int(row["tv_show_id"])
            if row["period"] != "validation" or not start < row["last_watch_time"] <= end or item <= 0:
                raise ValueError("Некорректная строка validation interactions")
            if item in truth[row["user_id"]]:
                raise ValueError("Повторяющаяся пара validation user/item")
            truth[row["user_id"]].add(item)
    return dict(truth)


def cohort_diagnostics(users, truth, activity, candidates, train_items):
    relevant_items, pairs, in_candidates, unseen_pairs = set(), 0, 0, 0
    for user in users:
        relevant = truth.get(user, set())
        relevant_items.update(relevant)
        pairs += len(relevant)
        in_candidates += len(relevant & candidates)
        unseen_pairs += len(relevant - train_items)
    return {
        "users": len(users), "users_without_validation_truth": sum(not truth.get(user) for user in users),
        "cold_start_users": sum(activity.get(user, 0) == 0 for user in users),
        "relevant_user_programme_pairs": pairs, "unique_relevant_programmes": len(relevant_items),
        "relevant_pairs_in_candidates_fraction": in_candidates / pairs if pairs else 0.0,
        "unique_relevant_programmes_in_candidates_fraction": len(relevant_items & candidates) / len(relevant_items) if relevant_items else 0.0,
        "relevant_pairs_absent_from_train_fraction": unseen_pairs / pairs if pairs else 0.0,
        "unique_relevant_programmes_absent_from_train": len(relevant_items - train_items),
        "unique_relevant_programmes_absent_from_train_fraction": len(relevant_items - train_items) / len(relevant_items) if relevant_items else 0.0,
    }


def evaluate_cohort(users, truth, predictions, activity):
    bands = {band: [user for user in users if activity_band(activity.get(user, 0)) == band] for band in ACTIVITY_BANDS}
    return {
        "map_at_5": mean_average_precision_at_k(truth, predictions, users, K),
        "by_train_activity": {
            band: {"users": len(group), "users_without_validation_truth": sum(not truth.get(user) for user in group),
                   "map_at_5": mean_average_precision_at_k(truth, predictions, group, K)}
            for band, group in bands.items()
        },
    }


def main():
    started, cpu_started = time.perf_counter(), time.process_time()
    preparation_path = ROOT / "results/data_preparation.json"
    protected = {path: file_hash(ROOT / path) for path in ("scripts/prepare_interactions.py", "results/data_preparation.json")}
    preparation = json.loads(preparation_path.read_text(encoding="utf-8"))
    bounds = preparation["parameters"]["period_boundaries"]["validation"]
    start, end = bounds["start_inclusive"], bounds["end_exclusive"]
    schedule_names = ("export_arh_11-20-final.csv", "export_arh_21-30-final.csv")
    inputs = {"data/" + name: preparation["inputs"][name]["sha256"]
              for name in (*schedule_names, "dataset11-30.csv", "submission.csv")}
    for period in ("train", "validation"):
        inputs[f"data/processed/{period}_interactions.csv"] = preparation["checks"]["output_validation"][period]["sha256"]
    for path, expected in inputs.items():
        if file_hash(ROOT / path) != expected:
            raise ValueError(f"Вход изменился после подготовки: {path}")
    tests = subprocess.run([sys.executable, "-B", "-m", "unittest", "discover", "-s", str(ROOT / "tests"), "-v"],
                           cwd=ROOT, capture_output=True, text=True)
    if tests.returncode:
        raise RuntimeError(tests.stdout + tests.stderr)

    fit_started = time.perf_counter()
    train_path = ROOT / "data/processed/train_interactions.csv"
    scores, activity = fit_popularity(train_path)
    fit_seconds = time.perf_counter() - fit_started
    candidates = validation_candidates([ROOT / "data" / name for name in schedule_names], start, end)
    if len(candidates) != preparation["candidate_programmes_by_calendar_period"]["validation"]:
        raise ValueError("Validation candidate pool не совпал с подготовкой")
    truth = load_ground_truth(ROOT / "data/processed/validation_interactions.csv", start, end)
    active_validation = validation_users(ROOT / "data/dataset11-30.csv", start, end)
    if len(active_validation) != preparation["interactions"]["validation"]["source_users"] or not truth.keys() <= active_validation:
        raise ValueError("Некорректная популяция validation")
    with (ROOT / "data/submission.csv").open(encoding="utf-8-sig", newline="") as stream:
        primary = [row["user_id"] for row in csv.DictReader(stream)]
    if len(primary) != 1260 or len(set(primary)) != 1260 or not all(isinstance(user, str) and user for user in primary):
        raise ValueError("Ожидалось 1260 различных строковых user_id")
    cohorts = {
        "submission_all": primary,
        "submission_nonempty_validation": [user for user in primary if truth.get(user)],
        "validation_active_all": sorted(active_validation),
        "validation_nonempty": sorted(truth),
    }
    train_items = set(scores["unique_users"])
    diagnostics = {name: cohort_diagnostics(users, truth, activity, candidates, train_items) for name, users in cohorts.items()}
    all_users = set(primary) | active_validation
    variants = {}
    for name in VARIANTS:
        top5 = top_popular(scores[name], candidates, K)
        if len(top5) != K or len(set(top5)) != K or not set(top5) <= candidates:
            raise ValueError("Неполный или недопустимый Top-5")
        predictions = {user: top5 for user in all_users}
        variants[name] = {
            "score_definition": {"unique_users": "distinct train users per programme", "watch_count": "sum of train watch_count", "watch_seconds": "sum of train total_watch_seconds; no capping"}[name],
            "top5": top5, "top5_train_scores": [scores[name][item] for item in top5],
            "cohorts": {cohort: evaluate_cohort(users, truth, predictions, activity) for cohort, users in cohorts.items()},
            "recommendation_coverage": {"unique_programmes": len(set(top5)), "candidate_catalog_fraction": len(set(top5)) / len(candidates)},
        }
    selected = max(VARIANTS, key=lambda name: variants[name]["cohorts"]["submission_all"]["map_at_5"])

    # Fit once more from the same train file; candidate iteration order must not
    # affect ties. Re-evaluate the primary metric independently of cached values.
    repeated_scores, repeated_activity = fit_popularity(train_path)
    if repeated_scores != scores or repeated_activity != activity:
        raise AssertionError("Повторное обучение popularity дало другой результат")
    for name in VARIANTS:
        repeated_top = top_popular(repeated_scores[name], reversed(sorted(candidates)), K)
        repeated_map = mean_average_precision_at_k(truth, {user: repeated_top for user in primary}, primary, K)
        if repeated_top != variants[name]["top5"] or repeated_map != variants[name]["cohorts"]["submission_all"]["map_at_5"]:
            raise AssertionError("Рекомендации или метрика невоспроизводимы")
    for path, expected in {**inputs, **protected}.items():
        if file_hash(ROOT / path) != expected:
            raise AssertionError(f"Во время эксперимента изменился файл: {path}")
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result = {
        "protocol": {
            "id": "sweet-tv-offline-v1", "metric": "MAP@5", "k": K,
            "ap_denominator": "min(5, number of unique full validation-relevant programmes)",
            "empty_ground_truth_ap": 0.0, "duplicate_prediction_policy": "occupies its original rank; only first occurrence can hit",
            "empty_evaluation_population_map": 0.0, "primary_cohort": "submission_all",
            "primary_user_ids": primary, "validation_active_user_ids": sorted(active_validation),
            "candidate_ids": sorted(candidates), "candidate_count": len(candidates),
            "candidate_definition": "positive-duration nonzero programme airings overlapping [validation_start, validation_end); union of first two schedule files",
            "validation_start_inclusive": start, "validation_end_exclusive": end,
            "ground_truth": "all unique validation interaction IDs; never filtered to candidates or train items",
            "seen_programmes_allowed": True, "required_output": "exactly five distinct positive candidate IDs per user",
            "score_tie_break": "numeric tv_show_id ascending", "unseen_train_item_popularity": 0,
            "cold_start_definition": "no train interactions", "fallback_variant": selected,
            "activity_measure": "number of unique train programmes", "activity_bands": list(ACTIVITY_BANDS),
            "selection": "highest submission_all MAP@5; variant-order tie break", "variant_order": list(VARIANTS),
        },
        "inputs_sha256": inputs, "protected_sha256": protected,
        "population_diagnostics": diagnostics,
        "train": {"users": len(activity), "programmes": len(train_items), "user_programme_pairs": sum(activity.values())},
        "candidate_diagnostics": {"with_train_interactions": len(candidates & train_items), "without_train_interactions": len(candidates - train_items)},
        "variants": variants, "selected_baseline": selected,
        "selected_fallback_top5": variants[selected]["top5"],
        "checks": {"all_recommendations_valid_and_complete": True, "no_seen_filter": True, "train_only_popularity": True,
                   "ground_truth_not_filtered": True, "repeated_fit_rank_and_primary_map_identical": True,
                   "inputs_and_preparation_unchanged": True, "test_viewing_and_test_schedule_not_read": True,
                   "unit_tests_passed": True, "unit_test_output": tests.stderr.strip()},
        "runtime": {"wall_seconds": round(time.perf_counter() - started, 3), "cpu_seconds": round(time.process_time() - cpu_started, 3),
                    "fit_all_three_seconds": round(fit_seconds, 3), "peak_rss_mib": round(peak / (1024**2 if sys.platform == "darwin" else 1024), 2),
                    "python": platform.python_version(), "additional_dependencies": []},
        "limitations": ["A single hold-out selected the variant; its score is not an independent test estimate.",
                        "Relevance is the prepared proxy for watching and may differ from Kaggle private labels.",
                        "Popularity returns the same five programmes to every user; watch_seconds can favour long programmes and heavy viewers."],
    }
    temporary = RESULT.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(RESULT)
    print(json.dumps({"primary_map_at_5": {name: variants[name]["cohorts"]["submission_all"]["map_at_5"] for name in VARIANTS},
                      "selected": selected, "runtime": result["runtime"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
