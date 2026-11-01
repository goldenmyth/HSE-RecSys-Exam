"""Независимые ALS/CBF и два rank-fusion на замороженном offline-протоколе."""

import os
for _variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_variable] = "1"

import csv
import hashlib
import json
import math
import platform
import resource
import subprocess
import sys
import time
from array import array
from collections import Counter
from fractions import Fraction
from importlib.metadata import version
from pathlib import Path

import numpy as np
from scipy import sparse
from threadpoolctl import threadpool_info, threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluate_popularity import evaluate_cohort, file_hash, load_ground_truth
from src.content_based import fit_content_model, load_train_metadata, score_content
from src.matrix_factorization import fit_als, score_als
from src.metrics import average_precision_at_k
from src.popularity import top_popular
from src.ranking import rank_with_fallback, reciprocal_rank_fusion

SEED = 42
ALS_CONFIGS = [
    {"factors": f, "regularization": r, "weighting": w, "confidence_alpha": a, "iterations": 20}
    for f, r, w, a in ((32, .1, "binary", 10), (32, .1, "log_count", 10),
                       (32, .1, "log_count", 40), (64, .1, "log_count", 10),
                       (64, .1, "log_count", 40), (64, 1., "log_count", 40))
]
CONTENT_CONFIGS = [
    {"feature_set": "basic", "history_weighting": "binary"},
    {"feature_set": "extended", "history_weighting": "binary"},
    {"feature_set": "extended", "history_weighting": "log1p"},
]


def load_train(path):
    users, items, activity = {}, {}, Counter()
    row_indices, column_indices, values = array("i"), array("i"), array("f")
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            user, item, count = row["user_id"], int(row["tv_show_id"]), int(row["watch_count"])
            if row["period"] != "train" or item <= 0 or count <= 0:
                raise ValueError("Ожидались положительные train interactions")
            if not "2020-03-09 00:00:00" < row["last_watch_time"] <= "2020-05-18 00:00:00":
                raise ValueError("Train interaction вышло за границы train")
            row_indices.append(users.setdefault(user, len(users)))
            column_indices.append(items.setdefault(item, len(items)))
            values.append(count)
            activity[user] += 1
    counts = sparse.csr_matrix((values, (row_indices, column_indices)), shape=(len(users), len(items)), dtype=np.float32)
    if counts.nnz != len(values):
        raise ValueError("В train найдены повторяющиеся пары")
    return counts, users, np.asarray(list(items), dtype=np.int64), activity


def audit_baseline(counts, user_index, item_ids, users, candidates, truth, baseline):
    fit_started = time.perf_counter()
    frequencies = np.asarray(counts.getnnz(axis=0)).ravel()
    popularity = {int(item): int(score) for item, score in zip(item_ids, frequencies)}
    fit_seconds = time.perf_counter() - fit_started
    rank_started = time.perf_counter()
    top5 = top_popular(popularity, candidates)
    rank_seconds = time.perf_counter() - rank_started
    saved = baseline["variants"]["unique_users"]
    if top5 != saved["top5"] or [popularity[item] for item in top5] != saved["top5_train_scores"]:
        raise AssertionError("Критическая ошибка: train baseline не воспроизведён")
    exact, hit_counts, ap_counts, examples = [], Counter(), Counter(), {}
    for user in users:
        relevant = truth.get(user, set())
        hits = [int(item in relevant) for item in top5]
        running, numerator = 0, Fraction(0)
        for rank, hit in enumerate(hits, 1):
            running += hit
            if hit:
                numerator += Fraction(running, rank)
        denominator = min(5, len(relevant))
        ap = numerator / denominator if denominator else Fraction(0)
        if not math.isclose(float(ap), average_precision_at_k(relevant, top5), abs_tol=1e-15):
            raise AssertionError("Критическая ошибка AP@5")
        exact.append(ap)
        hit_counts[sum(hits)] += 1
        kind = "zero_empty" if not relevant else "zero_nonempty" if ap == 0 else "one" if ap == 1 else "partial"
        ap_counts[kind] += 1
        if kind not in examples:
            examples[kind] = {"user_id": user, "gt_size": len(relevant), "hits": hits,
                              "precision_sum": str(numerator), "denominator": denominator,
                              "ap_fraction": str(ap), "ap": float(ap)}
    exact_mean = sum(exact, Fraction(0)) / len(users)
    expected = saved["cohorts"]["submission_all"]["map_at_5"]
    if len(users) != 1260 or len(set(users)) != 1260 or not math.isclose(float(exact_mean), expected, abs_tol=1e-15):
        raise AssertionError("Критическая ошибка основной популяции или MAP")
    return popularity, top5, {
        "passed": True, "users": len(users), "map_exact_fraction": str(exact_mean), "map_at_5": float(exact_mean),
        "top5": top5, "top5_train_unique_users": [popularity[item] for item in top5],
        "validation_hit_users": [sum(item in truth.get(user, set()) for user in users) for item in top5],
        "top5_hit_count_distribution": dict(hit_counts), "ap_user_counts": dict(ap_counts),
        "manual_examples": examples, "user_ids_are_strings": all(isinstance(user, str) for user in users),
        "user_id_length_distribution": dict(Counter(map(len, users))),
        "ground_truth_source": "validation_interactions.csv; SHA-256 and period/date checks passed; never replaced by train",
        "popularity_fit_seconds": fit_seconds, "popularity_rank_seconds": rank_seconds,
    }


def record_result(name, family, config, predictions, fallback_used, users, truth, activity, baseline_ap, timing, audit):
    by_user = dict(zip(users, predictions))
    aps = {user: average_precision_at_k(truth.get(user, ()), by_user[user]) for user in users}
    recommended = {item for row in predictions for item in row}
    return {
        "family": family, "config": config, "timing": timing, "audit": audit,
        "cohorts": {"submission_all": evaluate_cohort(users, truth, by_user, activity),
                    "submission_nonempty_validation": evaluate_cohort([u for u in users if truth.get(u)], truth, by_user, activity)},
        "unique_recommended_programmes": len(recommended),
        "recommendation_candidate_fraction": len(recommended) / 6384,
        "fallback_users": int(sum(fallback_used)), "fallback_user_fraction": float(sum(fallback_used) / len(users)),
        "versus_popularity": {"users_with_higher_ap": sum(aps[u] > baseline_ap[u] + 1e-12 for u in users),
                              "users_with_lower_ap": sum(aps[u] < baseline_ap[u] - 1e-12 for u in users),
                              "users_with_equal_ap": sum(abs(aps[u] - baseline_ap[u]) <= 1e-12 for u in users)},
    }


def main():
    started, cpu_started = time.perf_counter(), time.process_time()
    baseline = json.loads((ROOT / "results/popularity_baseline.json").read_text())
    preparation = json.loads((ROOT / "results/data_preparation.json").read_text())
    protocol = baseline["protocol"]
    users, candidates = protocol["primary_user_ids"], np.asarray(protocol["candidate_ids"], dtype=np.int64)
    protected_paths = ("scripts/prepare_interactions.py", "results/data_preparation.json", "results/popularity_baseline.json", "src/metrics.py", "src/popularity.py")
    protected = {path: file_hash(ROOT / path) for path in protected_paths}
    for path, expected in baseline["inputs_sha256"].items():
        if file_hash(ROOT / path) != expected:
            raise ValueError("Вход не совпадает с замороженным baseline: " + path)
    tests = subprocess.run([sys.executable, "-B", "-m", "unittest", "discover", "-s", str(ROOT / "tests"), "-v"],
                           cwd=ROOT, capture_output=True, text=True)
    if tests.returncode:
        raise RuntimeError(tests.stdout + tests.stderr)
    load_started = time.perf_counter()
    counts, user_index, item_ids, activity = load_train(ROOT / "data/processed/train_interactions.csv")
    truth = load_ground_truth(ROOT / "data/processed/validation_interactions.csv", protocol["validation_start_inclusive"], protocol["validation_end_exclusive"])
    data_load_seconds = time.perf_counter() - load_started
    fit_start = time.perf_counter()
    popularity, fallback, baseline_audit = audit_baseline(counts, user_index, item_ids, users, candidates, truth, baseline)
    if not set(fallback) <= set(candidates):
        raise AssertionError("Baseline содержит недопустимые кандидаты")
    audit_seconds = time.perf_counter() - fit_start
    baseline_ap = {user: average_precision_at_k(truth.get(user, ()), fallback) for user in users}
    predict_started = time.perf_counter()
    predictions = {"popularity": [fallback[:] for _ in users]}
    popularity_predict_seconds = time.perf_counter() - predict_started + baseline_audit["popularity_rank_seconds"]
    results = {"popularity": record_result("popularity", "popularity", {"variant": "unique_users"}, predictions["popularity"],
        [False] * len(users), users, truth, activity, baseline_ap, {"independent_audit_seconds": audit_seconds,
        "train_seconds": baseline_audit["popularity_fit_seconds"], "recommend_seconds": popularity_predict_seconds}, {})}
    user_indices = [user_index.get(user, -1) for user in users]
    item_index = {int(item): position for position, item in enumerate(item_ids)}
    candidate_indices = [item_index.get(int(item), -1) for item in candidates]
    metadata, metadata_audit = load_train_metadata([ROOT / "data/export_arh_11-20-final.csv"],
        preparation["parameters"]["period_boundaries"]["train"]["start_inclusive"],
        preparation["parameters"]["period_boundaries"]["train"]["end_exclusive"])
    best = {}
    best_scores = {}
    for family, configurations in (("als", ALS_CONFIGS), ("content", CONTENT_CONFIGS)):
        for number, config in enumerate(configurations, 1):
            name = f"{family}_{number}"
            fit_started = time.perf_counter()
            model = fit_als(counts, config, SEED) if family == "als" else fit_content_model(metadata, counts, item_ids, **config)
            fit_seconds = time.perf_counter() - fit_started
            predict_started = time.perf_counter()
            scores = score_als(model, user_indices, candidate_indices) if family == "als" else score_content(model, user_indices, candidates)
            predicted, used_fallback = rank_with_fallback(scores, candidates, fallback)
            predict_seconds = time.perf_counter() - predict_started
            model_audit = {"known_candidate_count": sum(index >= 0 for index in candidate_indices), "confidence_formula": "c=1+confidence_alpha*f(watch_count); library alpha=1", "seed": SEED, "threads": 1} if family == "als" else model["audit"]
            results[name] = record_result(name, family, config, predicted, used_fallback, users, truth, activity, baseline_ap,
                {"train_seconds": fit_seconds, "recommend_seconds": predict_seconds}, model_audit)
            predictions[name] = predicted
            value = results[name]["cohorts"]["submission_all"]["map_at_5"]
            if family not in best or value > results[best[family]]["cohorts"]["submission_all"]["map_at_5"]:
                best[family], best_scores[family] = name, scores
            del model
            print(f"{name}: MAP@5={value:.9f}; fit={fit_seconds:.3f}s; predict={predict_seconds:.3f}s", flush=True)
    # Two fixed, dimensionless rank blends after independent model evaluation.
    personal_family = max(best, key=lambda family: results[best[family]]["cohorts"]["submission_all"]["map_at_5"])
    source_name, personal = best[personal_family], best_scores[personal_family]
    source_config = results[source_name]["config"]
    personal_model = fit_als(counts, source_config, SEED) if personal_family == "als" else fit_content_model(metadata, counts, item_ids, **source_config)
    for weight in (0.2, 0.5):
        name = f"rrf_{weight:g}"
        predict_started = time.perf_counter()
        # Measure inference as well as fusion; no cached-score timing shortcut.
        fresh_personal = score_als(personal_model, user_indices, candidate_indices) if personal_family == "als" else score_content(personal_model, user_indices, candidates)
        scores = reciprocal_rank_fusion(fresh_personal, candidates, popularity, weight, offset=10)
        predicted, _ = rank_with_fallback(scores, candidates, fallback)
        recommend_seconds = time.perf_counter() - predict_started
        used_fallback = ~np.isfinite(fresh_personal).any(axis=1)
        results[name] = record_result(name, "rank_fusion", {"personal_method": source_name, "personal_weight": weight,
            "popularity_weight": 1 - weight, "rank_offset": 10}, predicted, used_fallback, users, truth, activity, baseline_ap,
            {"train_seconds": results[source_name]["timing"]["train_seconds"], "additional_train_seconds": 0.0,
             "recommend_seconds": recommend_seconds, "includes_personal_inference": True},
            {"scores_are_reciprocal_ranks": True, "cold_user_has_identical_popularity_order": True})
        predictions[name] = predicted
        print(f"{name}: MAP@5={results[name]['cohorts']['submission_all']['map_at_5']:.9f}", flush=True)
    del personal_model, fresh_personal

    # Refit each selected independent model from scratch in the same environment.
    for family, name in best.items():
        config = results[name]["config"]
        model = fit_als(counts, config, SEED) if family == "als" else fit_content_model(metadata, counts, item_ids, **config)
        scores = score_als(model, user_indices, candidate_indices) if family == "als" else score_content(model, user_indices, candidates)
        repeated, _ = rank_with_fallback(scores, candidates, fallback)
        if repeated != predictions[name]:
            raise AssertionError("Повторный fit изменил Top-5: " + name)
        del model, scores
    pool = set(map(int, candidates))
    seen_by_user = {}
    for user, index in zip(users, user_indices):
        columns = counts.indices[counts.indptr[index]:counts.indptr[index + 1]] if index >= 0 else []
        seen_by_user[user] = set(map(int, item_ids[columns])) if len(columns) else set()
    for name, rows in predictions.items():
        if len(rows) != 1260 or any(len(row) != 5 or len(set(row)) != 5 or not set(row) <= pool for row in rows):
            raise AssertionError("Некорректный Top-5: " + name)
        for pos, index in enumerate(user_indices):
            if index == -1 and rows[pos] != fallback:
                raise AssertionError("Cold-start fallback различается: " + name)
        results[name]["ranking_diagnostics"] = {
            "mean_train_unique_user_popularity_of_recommended_items": math.fsum(popularity.get(item, 0) for row in rows for item in row) / (5 * len(users)),
            "mean_validation_hits_at_5": math.fsum(len(set(row) & truth.get(user, set())) for user, row in zip(users, rows)) / len(users),
            "recommended_pairs_seen_in_train_fraction": sum(item in seen_by_user[user] for user, row in zip(users, rows) for item in row) / (5 * len(users)),
        }
    for path, expected in {**baseline["inputs_sha256"], **protected}.items():
        if file_hash(ROOT / path) != expected:
            raise AssertionError("Изменился защищённый вход: " + path)

    selected = max(results, key=lambda name: results[name]["cohorts"]["submission_all"]["map_at_5"])
    recommendations_path = ROOT / "results/model_recommendations.csv"
    temporary = recommendations_path.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["method", "user_id", "item_1", "item_2", "item_3", "item_4", "item_5"])
        for name, rows in predictions.items():
            for user, row in zip(users, rows):
                writer.writerow([name, user, *row])
    temporary.replace(recommendations_path)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result = {
        "protocol_id": protocol["id"], "protocol_sha256": protected["results/popularity_baseline.json"],
        "protocol": {key: value for key, value in protocol.items() if not key.endswith("_ids")},
        "baseline_audit": baseline_audit, "methods": results, "best_independent_models": best, "selected_strategy": selected,
        "content_metadata_audit": metadata_audit,
        "candidate_coverage": {"candidate_count": len(candidates), "with_train_factors": sum(index >= 0 for index in candidate_indices),
                               "with_train_metadata": sum(int(item) in metadata for item in candidates)},
        "training_matrix": {"shape": list(counts.shape), "nnz": counts.nnz, "density": counts.nnz / np.prod(counts.shape)},
        "inputs_sha256": baseline["inputs_sha256"], "protected_sha256": protected,
        "recommendations": {"path": "results/model_recommendations.csv", "sha256": file_hash(recommendations_path), "rows": len(predictions) * len(users)},
        "checks": {"unit_tests_passed": True, "unit_test_output": tests.stderr.strip(), "train_only_fit_and_metadata": True,
                   "validation_only_used_for_evaluation_and_selection": True, "full_relevance_retained": True,
                   "all_top5_valid_unique_and_complete": True, "same_cold_start_fallback": True,
                   "best_independent_models_refit_same_top5": True, "protected_inputs_unchanged": True},
        "environment": {"seed": SEED, "als_threads": 1, "blas_threads": 1, "python": platform.python_version(),
                        "platform": platform.platform(), "dependencies": {name: version(name) for name in ("numpy", "scipy", "implicit", "threadpoolctl")},
                        "threadpools": threadpool_info()},
        "runtime": {"wall_seconds": time.perf_counter() - started, "cpu_seconds": time.process_time() - cpu_started,
                    "data_load_seconds": data_load_seconds, "peak_rss_mib": peak / (1024**2 if sys.platform == "darwin" else 1024)},
        "limitations": ["All selection uses one hold-out; improvements are not independent test estimates.",
                        "MF cannot score items without train factors; CBF cannot score items without train metadata.",
                        "Future metadata and future viewing are excluded from fitting; no item attributes are invented.",
                        "Fallback fractions count users requiring supplemental popularity, not normal popularity participation in fusion.",
                        "Thread/seed reproducibility is checked in this pinned environment, not promised across BLAS implementations."],
    }
    result_path = ROOT / "results/model_comparison.json"
    temporary = result_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(result_path)
    print("Selected:", selected, results[selected]["cohorts"]["submission_all"]["map_at_5"], flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
