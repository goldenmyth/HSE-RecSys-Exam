"""Новая offline-серия: strict single-session GT, Repeat, PPR и Top-100 RRF."""

import os
for _name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "1"

import csv
import json
import math
import resource
import subprocess
import sys
import time
from array import array
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy import sparse
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_sessions import POLICY, prepare, protected_hashes
from scripts.prepare_interactions import file_sha256, quantiles, wall_seconds
from scripts.evaluate_popularity import evaluate_cohort, load_ground_truth
from src.content_based import fit_content_model, load_train_metadata, score_content
from src.fusion import ranked_lists, weighted_rrf
from src.matrix_factorization import fit_als, score_als
from src.metrics import average_precision_at_k
from src.personal_repeat import repeat_scores
from src.personalized_pagerank import build_graph, personalized_scores
from src.popularity import top_popular
from src.ranking import rank_with_fallback

PPR_CONFIGS = [
    {"normalisation": "user_cosine", "restart": .15, "weighting": "log_count"},
    {"normalisation": "user_cosine", "restart": .30, "weighting": "log_count"},
    {"normalisation": "cosine", "restart": .15, "weighting": "log_count"},
    {"normalisation": "user_cosine", "restart": .15, "weighting": "binary"},
]
REPEAT_CONFIGS = [
    {"threshold_percent": 50, "half_life_days": None},
    {"threshold_percent": 80, "half_life_days": None},
    {"threshold_percent": 50, "half_life_days": 30},
]
FUSIONS = [
    {"pop": .2, "repeat": .8}, {"pop": .5, "repeat": .5}, {"pop": .5, "ppr": .5},
    {"ppr": .5, "als": .25, "content": .25},
    {"pop": .4, "ppr": .3, "als": .15, "content": .15},
    {"pop": .2, "repeat": .5, "ppr": .2, "als": .05, "content": .05},
    {"pop": .4, "repeat": .3, "ppr": .2, "als": .05, "content": .05},
]


def load_history(path, period="train", lower="2020-03-09 00:00:00", upper="2020-05-18 00:00:00"):
    users, items, activity = {}, {}, Counter()
    rows, columns, counts, last = array("i"), array("i"), array("f"), array("d")
    with Path(path).open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            user, item, count = row["user_id"], int(row["tv_show_id"]), int(row["watch_count"])
            if row["period"] != period or not lower < row["last_watch_time"] <= upper or item <= 0 or count <= 0:
                raise ValueError("Некорректная история или выход за training cutoff")
            rows.append(users.setdefault(user, len(users)))
            columns.append(items.setdefault(item, len(items)))
            counts.append(count)
            last.append(wall_seconds(row["last_watch_time"]))
            activity[user] += 1
    shape = len(users), len(items)
    matrix = sparse.csr_matrix((counts, (rows, columns)), shape=shape, dtype=np.float32)
    stamps = sparse.csr_matrix((last, (rows, columns)), shape=shape, dtype=np.float64)
    if matrix.nnz != len(rows):
        raise ValueError("Дубли user/item в истории")
    return {"counts": matrix, "last": stamps, "users": users, "items": np.asarray(list(items), dtype=np.int64),
            "activity": activity, "cutoff": wall_seconds(upper), "path": str(Path(path).relative_to(ROOT))}


def mapped(history, users, candidates):
    item_index = {int(item): position for position, item in enumerate(history["items"])}
    return [history["users"].get(user, -1) for user in users], [item_index.get(int(item), -1) for item in candidates]


def model_scores(family, config, history, users, candidates, metadata=None, graphs=None):
    user_indices, item_indices = mapped(history, users, candidates)
    started = time.perf_counter()
    fit_seconds, audit = 0.0, {}
    if family == "repeat":
        scores = repeat_scores(history["counts"], history["last"], user_indices, item_indices, history["cutoff"], config["half_life_days"])
    elif family == "als":
        model = fit_als(history["counts"], config, seed=42)
        fit_seconds = time.perf_counter() - started
        scores = score_als(model, user_indices, item_indices)
    elif family == "content":
        model = fit_content_model(metadata, history["counts"], history["items"], **config)
        fit_seconds = time.perf_counter() - started
        scores = score_content(model, user_indices, candidates)
        audit = model["audit"]
    elif family == "ppr":
        if graphs is None:
            graphs = {}
        mode = config["normalisation"]
        if mode not in graphs:
            graphs[mode] = build_graph(history["counts"], history["items"], normalisation=mode, neighbors=50)
        transition, graph_audit = graphs[mode]
        fit_seconds = time.perf_counter() - started
        scores, convergence = personalized_scores(transition, history["counts"], user_indices, item_indices,
            restart=config["restart"], weighting=config["weighting"], tol=1e-8, max_iter=200, batch_size=32)
        scores[scores <= 0] = np.nan  # Unreachable zero-mass items do not replace fallback.
        audit = {"graph": graph_audit, "convergence": convergence, "zero_mass_ranking_policy": "unreachable candidates omitted"}
    else:
        raise ValueError("Неизвестное семейство модели")
    return scores, {"train_seconds": fit_seconds, "score_seconds": time.perf_counter() - started - fit_seconds}, audit


def gt_summary(truth, users, candidates):
    sizes = Counter(len(truth.get(user, ())) for user in users)
    pairs = sum(size * count for size, count in sizes.items())
    covered = sum(len(truth.get(user, set()) & set(candidates)) for user in users)
    return {"users": len(users), "nonempty_users": len(users) - sizes[0], "user_programme_pairs": pairs,
            "programmes_per_user": quantiles(sizes), "candidate_coverage_fraction": covered / pairs if pairs else 0.0}


def evaluate(predictions, users, truth, activity, candidates, timing, training_source, fallback_used=None):
    mapping = dict(zip(users, predictions))
    for row in predictions:
        if len(row) != 5 or len(set(row)) != 5 or not set(row) <= set(candidates):
            raise AssertionError("Некорректные рекомендации")
    return {"strict_protocol": {"submission_all": evaluate_cohort(users, truth, mapping, activity),
                               "submission_nonempty_validation": evaluate_cohort([u for u in users if truth.get(u)], truth, mapping, activity)},
            "unique_recommended_programmes": len({item for row in predictions for item in row}),
            "timing": timing, "training_source": training_source,
            "fallback_users": None if fallback_used is None else int(sum(fallback_used)),
            "actual_training_unknown_users": None}


def compare_truth(old, new, users, candidates):
    return {"old_60s": gt_summary(old, users, candidates), "strict_session_80": gt_summary(new, users, candidates),
        "pairs_removed": sum(len(old.get(u, set()) - new.get(u, set())) for u in users),
        "pairs_added": sum(len(new.get(u, set()) - old.get(u, set())) for u in users),
        "overlap_pairs": sum(len(old.get(u, set()) & new.get(u, set())) for u in users)}


def write_predictions(path, predictions, users):
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["method", "user_id", "item_1", "item_2", "item_3", "item_4", "item_5"])
        for name, rows in predictions.items():
            writer.writerows([name, user, *row] for user, row in zip(users, rows))
    temporary.replace(path)


def paired_improvement(best, reference, users, truth):
    deltas = np.asarray([average_precision_at_k(truth.get(user, ()), a) - average_precision_at_k(truth.get(user, ()), b)
                         for user, a, b in zip(users, best, reference)])
    generator = np.random.default_rng(42)
    sampled = np.asarray([deltas[generator.integers(0, len(users), len(users))].mean() for _ in range(1000)])
    lower, upper = np.quantile(sampled, [.025, .975])
    return {"mean_map_delta": float(deltas.mean()), "paired_bootstrap_95_interval": [float(lower), float(upper)],
            "bootstrap_samples": 1000, "seed": 42, "users_improved": int(np.count_nonzero(deltas > 1e-12)),
            "users_worsened": int(np.count_nonzero(deltas < -1e-12)),
            "v2_criterion": "mean improvement >= 0.005 and paired-bootstrap lower bound > 0",
            "convincing_for_v2": bool(deltas.mean() >= .005 and lower > 0),
            "limitation": "descriptive post-selection bootstrap; not corrected for model selection and not an independent test"}


def user_examples(predictions, names, users, truth):
    chosen = ["11595812149006555679", "8150816946177670084"]
    needed = {item for name in names for user in chosen for item in predictions[name][users.index(user)]}
    attributes = defaultdict(Counter)
    lower, upper = wall_seconds("2020-05-18 00:00:00"), wall_seconds("2020-07-27 00:00:00")
    for filename in ("export_arh_11-20-final.csv", "export_arh_21-30-final.csv"):
        with (ROOT / "data" / filename).open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                item = int(row["tv_show_id"])
                if item in needed:
                    start = wall_seconds(row["start_time"], True)
                    if start < upper and start + int(row["duration"]) > lower:
                        attributes[item][(row["tv_show_title"], row["tv_show_category"])] += 1
    display = {item: min(values, key=lambda pair: (-values[pair], pair)) for item, values in attributes.items()}
    return [{"user_id": user, "selection": "train activity near p90" if number == 0 else "minimum positive train history",
        "strict_gt_size": len(truth.get(user, ())), "methods": {name: [
            {"tv_show_id": item, "title": display.get(item, (None, None))[0], "category": display.get(item, (None, None))[1],
             "strict_validation_relevant": item in truth.get(user, set())} for item in predictions[name][users.index(user)]] for name in names}}
        for number, user in enumerate(chosen)]


def write_user_examples(path, examples):
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["user_id", "selection", "method", "rank", "tv_show_id", "title", "category", "strict_validation_relevant"])
        for example in examples:
            for method, recommendations in example["methods"].items():
                for rank, row in enumerate(recommendations, 1):
                    writer.writerow([example["user_id"], example["selection"], method, rank, row["tv_show_id"],
                                     row["title"], row["category"], row["strict_validation_relevant"]])


def replacement_diagnostics(predictions, users, truth, references):
    result = {}
    for reference in references:
        reference_rows = predictions[reference]
        reference_ap = [average_precision_at_k(truth.get(user, ()), row) for user, row in zip(users, reference_rows)]
        comparisons = {}
        for name, rows in predictions.items():
            removed = added = relevant_removed = relevant_added = better = worse = 0
            deltas = []
            for user, old, new, old_ap in zip(users, reference_rows, rows, reference_ap):
                lost, gained = set(old) - set(new), set(new) - set(old)
                relevant = truth.get(user, set())
                removed += len(lost)
                added += len(gained)
                relevant_removed += len(lost & relevant)
                relevant_added += len(gained & relevant)
                delta = average_precision_at_k(relevant, new) - old_ap
                deltas.append(delta)
                better += delta > 1e-12
                worse += delta < -1e-12
            comparisons[name] = {"removed_items": removed, "added_items": added,
                "removed_relevant_items": relevant_removed, "added_relevant_items": relevant_added,
                "users_with_higher_ap": better, "users_with_lower_ap": worse,
                "mean_ap_delta": math.fsum(deltas) / len(users)}
        result[reference] = comparisons
    return result


def write_metrics(path, methods):
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["protocol_id", "method", "family", "map_at_5", "map_at_5_nonempty_gt", "unique_items",
                         "fallback_users", "score_seconds", "graph_build_seconds", "fusion_seconds", "training_source"])
        for name, value in methods.items():
            timing, audit = value["timing"], value.get("audit", {})
            writer.writerow([POLICY["id"], name, value["family"], value["strict_protocol"]["submission_all"]["map_at_5"],
                value["strict_protocol"]["submission_nonempty_validation"]["map_at_5"], value["unique_recommended_programmes"],
                value["fallback_users"], timing.get("score_seconds"), audit.get("graph", {}).get("elapsed_seconds"),
                timing.get("fusion_seconds"), value["training_source"]])


def main():
    started = time.perf_counter()
    protected = protected_hashes()
    session_manifest = prepare()
    baseline = json.loads((ROOT / "results/popularity_baseline.json").read_text())
    previous = json.loads((ROOT / "results/model_comparison.json").read_text())
    protocol = baseline["protocol"]
    users, candidates = protocol["primary_user_ids"], np.asarray(protocol["candidate_ids"], dtype=np.int64)
    lower, upper = protocol["validation_start_inclusive"], protocol["validation_end_exclusive"]
    truth = load_ground_truth(ROOT / session_manifest["outputs"]["validation_80"]["path"], lower, upper)
    old_truth = load_ground_truth(ROOT / "data/processed/validation_interactions.csv", lower, upper)
    row_truth = load_ground_truth(ROOT / session_manifest["outputs"]["validation_80_row_only"]["path"], lower, upper)
    histories = {60: load_history(ROOT / "data/processed/train_interactions.csv"),
                 50: load_history(ROOT / session_manifest["outputs"]["train_50"]["path"]),
                 80: load_history(ROOT / session_manifest["outputs"]["train_80"]["path"])}
    activity = histories[60]["activity"]  # Frozen activity cohorts, independent of stricter label.
    frequencies = histories[60]["counts"].getnnz(axis=0)
    pop_scores = {int(item): int(score) for item, score in zip(histories[60]["items"], frequencies)}
    fallback = top_popular(pop_scores, candidates)
    if fallback != baseline["variants"]["unique_users"]["top5"]:
        raise AssertionError("Изменился общий fallback")
    predictions, results, wide = {}, {}, {}
    saved = defaultdict(dict)
    with (ROOT / "results/model_recommendations.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            saved[row["method"]][row["user_id"]] = [int(row[f"item_{rank}"]) for rank in range(1, 6)]
    for name, mapping in saved.items():
        if set(mapping) != set(users):
            raise AssertionError("Изменена популяция сохранённых рекомендаций")
        key = "saved_" + name
        predictions[key] = [mapping[user] for user in users]
        results[key] = evaluate(predictions[key], users, truth, activity, candidates,
            previous["methods"][name]["timing"], "data/processed/train_interactions.csv (60 seconds)")
        results[key].update({"family": previous["methods"][name]["family"], "config": previous["methods"][name]["config"],
                            "old_60s_protocol_map_at_5": previous["methods"][name]["cohorts"]["submission_all"]["map_at_5"]})
    def score_of(name):
        return results[name]["strict_protocol"]["submission_all"]["map_at_5"]
    for number, config in enumerate(REPEAT_CONFIGS, 1):
        name = f"repeat_{number}"
        history = histories[config["threshold_percent"]]
        scores, timing, audit = model_scores("repeat", config, history, users, candidates)
        predicted, used = rank_with_fallback(scores, candidates, fallback)
        predictions[name], wide[name] = predicted, ranked_lists(scores, candidates, 100)
        results[name] = evaluate(predicted, users, truth, activity, candidates, timing, history["path"], used)
        results[name].update({"family": "repeat", "config": config, "audit": audit,
                              "actual_training_unknown_users": sum(user not in history["users"] for user in users)})
        print(f"{name}: {score_of(name):.9f}", flush=True)
    graphs = {}
    for number, config in enumerate(PPR_CONFIGS, 1):
        name = f"ppr_{number}"
        history = histories[50]
        if number == 1:
            probe_started = time.perf_counter()
            _, probe_timing, probe_audit = model_scores("ppr", config, history, users[:32], candidates, graphs=graphs)
            probe = {"users": 32, "wall_seconds": time.perf_counter() - probe_started,
                     "timing": probe_timing, "convergence": probe_audit["convergence"]}
            print("PPR probe:", probe["wall_seconds"], "seconds", flush=True)
        scores, timing, audit = model_scores("ppr", config, history, users, candidates, graphs=graphs)
        predicted, used = rank_with_fallback(scores, candidates, fallback)
        predictions[name], wide[name] = predicted, ranked_lists(scores, candidates, 100)
        results[name] = evaluate(predicted, users, truth, activity, candidates, timing, history["path"], used)
        results[name].update({"family": "ppr", "config": config, "audit": audit,
                              "actual_training_unknown_users": sum(user not in history["users"] for user in users)})
        print(f"{name}: {score_of(name):.9f}; score={timing['score_seconds']:.3f}s", flush=True)
    best = {family: max((name for name, value in results.items() if value["family"] == family), key=score_of)
            for family in ("repeat", "ppr", "als", "content")}
    metadata, _ = load_train_metadata([ROOT / "data/export_arh_11-20-final.csv"], "2020-03-09 00:00:00", "2020-05-18 00:00:00")
    for family in ("als", "content"):
        name = best[family]
        scores, _, _ = model_scores(family, results[name]["config"], histories[60], users, candidates, metadata)
        repeated, _ = rank_with_fallback(scores, candidates, fallback)
        if repeated != predictions[name]:
            raise AssertionError("Refit сохранённого компонента изменил Top-5: " + name)
        wide[name] = ranked_lists(scores, candidates, 100)
    pop_wide = top_popular(pop_scores, candidates, 100)
    wide["saved_popularity"] = [pop_wide] * len(users)
    sources = {"pop": "saved_popularity", **best}
    for number, weights in enumerate(FUSIONS, 1):
        name = f"wrrf_{number}"
        fusion_started = time.perf_counter()
        predicted, used = weighted_rrf({key: wide[sources[key]] for key in weights}, weights, candidates, fallback, offset=60)
        predictions[name] = predicted
        results[name] = evaluate(predicted, users, truth, activity, candidates,
            {"fusion_seconds": time.perf_counter() - fusion_started, "component_inference_excluded": True},
            "separate train-only components; see component_sources", used)
        results[name].update({"family": "weighted_rrf", "config": {"weights": weights, "offset": 60, "list_limit": 100},
                              "component_sources": {key: sources[key] for key in weights}})
        print(f"{name}: {score_of(name):.9f}", flush=True)
    selected = max(results, key=score_of)
    significance = paired_improvement(predictions[selected], predictions["saved_popularity"], users, truth)
    main_names = ["saved_popularity", best["repeat"], best["ppr"], best["als"], best["content"], selected]
    main_names = list(dict.fromkeys(main_names))
    examples = user_examples(predictions, main_names, users, truth)
    recommendation_path = ROOT / "results/strict_recommendations.csv"
    write_predictions(recommendation_path, predictions, users)
    tests = subprocess.run([sys.executable, "-B", "-m", "unittest", "discover", "-s", str(ROOT / "tests"), "-v"], cwd=ROOT, capture_output=True, text=True)
    if tests.returncode:
        raise RuntimeError(tests.stdout + tests.stderr)
    for path, expected in protected.items():
        if file_sha256(ROOT / path) != expected:
            raise AssertionError("Изменился старый артефакт: " + path)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result = {"protocol_id": POLICY["id"], "policy": POLICY, "primary_population": "same frozen 1260 submission users, empty GT AP=0",
        "candidate_pool": {"count": len(candidates), "source_protocol_sha256": file_sha256(ROOT / "results/popularity_baseline.json")},
        "activity_cohorts": "frozen 60-second train unique-item bands, not changed using validation", "ground_truth_comparison": compare_truth(old_truth, truth, users, candidates),
        "all_validation_truth": {"old_60s_pairs": sum(map(len, old_truth.values())), "strict_80_pairs": sum(map(len, truth.values())),
                                 "old_nonempty_users": len(old_truth), "strict_nonempty_users": len(truth)},
        "row_only_sensitivity": {"scope": "diagnostic only; not used to select the primary label policy", "ground_truth": gt_summary(row_truth, users, candidates),
                                 "popularity_map_at_5": evaluate_cohort(users, row_truth, dict(zip(users, predictions["saved_popularity"])), activity)["map_at_5"]},
        "reported_kaggle_v1": {"public": .28234, "private": .27809, "source": "user-reported after first submission; never used for tuning"},
        "methods": results, "best_components": best, "selected_strategy": selected, "paired_improvement": significance, "ppr_probe": probe,
        "user_examples": examples, "recommendations": {"path": str(recommendation_path.relative_to(ROOT)), "rows": len(predictions) * len(users), "sha256": file_sha256(recommendation_path)},
        "component_rules": {"shared_fallback": "unchanged unique-user popularity from 60-second train", "wide_lists": "Top-100 finite native scores, no per-component padding", "fusion": "sum weight/(60+rank); shared fallback only after fusion"},
        "checks": {"old_artifacts_unchanged": True, "same_candidate_pool_and_population": True, "train_only_graph_profiles_and_history": True,
                   "all_recommendations_valid": True, "unit_tests_passed": True, "unit_test_output": tests.stderr.strip()},
        "protected_sha256": protected,
        "runtime": {"wall_seconds": time.perf_counter() - started, "peak_rss_mib": peak / (1024**2 if sys.platform == "darwin" else 1024)},
        "limitations": ["Participant's label description is not independently verified Kaggle evaluator code.", "No device ID: concurrency masking is a conservative assumption; row-only sensitivity is reported.",
                        "Test spans twelve weeks and public/private partition differs from this ten-week hold-out.", "All hyperparameter/fusion selection uses this one validation window; no Kaggle score tuning."],
    }
    path = ROOT / "results/strict_experiments.json"
    example_path = ROOT / "results/strict_user_examples.csv"
    write_user_examples(example_path, examples)
    write_metrics(ROOT / "results/strict_metrics.csv", results)
    result["user_examples_file"] = {"path": str(example_path.relative_to(ROOT)), "sha256": file_sha256(example_path)}
    replacement_path = ROOT / "results/strict_fusion_diagnostics.json"
    replacement_path.write_text(json.dumps({"protocol_id": POLICY["id"],
        "comparisons": replacement_diagnostics(predictions, users, truth, ["saved_popularity", best["ppr"]]),
        "ppr_vs_repeat": paired_improvement(predictions[best["ppr"]], predictions[best["repeat"]], users, truth)},
        ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
    print("Selected:", selected, score_of(selected), "v2 criterion:", significance, flush=True)
    return result


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
