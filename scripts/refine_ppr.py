"""Ограниченная V3-серия: персональный seed и объяснимый re-ranking PPR.

Запуск: .venv/bin/python scripts/refine_ppr.py
После основной серии проверяется только зафиксированный победитель на early.
--final создаёт V3 лишь при выполнении заранее заданного правила отбора.
"""

import os
for _name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "1"

import argparse
import csv
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_submission_v2 import write_personal_submission
from scripts.evaluate_popularity import load_ground_truth, validation_candidates
from scripts.prepare_interactions import SCHEDULE_FILES, file_sha256
from scripts.strict_experiments import evaluate, load_history, mapped, paired_improvement, write_predictions
from src.fusion import ranked_lists, weighted_rrf
from src.history_reranking import (bounded_rerank, load_train_context, restart_weights,
                                   schedule_affinity, schedule_context)
from src.personal_repeat import repeat_scores
from src.personalized_pagerank import build_graph, personalized_scores
from src.popularity import fit_popularity, top_popular
from src.ranking import rank_with_fallback

MAIN = ("2020-03-09 00:00:00", "2020-05-18 00:00:00", "2020-07-27 00:00:00")
EARLY = ("2020-03-09 00:00:00", "2020-04-13 00:00:00", "2020-05-18 00:00:00")
CONTROL = {"seed": "log_count", "restart": .30, "neighbors": 50, "post": "none"}
# Зафиксированы до просмотра новых метрик. Граф первых вариантов один и тот же.
VARIANTS = {
    "ppr_control": CONTROL,
    "seed_recency_30": {**CONTROL, "seed": "recency", "half_life_days": 30},
    "seed_regularity": {**CONTROL, "seed": "regularity"},
    "rrf_ppr80_repeat20": {**CONTROL, "post": "rrf", "repeat_weight": .20},
    "repeat_boost_20": {**CONTROL, "post": "repeat_boost", "strength": .20},
    "schedule_channel_25": {**CONTROL, "post": "channel", "strength": .25},
    "schedule_slot_25": {**CONTROL, "post": "slot", "strength": .25},
    "schedule_availability_25": {**CONTROL, "post": "availability", "strength": .25},
    "restart_45": {**CONTROL, "restart": .45},
}
DECISION = {
    "main_min_delta": .005, "early_min_delta": .0025,
    "positive_bootstrap_lower_bound_in_both_windows": True,
    "scope": "descriptive post-selection intervals; earlier window is temporal robustness, not untouched confirmation",
    "early_selection": "one main winner frozen before early evaluation; no runner-up search on early",
}


def dump(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def implementation_signature():
    return {name: file_sha256(ROOT / name) for name in (
        "scripts/refine_ppr.py", "src/history_reranking.py", "src/personalized_pagerank.py")}


def history_for(path, start, cutoff, period="train"):
    history = load_history(path, period=period, lower=start, upper=cutoff)
    history.update({"lower": start, "period": period})
    return history


def repeat_feature(history, context, users, candidates):
    """[0,1] вес только многодневных повторов; не смешивает raw count с PPR."""
    user_indices, item_indices = mapped(history, users, candidates)
    counts = history["counts"].copy().astype(np.float64)
    counts.data = np.where((counts.data >= 2) & (context["days"].data >= 2), np.log1p(counts.data), 0)
    counts.eliminate_zeros()
    values = np.zeros((len(users), len(candidates)), dtype=np.float64)
    known_u, known_i = np.flatnonzero(np.asarray(user_indices) >= 0), np.flatnonzero(np.asarray(item_indices) >= 0)
    if len(known_u) and len(known_i):
        values[np.ix_(known_u, known_i)] = counts[np.asarray(user_indices)[known_u]][:, np.asarray(item_indices)[known_i]].toarray()
    maxima = values.max(axis=1, keepdims=True)
    return np.divide(values, maxima, out=np.zeros_like(values), where=maxima > 0)


def score_recipe(recipe, history, context, users, candidates, fallback, schedule=None, cache=None):
    """Каждый cache принадлежит одному train-window; final создаёт свежий."""
    if cache is None:
        cache = {}
    started = time.perf_counter()
    graph_key = ("graph", recipe["neighbors"])
    if graph_key not in cache:
        cache[graph_key] = build_graph(history["counts"], history["items"], normalisation="user_cosine", neighbors=recipe["neighbors"])
    transition, graph_audit = cache[graph_key]
    seed_key = (recipe["seed"], recipe.get("half_life_days"), recipe["restart"], recipe["neighbors"])
    if seed_key not in cache:
        user_indices, item_indices = mapped(history, users, candidates)
        weights = None if recipe["seed"] == "log_count" else restart_weights(
            history, context, recipe["seed"], half_life_days=recipe.get("half_life_days", 30))
        inferred = time.perf_counter()
        scores, audit = personalized_scores(transition, history["counts"], user_indices, item_indices,
            restart=recipe["restart"], weighting="log_count", restart_weights=weights)
        scores[scores <= 0] = np.nan
        cache[seed_key] = (scores, audit, time.perf_counter() - inferred)
    scores, convergence, inference_seconds = cache[seed_key]
    post = recipe["post"]
    if post == "rrf":
        if "repeat_lists" not in cache:
            ui, ii = mapped(history, users, candidates)
            repeat = repeat_scores(history["counts"], history["last"], ui, ii, history["cutoff"])
            cache["repeat_lists"] = ranked_lists(repeat, candidates, 100)
        weight = recipe["repeat_weight"]
        predictions, used = weighted_rrf(
            {"ppr": ranked_lists(scores, candidates, 100), "repeat": cache["repeat_lists"]},
            {"ppr": 1 - weight, "repeat": weight}, candidates, fallback, offset=60)
    else:
        if post == "none":
            ranked = scores
        else:
            if post not in cache:
                cache[post] = (repeat_feature(history, context, users, candidates) if post == "repeat_boost"
                               else schedule_affinity(context, schedule, users, candidates, "joint" if post == "slot" else post))
            ranked = bounded_rerank(scores, cache[post], candidates, strength=recipe["strength"], limit=100)
        predictions, used = rank_with_fallback(ranked, candidates, fallback)
    return predictions, used, {
        "elapsed_seconds": time.perf_counter() - started,
        "ppr_inference_seconds": inference_seconds,
        "graph_build_seconds": graph_audit["elapsed_seconds"],
        "cached_components_reused": len(cache) > 2,
    }, {"graph": graph_audit, "convergence": convergence}


def paired_summary(predictions, reference, users, truth):
    result = paired_improvement(predictions, reference, users, truth)
    # Не наследуем критерий V2, где контрольной точкой была popularity.
    for key in ("v2_criterion", "convincing_for_v2"):
        result.pop(key)
    result["relevant_recommendations_added"] = sum(len((set(a) - set(b)) & truth.get(user, set()))
                                                   for user, a, b in zip(users, predictions, reference))
    result["relevant_recommendations_removed"] = sum(len((set(b) - set(a)) & truth.get(user, set()))
                                                     for user, a, b in zip(users, predictions, reference))
    return result


def run():
    from scripts.prepare_history_context import prepare
    started = time.perf_counter()
    manifest = prepare("development")
    baseline = json.loads((ROOT / "results/popularity_baseline.json").read_text())
    old = json.loads((ROOT / "results/strict_experiments.json").read_text())
    users, candidates = baseline["protocol"]["primary_user_ids"], np.asarray(baseline["protocol"]["candidate_ids"], dtype=np.int64)
    history = history_for(ROOT / "data/processed/session_train_50.csv", *MAIN[:2])
    context = load_train_context(ROOT / manifest["main"]["outputs"]["train_airings"]["path"], history)
    pop_scores, activity = fit_popularity(ROOT / "data/processed/train_interactions.csv")
    fallback = top_popular(pop_scores["unique_users"], candidates)
    truth = load_ground_truth(ROOT / "data/processed/session_validation_80.csv", *MAIN[1:])
    paths = [ROOT / "data" / name for name in SCHEDULE_FILES[:2]]
    schedule_started = time.perf_counter()
    schedule = schedule_context(paths, *MAIN[1:], candidates)
    schedule_seconds = time.perf_counter() - schedule_started
    predictions, results, cache = {}, {}, {}

    def record(name, recipe):
        predicted, used, timing, audit = score_recipe(recipe, history, context, users, candidates, fallback, schedule, cache)
        predictions[name] = predicted
        value = evaluate(predicted, users, truth, activity, candidates, timing, history["path"], used)
        value.update({"recipe": recipe, "versus_control": paired_summary(predicted, predictions["ppr_control"], users, truth), "audit": audit})
        results[name] = value
        print(name, f"MAP={value['strict_protocol']['submission_all']['map_at_5']:.9f}", flush=True)

    for name, recipe in VARIANTS.items():
        record(name, recipe)
        if name == "ppr_control":
            expected = old["methods"]["ppr_2"]["strict_protocol"]["submission_all"]["map_at_5"]
            if abs(results[name]["strict_protocol"]["submission_all"]["map_at_5"] - expected) > 1e-12:
                raise AssertionError("Контрольный PPR не воспроизвёл исходную метрику")
    # Не более одного сочетания двух независимо полезных изменений.
    seeds = ("seed_recency_30", "seed_regularity")
    posts = ("rrf_ppr80_repeat20", "repeat_boost_20", "schedule_channel_25", "schedule_slot_25", "schedule_availability_25")
    delta = lambda name: results[name]["versus_control"]["mean_map_delta"]
    best_seed, best_post = max(seeds, key=delta), max(posts, key=delta)
    if delta(best_seed) >= .002 and delta(best_post) >= .002:
        combined = {**results[best_seed]["recipe"], **{key: value for key, value in results[best_post]["recipe"].items()
                                                   if key in ("post", "strength", "repeat_weight")}}
        record("combined_two_changes", combined)
    selected = max(results, key=lambda name: results[name]["strict_protocol"]["submission_all"]["map_at_5"])
    # Freeze selected recipe before looking at ANY early-window metric.
    cache.clear()  # Dense main scores/features are not needed in the early run.
    early = check_early(results[selected]["recipe"], manifest["early"], users)
    main_gain = results[selected]["versus_control"]
    early_gain = early["versus_control"]
    convincing = (main_gain["mean_map_delta"] >= DECISION["main_min_delta"]
        and early_gain["mean_map_delta"] >= DECISION["early_min_delta"]
        and main_gain["paired_bootstrap_95_interval"][0] > 0 and early_gain["paired_bootstrap_95_interval"][0] > 0)
    write_predictions(ROOT / "results/ppr_refinement_recommendations.csv", predictions, users)
    with (ROOT / "results/ppr_refinement_metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["method", "map_at_5", "map_nonempty_gt", "delta", "ap_improved", "ap_worsened", "unique_items", "elapsed_seconds"])
        for name, value in results.items():
            change = value["versus_control"]
            writer.writerow([name, value["strict_protocol"]["submission_all"]["map_at_5"],
                value["strict_protocol"]["submission_nonempty_validation"]["map_at_5"], change["mean_map_delta"],
                change["users_improved"], change["users_worsened"], value["unique_recommended_programmes"], value["timing"]["elapsed_seconds"]])
    result = {"protocol": "unchanged strict >80% GT, 1260 users, 6384 candidates; empty AP=0; numeric tie-break",
        "implementation_signature": implementation_signature(),
        "train": MAIN[:2], "validation": MAIN[1:], "methods": results, "selected": selected,
        "early_check": early, "decision_rule": DECISION, "convincing_for_v3": bool(convincing),
        "feature_preparation": manifest, "context_audit": context.get("audit"), "schedule_audit": schedule.get("audit"),
        "schedule_preparation_seconds": schedule_seconds,
        "reported_kaggle_v2": {"public": .40902, "private": .38282, "source": "user-reported; not used in selection"},
        "external_methodology": "No new external methodology or code; extends existing PPR and rank fusion implementations.",
        "limitations": ["Repeated selection on one holdout; bootstrap intervals are descriptive, not selection-adjusted.",
                        "Early data already occur in main training; temporal robustness is not an untouched independent test.",
                        "Calendar channel/time preferences are proxies; absolute timezone and actual devices remain unknown."],
        "runtime": {"wall_seconds": time.perf_counter() - started, "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2 if sys.platform == "darwin" else 1024)}}
    dump(ROOT / "results/ppr_refinement.json", result)
    print("Selected", selected, "V3 eligible", convincing, "Early delta", early_gain["mean_map_delta"], flush=True)
    return result


def check_early(recipe, manifest, users):
    history = history_for(ROOT / manifest["outputs"]["train_50"]["path"], *EARLY[:2])
    context = load_train_context(ROOT / manifest["outputs"]["train_airings"]["path"], history)
    fallback_scores, activity = fit_popularity(ROOT / manifest["outputs"]["train_60"]["path"])
    paths = [ROOT / "data" / SCHEDULE_FILES[0]]
    candidates = np.asarray(sorted(validation_candidates(paths, *EARLY[1:])), dtype=np.int64)
    fallback = top_popular(fallback_scores["unique_users"], candidates)
    truth = load_ground_truth(ROOT / manifest["outputs"]["validation_80"]["path"], *EARLY[1:])
    schedule = schedule_context(paths, *EARLY[1:], candidates) if recipe["post"] in ("channel", "slot", "availability") else None
    cache, evaluated, predictions = {}, {}, {}
    for name, configuration in (("ppr_control", CONTROL), ("frozen_main_winner", recipe)):
        predicted, used, timing, audit = score_recipe(configuration, history, context, users, candidates, fallback, schedule, cache)
        predictions[name] = predicted
        evaluated[name] = evaluate(predicted, users, truth, activity, candidates, timing, history["path"], used)
        evaluated[name]["recipe"] = configuration
    write_predictions(ROOT / "results/ppr_refinement_early_recommendations.csv", predictions, users)
    return {"train": EARLY[:2], "validation": EARLY[1:], "users": len(users), "candidate_count": len(candidates),
        "nonempty_gt_users": sum(bool(truth.get(user)) for user in users), "methods": evaluated,
        "versus_control": paired_summary(predictions["frozen_main_winner"], predictions["ppr_control"], users, truth),
        "no_early_holdout_in_graph_seed_preferences_or_fallback": True, "no_retuning_after_early": True}


def final():
    from scripts.prepare_history_context import prepare
    result = json.loads((ROOT / "results/ppr_refinement.json").read_text())
    if not result["convincing_for_v3"]:
        print("Прирост не прошёл заранее заданное правило; submission_v3 не создаётся.")
        return
    if result["implementation_signature"] != implementation_signature() or result["decision_rule"] != DECISION:
        raise ValueError("Код или правило отбора изменились после эксперимента; нельзя применять устаревший результат")
    for name in ("main", "early"):
        for relative, expected in result["feature_preparation"][name]["source_signature"].items():
            info = (ROOT / relative).stat()
            if {"bytes": info.st_size, "mtime_ns": info.st_mtime_ns} != expected:
                raise ValueError("Изменился источник после эксперимента: " + relative)
    started = time.perf_counter()
    original_hash = file_sha256(ROOT / "submission_v2.csv")
    manifest = prepare("final")
    baseline = json.loads((ROOT / "results/popularity_baseline.json").read_text())
    users = baseline["protocol"]["primary_user_ids"]
    start, cutoff, end = MAIN[0], MAIN[2], "2020-10-19 00:00:00"
    history = history_for(ROOT / manifest["outputs"]["train_50"]["path"], start, cutoff, "final_train")
    context = load_train_context(ROOT / manifest["outputs"]["train_airings"]["path"], history)
    paths = [ROOT / "data" / name for name in SCHEDULE_FILES]
    candidates = np.asarray(sorted(validation_candidates(paths, cutoff, end)), dtype=np.int64)
    pop_scores, _ = fit_popularity(ROOT / "data/processed/final_interactions.csv", expected_period="final_train")
    fallback = top_popular(pop_scores["unique_users"], candidates)
    recipe = result["methods"][result["selected"]]["recipe"]
    schedule = schedule_context(paths, cutoff, end, candidates) if recipe["post"] in ("channel", "slot", "availability") else None
    predicted, used, timing, audit = score_recipe(recipe, history, context, users, candidates, fallback, schedule)
    check = write_personal_submission(ROOT / "data/submission.csv", ROOT / "submission_v3.csv", dict(zip(users, predicted)), candidates)
    if file_sha256(ROOT / "submission_v2.csv") != original_hash:
        raise AssertionError("Изменён V2")
    dump(ROOT / "results/submission_v3.json", {"selected": result["selected"], "recipe": recipe,
        "train": [start, cutoff], "test": [cutoff, end], "history_pairs": history["counts"].nnz,
        "history_users": len(history["users"]), "history_items": len(history["items"]),
        "candidate_count": len(candidates), "fallback_users": int(sum(used)), "unique_items": len({i for row in predicted for i in row}),
        "submission": {"path": "submission_v3.csv", **check}, "model_timing": timing, "model_audit": audit,
        "runtime_seconds": time.perf_counter() - started, "v2_unchanged": True, "kaggle_submitted": False})
    print("Готов submission_v3.csv", check, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--final", action="store_true")
    arguments = parser.parse_args()
    with threadpool_limits(limits=1):
        final() if arguments.final else run()
