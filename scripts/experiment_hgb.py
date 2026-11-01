"""Four temporal HGB ablations; reuse frozen base algorithms and preserve V3.

.venv/bin/python scripts/experiment_hgb.py
.venv/bin/python scripts/experiment_hgb.py --final   # only if offline gate passes
"""

import os
for _name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "1"

import argparse
import csv
import gc
import json
import platform
import resource
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_submission_v2 import write_personal_submission
from scripts.evaluate_popularity import load_ground_truth, validation_candidates
from scripts.prepare_interactions import SCHEDULE_FILES, file_sha256, wall_seconds
from scripts.refine_ppr import EARLY, MAIN, dump, history_for, paired_summary
from scripts.strict_experiments import evaluate, mapped, model_scores, write_predictions
from src.content_based import load_train_metadata
from src.fusion import ranked_lists
from src.hgb_reranker import (FEATURE_GROUPS, candidate_diagnostics, candidate_union,
                              history_features, rank_candidate_predictions)
from src.history_reranking import bounded_rerank, load_train_context, schedule_affinity, schedule_context
from src.personal_repeat import repeat_scores
from src.personalized_pagerank import build_graph, personalized_scores
from src.popularity import fit_popularity, top_popular
from src.ranking import rank_with_fallback

DATA = ROOT / "data/processed/hgb"
ALS = {"factors": 16, "regularization": .1, "weighting": "log_count", "confidence_alpha": 10, "iterations": 20}
CONTENT = {"feature_set": "extended", "history_weighting": "log1p"}
HGB = {"learning_rate": .08, "max_iter": 100, "max_leaf_nodes": 15, "max_depth": 4,
       "min_samples_leaf": 100, "l2_regularization": 10.0, "max_bins": 128,
       "early_stopping": False, "class_weight": "balanced", "categorical_features": None, "random_state": 42}
BASE_NAMES = ["ppr_score", "ppr_rank", "ppr_relative_score", "v3_rank", "als_score", "als_rank",
              "content_score", "content_rank", "log_repeat_count", "popularity_share", "popularity_rank",
              "log_future_airings", "airings_per_forecast_day", "forecast_day_coverage", "availability"]
DECISION = {"reference": "V3 strict MAP 0.524651455026455", "minimum_main_delta": .01,
            "positive_descriptive_bootstrap_lower_bound": True, "selection": "best of four predeclared feature groups"}
SOURCE = "https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html"


def feature_signature():
    return {p: file_sha256(ROOT / p) for p in ("scripts/experiment_hgb.py", "src/hgb_reranker.py",
        "src/personalized_pagerank.py", "src/matrix_factorization.py", "src/content_based.py",
        "src/history_reranking.py", "src/ranking.py", "src/fusion.py", "src/personal_repeat.py",
        "scripts/strict_experiments.py", "scripts/refine_ppr.py")}


def temporal_check(history_start, feature_cutoff, target_start, target_end):
    if not wall_seconds(history_start) < wall_seconds(feature_cutoff) <= wall_seconds(target_start) < wall_seconds(target_end):
        raise ValueError("Historical features must precede a nonempty target interval")


def load_saved_oof(name, expected):
    """Final fit must reuse the selected experiment's actual OOF snapshots."""
    path, meta_path = DATA / f"{name}.npz", DATA / f"{name}.json"
    if not path.exists() or not meta_path.exists():
        raise ValueError("Saved OOF snapshot missing; refusing historical feature regeneration")
    meta = json.loads(meta_path.read_text())
    if meta["signature"] != expected["signature"]:
        raise ValueError("OOF snapshot differs from selected experiment")
    for relative, stat in meta["signature"]["inputs"].items():
        current = (ROOT / relative).stat()
        if [current.st_size, current.st_mtime_ns] != stat:
            raise ValueError("OOF source changed after selection: " + relative)
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key] for key in saved.files}, meta


def source_lists(history, users, candidates):
    ui, ii = mapped(history, users, candidates)
    repeats = repeat_scores(history["counts"], history["last"], ui, ii, history["cutoff"])
    result = ranked_lists(repeats, candidates, 200)
    del repeats
    return result


def sampled_scores_and_ranks(scores, columns, candidate_ids):
    values, ranks = np.empty(columns.shape, dtype=np.float32), np.full(columns.shape, np.nan, dtype=np.float32)
    for row, chosen in enumerate(columns):
        values[row] = scores[row, chosen]
        finite = np.flatnonzero(np.isfinite(scores[row]))
        order = finite[np.lexsort((candidate_ids[finite], -scores[row, finite]))]
        lookup = np.full(len(candidate_ids), np.nan, dtype=np.float32)
        lookup[order] = np.arange(1, len(order) + 1)
        ranks[row] = lookup[chosen]
    return values, ranks


def snapshot(name, bounds, users=None, training_users=False, final=False):
    """Cache OOF features with their own cutoff; labels are attached only AFTER X.

    Main contains past-defined users plus supplied evaluation users so final
    refitting can reuse all main OOF rows without rebuilding them from future data.
    """
    started = time.perf_counter()
    start, cutoff, end = bounds
    temporal_check(start, cutoff, cutoff, end)
    mode = "final" if final else "early" if name == "early" else "main"
    manifest = json.loads((ROOT / f"results/history_context_{mode}.json").read_text())
    history_path = ROOT / manifest["outputs"]["train_50"]["path"]
    pop_path = ROOT / ("data/processed/final_interactions.csv" if final else manifest["outputs"]["train_60"]["path"])
    period = "final_train" if final else "train"
    requested = [] if users is None else users
    inputs = [history_path, pop_path, ROOT / manifest["outputs"]["train_airings"]["path"]]
    if not final:
        inputs.append(ROOT / manifest["outputs"]["validation_80"]["path"])
    inputs += [ROOT / "data" / f for f in SCHEDULE_FILES[:(3 if final else 2)]]
    signature = {"implementation": feature_signature(), "bounds": list(bounds), "requested_users": requested,
                 "training_users": training_users, "ALS": ALS, "CONTENT": CONTENT,
                 "inputs": {str(p.relative_to(ROOT)): [p.stat().st_size, p.stat().st_mtime_ns] for p in inputs}}
    DATA.mkdir(exist_ok=True)
    path, meta_path = DATA / f"{name}.npz", DATA / f"{name}.json"
    if path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta["signature"] == signature:
            print("Feature snapshot cache", name, flush=True)
            with np.load(path, allow_pickle=False) as saved:
                return {key: saved[key] for key in saved.files}, meta
    history = history_for(history_path, start, cutoff, period)
    if training_users:
        users = list(history["users"])
        users.extend(user for user in requested if user not in history["users"])
    pop, activity = fit_popularity(pop_path, expected_period=period)
    paths = [ROOT / "data" / f for f in SCHEDULE_FILES[:(3 if final else 2)]]
    candidates = np.asarray(sorted(validation_candidates(paths, cutoff, end)), dtype=np.int64)
    fallback = top_popular(pop["unique_users"], candidates)
    pop_order = top_popular(pop["unique_users"], candidates, len(candidates))
    schedule = schedule_context(paths, cutoff, end, candidates)
    availability = schedule_affinity({}, schedule, users, candidates, "availability")
    ui, ii = mapped(history, users, candidates)
    base_timing = {}
    tick = time.perf_counter()
    transition, graph = build_graph(history["counts"], history["items"], normalisation="user_cosine", neighbors=50)
    ppr, convergence = personalized_scores(transition, history["counts"], ui, ii, restart=.30, weighting="log_count")
    ppr[ppr <= 0] = np.nan
    v3 = bounded_rerank(ppr, availability, candidates, strength=.25, limit=100)
    v3_predictions, _ = rank_with_fallback(v3, candidates, fallback)
    lists = {"ppr": ranked_lists(v3, candidates, 100)}
    base_timing["ppr_and_v3_seconds"] = time.perf_counter() - tick
    ppr_top = np.nanmax(ppr, axis=1, initial=0)
    base_scores = {"ppr": ppr}
    del transition
    for family, config in (("als", ALS), ("content", CONTENT)):
        tick = time.perf_counter()
        metadata = load_train_metadata([ROOT / "data" / f for f in SCHEDULE_FILES[:2]], start, cutoff)[0] if family == "content" else None
        score, timing, _ = model_scores(family, config, history, users, candidates, metadata)
        base_scores[family] = score
        lists[family] = ranked_lists(score, candidates, 200)
        base_timing[family] = {**timing, "including_metadata_and_lists_seconds": time.perf_counter() - tick}
    lists["repeat"] = source_lists(history, users, candidates)
    pairs = candidate_union(lists, v3_predictions, pop_order, k=200)
    column_index = {int(item): number for number, item in enumerate(candidates)}
    cols = np.asarray([[column_index[int(item)] for item in row] for row in pairs], dtype=np.int32)
    x_base = np.zeros((*pairs.shape, len(BASE_NAMES)), dtype=np.float32)
    for key, value_col, rank_col in (("ppr", 0, 1), ("als", 4, 5), ("content", 6, 7)):
        x_base[:, :, value_col], x_base[:, :, rank_col] = sampled_scores_and_ranks(base_scores[key], cols, candidates)
    x_base[:, :, 2] = np.divide(x_base[:, :, 0], ppr_top[:, None], out=np.zeros(pairs.shape), where=ppr_top[:, None] > 0)
    _, x_base[:, :, 3] = sampled_scores_and_ranks(v3, cols, candidates)
    pop_rank = {item: rank for rank, item in enumerate(pop_order, 1)}
    training_user_count = max(1, len(activity))
    forecast_days = (wall_seconds(end) - wall_seconds(cutoff)) / 86400
    item_map = {int(item): position for position, item in enumerate(history["items"])}
    for row, user in enumerate(users):
        number = history["users"].get(user)
        counts = {} if number is None else dict(zip(history["counts"].getrow(number).indices, history["counts"].getrow(number).data))
        for col, item in enumerate(pairs[row]):
            item = int(item)
            exposure = schedule["items"].get(item, {})
            count = exposure.get("airings", 0)
            x_base[row, col, 8:15] = (np.log1p(counts.get(item_map.get(item), 0)), pop["unique_users"].get(item, 0) / training_user_count,
                pop_rank[item], np.log1p(count), count / forecast_days, exposure.get("days", 0) / forecast_days,
                availability[row, column_index[item]])
    del base_scores, ppr, v3, availability, lists
    gc.collect()
    context = load_train_context(ROOT / manifest["outputs"]["train_airings"]["path"], history)
    extra, names_extra = history_features(history, context, users, pairs, schedule, forecast_days)
    X = np.concatenate((x_base, extra), axis=2).reshape(-1, len(BASE_NAMES) + len(names_extra))
    del extra, x_base
    # The only use of this fold's future viewing starts here, after features.
    truth = {} if final else load_ground_truth(ROOT / manifest["outputs"]["validation_80"]["path"], cutoff, end)
    y = np.asarray([int(int(item) in truth.get(user, set())) for user, row in zip(users, pairs) for item in row], dtype=np.uint8)
    warm = np.asarray([user in history["users"] for user in users])
    arrays = {"X": X, "y": y, "pairs": pairs, "users": np.asarray(users), "warm": warm,
              "v3": np.asarray(v3_predictions, dtype=np.int64), "fallback": np.asarray(fallback, dtype=np.int64),
              "candidate_pool": candidates, "feature_names": np.asarray(BASE_NAMES + names_extra)}
    np.savez_compressed(path, **arrays)
    meta = {"signature": signature, "bounds": bounds, "feature_cutoff": cutoff, "target_interval": [cutoff, end],
        "users": len(users), "training_user_definition": "past strict50 users plus known evaluation users when requested",
        "pairs": len(y), "positive_labels": None if final else int(y.sum()), "features": X.shape[1],
        "candidate_diagnostics": None if final else candidate_diagnostics(pairs, users, truth),
        "max_history_observation": float(history["last"].data.max()), "cutoff_seconds": history["cutoff"],
        "all_features_constructed_before_label_access": True, "metadata_end_exclusive": cutoff,
        "base_timing": base_timing, "graph": graph, "convergence": convergence, "context": context["audit"],
        "schedule": schedule["audit"], "runtime_seconds": time.perf_counter() - started,
        "path": str(path.relative_to(ROOT))}
    dump(meta_path, meta)
    print("Feature snapshot", name, len(users), X.shape, "positives", meta["positive_labels"], flush=True)
    return arrays, meta


def feature_columns(names, variant):
    if variant not in ("A", "B", "C", "D"):
        raise ValueError("Expected one of the four predeclared ablations")
    allowed = BASE_NAMES.copy()
    for letter in ("B", "C", "D"):
        if letter <= variant:
            allowed.extend(FEATURE_GROUPS[letter])
    return [list(names).index(name) for name in allowed]


def select_rows(data, users):
    lookup = {str(user): number for number, user in enumerate(data["users"])}
    selected = np.asarray([lookup[user] for user in users])
    shape = len(data["users"]), data["pairs"].shape[1], data["X"].shape[1]
    return {"X": data["X"].reshape(shape)[selected].reshape(-1, shape[2]),
            **{key: data[key][selected] for key in ("pairs", "warm", "v3")}, "feature_names": data["feature_names"], "fallback": data["fallback"],
            "candidate_pool": data["candidate_pool"], "users": np.asarray(users)}


def run():
    started = time.perf_counter()
    baseline = json.loads((ROOT / "results/popularity_baseline.json").read_text())
    users = baseline["protocol"]["primary_user_ids"]
    train, train_meta = snapshot("early", EARLY, training_users=True)
    main, main_meta = snapshot("main", MAIN, users=users, training_users=True)
    if not np.array_equal(train["feature_names"], main["feature_names"]):
        raise ValueError("Training and evaluation feature schemas differ")
    test = select_rows(main, users)
    truth = load_ground_truth(ROOT / "data/processed/session_validation_80.csv", *MAIN[1:])
    _, activity = fit_popularity(ROOT / "data/processed/train_interactions.csv")
    control = test["v3"].tolist()
    results, predictions, models = {}, {"v3_control": control}, {}
    reference = evaluate(control, users, truth, activity, test["candidate_pool"], {}, "unchanged V3")
    if abs(reference["strict_protocol"]["submission_all"]["map_at_5"] - .524651455026455) > 1e-12:
        raise AssertionError("V3 reference changed")
    for variant in ("A", "B", "C", "D"):
        columns = feature_columns(train["feature_names"], variant)
        tick = time.perf_counter()
        model = HistGradientBoostingClassifier(**HGB)
        model.fit(train["X"][:, columns], train["y"])
        fit_seconds = time.perf_counter() - tick
        tick = time.perf_counter()
        probabilities = model.predict_proba(test["X"][:, columns])[:, 1].reshape(test["pairs"].shape)
        predicted = rank_candidate_predictions(probabilities, test["pairs"], test["warm"], test["fallback"])
        score_seconds = time.perf_counter() - tick
        value = evaluate(predicted, users, truth, activity, test["candidate_pool"],
            {"fit_seconds": fit_seconds, "recommend_seconds": score_seconds}, "HGB fit early OOF rows only", ~test["warm"])
        value.update({"features": [str(train["feature_names"][i]) for i in columns], "parameters": HGB,
            "versus_v3": paired_summary(predicted, control, users, truth),
            "popular_top5_fraction": sum(len(set(row) & set(test["fallback"][:5])) for row in predicted) / (5 * len(users)),
            "iterations": model.n_iter_, "internal_early_stopping": bool(model.do_early_stopping_)})
        results[variant], predictions[variant] = value, predicted
        joblib.dump(model, DATA / f"model_{variant}.joblib")
        models[variant] = str((DATA / f"model_{variant}.joblib").relative_to(ROOT))
        print("HGB", variant, f"MAP={value['strict_protocol']['submission_all']['map_at_5']:.9f}",
              "fit", round(fit_seconds, 2), "seconds", flush=True)
        del model
        gc.collect()
    selected = max(results, key=lambda key: results[key]["strict_protocol"]["submission_all"]["map_at_5"])
    gain = results[selected]["versus_v3"]
    convincing = gain["mean_map_delta"] >= DECISION["minimum_main_delta"] and gain["paired_bootstrap_95_interval"][0] > 0
    write_predictions(ROOT / "results/hgb_recommendations.csv", predictions, users)
    with (ROOT / "results/hgb_metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream); writer.writerow(["variant", "map_at_5", "delta_v3", "ap_improved", "ap_worsened", "unique_items", "fit_seconds", "recommend_seconds"])
        for variant, value in results.items():
            change = value["versus_v3"]
            writer.writerow([variant, value["strict_protocol"]["submission_all"]["map_at_5"], change["mean_map_delta"],
                change["users_improved"], change["users_worsened"], value["unique_recommended_programmes"], *value["timing"].values()])
    result = {"methods": results, "reference": reference, "selected": selected, "convincing_for_v4": bool(convincing), "decision": DECISION,
        "candidate_generation": "V3 Top5 plus native V3 Top100; unseen items interleaved Repeat/ALS/CBF/popularity until exactly 200",
        "candidate_diagnostics_primary": candidate_diagnostics(test["pairs"], users, truth),
        "training_snapshot": train_meta, "main_snapshot": main_meta, "models": models,
        "training_scheme": {"HGB_train_features_cutoff": EARLY[1], "HGB_targets": EARLY[1:], "main_features_cutoff": MAIN[1], "main_targets": MAIN[1:],
            "no_target_overlap": EARLY[2] == MAIN[1], "no_main_rows_used_in_HGB_fit": True, "negatives": "all unobserved candidate pairs, not explicit dislikes"},
        "source": SOURCE, "dependencies": {"scikit-learn": sklearn.__version__, "joblib": joblib.__version__, "numpy": np.__version__, "python": platform.python_version()},
        "implementation": feature_signature(), "runtime": {"wall_seconds": time.perf_counter() - started,
            "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2 if sys.platform == "darwin" else 1024)},
        "limitations": ["Main holdout repeatedly used before this experiment; bootstrap is descriptive after four-group selection.",
            "Single meta-training target horizon 35 days differs from main70 and test84; balanced outputs are ranking scores, not calibrated probabilities.",
            "Cold users keep the common popularity fallback; negative labels mean no observed strict positive, not dislike."]}
    dump(ROOT / "results/hgb_experiments.json", result)
    print("Selected", selected, "V4 eligible", convincing, "delta", gain["mean_map_delta"], flush=True)


def final():
    result = json.loads((ROOT / "results/hgb_experiments.json").read_text())
    if not result["convincing_for_v4"]:
        print("Убедительного прироста нет; V3 сохранён, V4 не создаётся.")
        return
    if result["implementation"] != feature_signature():
        raise ValueError("HGB implementation changed after selection")
    started = time.perf_counter()
    previous = {name: file_sha256(ROOT / name) for name in ("submission.csv", "submission_v2.csv", "submission_v3.csv")}
    primary = json.loads((ROOT / "results/popularity_baseline.json").read_text())["protocol"]["primary_user_ids"]
    early, em = load_saved_oof("early", result["training_snapshot"])
    main, mm = load_saved_oof("main", result["main_snapshot"])
    if em["target_interval"][1] > mm["target_interval"][0] or mm["target_interval"][1] > MAIN[2]:
        raise ValueError("OOF targets overlap or extend into test")
    forecast, fm = snapshot("final", (MAIN[0], MAIN[2], "2020-10-19 00:00:00"), users=primary, final=True)
    if not (np.array_equal(early["feature_names"], main["feature_names"]) and np.array_equal(early["feature_names"], forecast["feature_names"])):
        raise ValueError("Historical and final feature schemas differ")
    columns = feature_columns(early["feature_names"], result["selected"])
    X = np.concatenate((early["X"][:, columns], main["X"][:, columns]))
    y = np.concatenate((early["y"], main["y"]))
    tick = time.perf_counter()
    model = HistGradientBoostingClassifier(**HGB).fit(X, y)
    fit_seconds = time.perf_counter() - tick
    joblib.dump(model, DATA / "model_final.joblib")
    probabilities = model.predict_proba(forecast["X"][:, columns])[:, 1].reshape(forecast["pairs"].shape)
    predicted = rank_candidate_predictions(probabilities, forecast["pairs"], forecast["warm"], forecast["fallback"])
    checks = write_personal_submission(ROOT / "data/submission.csv", ROOT / "submission_v4.csv", dict(zip(primary, predicted)), forecast["candidate_pool"])
    if any(file_sha256(ROOT / name) != digest for name, digest in previous.items()):
        raise AssertionError("Previous submission changed")
    dump(ROOT / "results/submission_v4_hgb.json", {"selected": result["selected"], "parameters": HGB,
        "meta_training_rows": len(y), "meta_positive_labels": int(y.sum()), "OOF_targets": [em["target_interval"], mm["target_interval"]],
        "OOF_features_reused_at_original_cutoffs": True, "final_snapshot": fm, "fit_seconds": fit_seconds,
        "submission": {"path": "submission_v4.csv", **checks}, "fallback_users": int((~forecast["warm"]).sum()),
        "unique_items": len({item for row in predicted for item in row}), "previous_submissions_unchanged": True,
        "runtime_seconds": time.perf_counter() - started, "kaggle_submitted": False})
    print("Готов submission_v4.csv", checks, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--final", action="store_true")
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        final() if args.final else run()
