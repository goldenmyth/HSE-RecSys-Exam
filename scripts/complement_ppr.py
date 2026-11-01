"""V4: train-50 ALS/CBF, диагностика дополнения V3 и максимум восемь конфигураций."""

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
from scripts.evaluate_popularity import activity_band, load_ground_truth, validation_candidates
from scripts.prepare_interactions import SCHEDULE_FILES, file_sha256
from scripts.refine_ppr import MAIN, EARLY, dump, history_for, paired_summary
from scripts.strict_experiments import evaluate, mapped, model_scores, write_predictions
from src.content_based import load_train_metadata
from src.fusion import native_agreement, ranked_lists
from src.history_reranking import bounded_rerank, schedule_affinity, schedule_context
from src.personalized_pagerank import build_graph, personalized_scores
from src.popularity import fit_popularity, top_popular
from src.ranking import rank_with_fallback

ALS_CONFIGS = [
    {"factors": factors, "regularization": .1, "weighting": "log_count", "confidence_alpha": 10, "iterations": 20}
    for factors in (32, 16)
]
CONTENT_CONFIG = {"feature_set": "extended", "history_weighting": "log1p"}
V3 = {"family": "ppr", "neighbors": 50, "restart": .30}
MAX_CONFIGS = 8
DECISION = {"main_min_delta": .005, "main_bootstrap_lower_bound_positive": True, "early_min_delta": 0.0,
            "early_selection": "only frozen main winner; no runner-up search", "intervals": "descriptive post-selection, not independent confirmation"}
SIGNAL_GATE = {"actionable_novel_hits_top20_min": 63, "boundary_flagged_slots_min": 126,
               "boundary_relevance_lift_min": 1.05,
               "scope": "exploratory aggregate selection only; no per-user validation-dependent policy"}


def signatures():
    return {path: file_sha256(ROOT / path) for path in (
        "scripts/complement_ppr.py", "src/fusion.py", "src/matrix_factorization.py", "src/content_based.py", "src/personalized_pagerank.py", "src/history_reranking.py")}


def knn_scores(history, users, candidates):
    transition, audit = build_graph(history["counts"], history["items"], normalisation="cosine", neighbors=100)
    ui, ii = mapped(history, users, candidates)
    weights = history["counts"].astype(np.float64)
    np.log1p(weights.data, out=weights.data)
    score = np.full((len(users), len(candidates)), np.nan)
    ku, ki = np.flatnonzero(np.asarray(ui) >= 0), np.flatnonzero(np.asarray(ii) >= 0)
    scored = time.perf_counter()
    score[np.ix_(ku, ki)] = (weights[np.asarray(ui)[ku]] @ transition)[:, np.asarray(ii)[ki]].toarray()
    score[score <= 0] = np.nan
    return score, {"fit_seconds": audit["elapsed_seconds"], "recommend_seconds": time.perf_counter() - scored}, audit


def ppr_v3_scores(history, users, candidates, availability, recipe, graphs):
    started = time.perf_counter()
    key = (recipe.get("normalisation", "user_cosine"), recipe["neighbors"])
    graph_reused = key in graphs
    if not graph_reused:
        graphs[key] = build_graph(history["counts"], history["items"], normalisation=key[0], neighbors=key[1])
    transition, audit = graphs[key]
    fitted = time.perf_counter()
    ui, ii = mapped(history, users, candidates)
    scores, convergence = personalized_scores(transition, history["counts"], ui, ii, restart=recipe["restart"], weighting="log_count")
    scores[scores <= 0] = np.nan
    scores = bounded_rerank(scores, availability, candidates, strength=.25, limit=100)
    return scores, {"fit_seconds": 0.0 if graph_reused else fitted - started,
                    "recommend_seconds": time.perf_counter() - fitted, "graph_reused": graph_reused}, {
        "graph": audit, "convergence": convergence, "availability_strength": .25, "native_candidate_limit": 100}


def diagnostics(scores, v3_scores, users, candidates, truth, activity):
    """Широкая novelty отдельно от реально доступных bounded boost кандидатов."""
    aux_lists = ranked_lists(scores, candidates, 50)
    base_lists = ranked_lists(v3_scores, candidates, 100)
    positions = {int(item): col for col, item in enumerate(candidates)}
    stats = {str(k): {"new_hits": 0, "new_slots": 0, "actionable_new_hits": 0, "actionable_new_slots": 0,
                      "users_with_actionable_hit": 0, "aux_hits": 0, "v3_hits_same_list_length": 0,
                      "jaccard_sum": 0.0, "common_pool_aux_hits": 0, "common_pool_v3_hits": 0} for k in (20, 50)}
    bands, concordance, base_concordance = {}, [], []
    flagged_hits = flagged_slots = unflagged_hits = unflagged_slots = pair_count = boundary_scoreable = boundary_slots = 0
    for row, (user, aux, base) in enumerate(zip(users, aux_lists, base_lists)):
        relevant = truth.get(user, set())
        base5, base100 = set(base[:5]), set(base)
        band = activity_band(activity.get(user, 0))
        group = bands.setdefault(band, {"users": 0, "new_hits_top20": 0, "actionable_hits_top20": 0})
        group["users"] += 1
        common = [item for item in base if np.isfinite(scores[row, positions[item]])]
        aux_common = sorted(common, key=lambda item: (-float(scores[row, positions[item]]), item))
        for k in (20, 50):
            value = stats[str(k)]
            a, b = set(aux[:k]), set(base[:k])
            novel, actionable = a - base5, (a - base5) & base100
            value["new_hits"] += len(novel & relevant)
            value["new_slots"] += len(novel)
            value["actionable_new_hits"] += len(actionable & relevant)
            value["actionable_new_slots"] += len(actionable)
            value["users_with_actionable_hit"] += bool(actionable & relevant)
            value["aux_hits"] += len(a & relevant)
            value["v3_hits_same_list_length"] += len(b & relevant)
            value["jaccard_sum"] += len(a & b) / len(a | b) if a | b else 0.0
            value["common_pool_aux_hits"] += len(set(aux_common[:k]) & relevant)
            value["common_pool_v3_hits"] += len(set(common[:k]) & relevant)
            if k == 20:
                group["new_hits_top20"] += len(novel & relevant)
                group["actionable_hits_top20"] += len(actionable & relevant)
        top_aux = set(aux[:20])
        for item in base[3:20]:
            boundary_slots += 1
            if not np.isfinite(scores[row, positions[item]]):
                continue
            boundary_scoreable += 1
            if item in top_aux:
                flagged_slots += 1
                flagged_hits += item in relevant
            else:
                unflagged_slots += 1
                unflagged_hits += item in relevant
        # Только пары по разные стороны cutoff, которые boost <=10% может переставить.
        corrected, previous, count = 0.0, 0.0, 0
        for high in base[3:5]:
            for low in base[5:20]:
                if (high in relevant) == (low in relevant):
                    continue
                h, l = positions[high], positions[low]
                if not np.isfinite(scores[row, [h, l]]).all() or v3_scores[row, h] > 1.1 * v3_scores[row, l]:
                    continue
                positive, negative = (h, l) if high in relevant else (l, h)
                difference = float(scores[row, positive] - scores[row, negative])
                corrected += 1 if difference > 0 else .5 if difference == 0 else 0
                previous += high in relevant
                count += 1
        if count:
            concordance.append(corrected / count)
            base_concordance.append(previous / count)
            pair_count += count
    for value in stats.values():
        value["marginal_precision"] = value["new_hits"] / value["new_slots"] if value["new_slots"] else 0
        value["actionable_marginal_precision"] = value["actionable_new_hits"] / value["actionable_new_slots"] if value["actionable_new_slots"] else 0
        value["mean_jaccard"] = value.pop("jaccard_sum") / len(users)
    precision = flagged_hits / flagged_slots if flagged_slots else 0
    other_precision = unflagged_hits / unflagged_slots if unflagged_slots else 0
    useful = (stats["20"]["actionable_new_hits"] >= SIGNAL_GATE["actionable_novel_hits_top20_min"]
              and flagged_slots >= SIGNAL_GATE["boundary_flagged_slots_min"]
              and unflagged_slots > 0
              and precision > SIGNAL_GATE["boundary_relevance_lift_min"] * other_precision)
    return {"top_lists": stats, "by_activity": bands, "unknown_user_rows": int(np.count_nonzero(~np.isfinite(scores).any(axis=1))),
        "boundary": {"ranks": [4, 20], "scoreable_fraction": boundary_scoreable / boundary_slots if boundary_slots else 0,
            "flagged_slots": flagged_slots, "unflagged_slots": unflagged_slots,
            "flagged_hits": flagged_hits, "unflagged_hits": unflagged_hits,
            "flagged_relevance_precision": precision, "unflagged_relevance_precision": other_precision,
            "near_cutoff_pair_users": len(concordance), "near_cutoff_pairs": pair_count,
            "aux_concordance_mean_per_user": float(np.mean(concordance)) if concordance else None,
            "v3_concordance_same_pairs": float(np.mean(base_concordance)) if base_concordance else None,
            "pair_policy": "ranks4-5 versus6-20, opposite GT, score ratio<=1.10, both native scores finite; users equally weighted"},
        "useful_for_small_boost": bool(useful)}


def write_metrics(results, early):
    bands = ("cold_0", "light_1_50", "medium_51_200", "heavy_201_plus")
    control = results["v3_control"]["strict_protocol"]["submission_all"]["by_train_activity"]
    with (ROOT / "results/ppr_complement_metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["method", "map_at_5", "delta_v3", "early_map_at_5", "ap_improved", "ap_worsened", "fit_seconds", "recommend_seconds", *["delta_" + b for b in bands]])
        for name, value in results.items():
            change, groups = value["versus_v3"], value["strict_protocol"]["submission_all"]["by_train_activity"]
            early_map = early["methods"].get(name, {}).get("strict_protocol", {}).get("submission_all", {}).get("map_at_5")
            writer.writerow([name, value["strict_protocol"]["submission_all"]["map_at_5"], change["mean_map_delta"], early_map,
                change["users_improved"], change["users_worsened"], value["timing"].get("fit_seconds", 0), value["timing"].get("recommend_seconds", 0),
                *[groups[b]["map_at_5"] - control[b]["map_at_5"] for b in bands]])


def evaluate_window(bounds, users, candidates, history_path, pop_path, gt_path, recipes=None):
    started = time.perf_counter()
    start, cutoff, end = bounds
    history = history_for(history_path, start, cutoff)
    pop_scores, activity = fit_popularity(pop_path)
    fallback = top_popular(pop_scores["unique_users"], candidates)
    truth = load_ground_truth(gt_path, cutoff, end)
    paths = [ROOT / "data" / name for name in SCHEDULE_FILES[:2]]
    schedule = schedule_context(paths, cutoff, end, candidates)
    availability = schedule_affinity({}, schedule, users, candidates, "availability")
    graphs, scores_by_name, predictions, results, wide, diag = {}, {}, {}, {}, {}, {}
    metadata = None

    def raw_model(recipe):
        nonlocal metadata
        if recipe["family"] == "ppr":
            return ppr_v3_scores(history, users, candidates, availability, recipe, graphs)
        if recipe["family"] in ("als", "content"):
            if recipe["family"] == "content" and metadata is None:
                metadata, _ = load_train_metadata(paths, start, cutoff)
            score, timing, audit = model_scores(recipe["family"], recipe["config"], history, users, candidates, metadata)
            return score, {"fit_seconds": timing["train_seconds"], "recommend_seconds": timing["score_seconds"]}, audit
        if recipe["family"] == "knn":
            return knn_scores(history, users, candidates)
        raise ValueError("Неизвестная модель")

    def record(name, recipe):
        if recipe["family"] == "agreement":
            tick = time.perf_counter()
            feature = np.zeros_like(scores_by_name["v3_control"])
            for source in recipe["components"]:
                if source not in scores_by_name:
                    score, _, _ = raw_model(recipe["components"][source])
                    scores_by_name[source] = score
                feature += native_agreement(scores_by_name[source], candidates, top_n=20)
            feature /= len(recipe["components"])
            score = bounded_rerank(scores_by_name["v3_control"], feature, candidates, strength=recipe["strength"], limit=100)
            timing, audit = {"fit_seconds": 0, "recommend_seconds": time.perf_counter() - tick, "component_cost_excluded": True}, {}
        else:
            score, timing, audit = raw_model(recipe)
        predicted, used = rank_with_fallback(score, candidates, fallback)
        scores_by_name[name], predictions[name] = score, predicted
        wide[name] = ranked_lists(score, candidates, 50)
        value = evaluate(predicted, users, truth, activity, candidates, timing, history["path"], used)
        value.update({"recipe": recipe, "versus_v3": paired_summary(predicted, predictions["v3_control"], users, truth), "audit": audit})
        results[name] = value
        print(name, f"MAP={value['strict_protocol']['submission_all']['map_at_5']:.9f}", flush=True)

    record("v3_control", V3)
    expected = .524651455026455 if bounds == MAIN else .5050469576719576
    if abs(results["v3_control"]["strict_protocol"]["submission_all"]["map_at_5"] - expected) > 1e-12:
        raise AssertionError("Не воспроизведён замороженный V3")
    if recipes is not None:
        for name, recipe in recipes.items():
            record(name, recipe)
    else:
        for number, config in enumerate(ALS_CONFIGS, 1):
            record(f"als50_{number}", {"family": "als", "config": config})
        record("content50", {"family": "content", "config": CONTENT_CONFIG})
        best_als = max(("als50_1", "als50_2"), key=lambda name: results[name]["strict_protocol"]["submission_all"]["map_at_5"])
        for name in (best_als, "content50"):
            diag[name] = diagnostics(scores_by_name[name], scores_by_name["v3_control"], users, candidates, truth, activity)
        useful = [name for name in (best_als, "content50") if diag[name]["useful_for_small_boost"]]
        for source in useful:
            for strength in ((.05, .10) if source == best_als else (.05,)):
                record(f"boost_{source}_{strength:.2f}", {"family": "agreement", "strength": strength,
                    "components": {source: results[source]["recipe"]}})
        if len(useful) == 2:
            # Дополнительная novelty CBF: более консервативный V3 Top50,
            # подмножество разрешённого native Top100; только общий агрегат.
            independent_hits = sum(len((set(wide["content50"][r][:20]) - set(wide[best_als][r][:20]) - set(predictions["v3_control"][r]))
                                       & set(wide["v3_control"][r]) & truth.get(user, set())) for r, user in enumerate(users))
            diag["content_novel_hits_beyond_als"] = independent_hits
            diag["joint_gate_v3_pool"] = "Top50, conservative subset of allowed native Top100"
            if independent_hits >= 63:
                record("boost_both_0.05", {"family": "agreement", "strength": .05,
                    "components": {source: results[source]["recipe"] for source in useful}})
        for name, recipe in (
            ("ppr_k100", {**V3, "neighbors": 100}), ("ppr_restart35", {**V3, "restart": .35}),
            ("ppr_k30", {**V3, "neighbors": 30}), ("ppr_restart25", {**V3, "restart": .25})):
            if len(results) - 1 >= MAX_CONFIGS:
                break
            record(name, recipe)
        promising = any(value["versus_v3"]["mean_map_delta"] >= DECISION["main_min_delta"]
                         and value["versus_v3"]["paired_bootstrap_95_interval"][0] > 0 for value in results.values())
        if not promising and len(results) - 1 < MAX_CONFIGS:
            record("item_knn", {"family": "knn", "normalisation": "cosine", "neighbors": 100, "history": "log1p"})
            diag["item_knn"] = diagnostics(scores_by_name["item_knn"], scores_by_name["v3_control"], users, candidates, truth, activity)
    return {"methods": results, "diagnostics": diag, "predictions": predictions, "wide": wide,
            "runtime_seconds": time.perf_counter() - started, "schedule_audit": schedule["audit"]}


def run():
    started = time.perf_counter()
    baseline = json.loads((ROOT / "results/popularity_baseline.json").read_text())
    users, candidates = baseline["protocol"]["primary_user_ids"], np.asarray(baseline["protocol"]["candidate_ids"], dtype=np.int64)
    main = evaluate_window(MAIN, users, candidates, ROOT / "data/processed/session_train_50.csv",
        ROOT / "data/processed/train_interactions.csv", ROOT / "data/processed/session_validation_80.csv")
    results = main["methods"]
    selected = max(results, key=lambda name: results[name]["strict_protocol"]["submission_all"]["map_at_5"])
    # Единственный победитель заморожен до ранней проверки; runner-up не ищется.
    recipe = results[selected]["recipe"]
    early_candidates = np.asarray(sorted(validation_candidates([ROOT / "data" / SCHEDULE_FILES[0]], *EARLY[1:])), dtype=np.int64)
    early = evaluate_window(EARLY, users, early_candidates, ROOT / "data/processed/early_train_50.csv",
        ROOT / "data/processed/early_train_60.csv", ROOT / "data/processed/early_validation_80.csv", {selected: recipe} if selected != "v3_control" else {})
    main_change, early_change = results[selected]["versus_v3"], early["methods"][selected]["versus_v3"]
    convincing = (main_change["mean_map_delta"] >= DECISION["main_min_delta"]
                  and main_change["paired_bootstrap_95_interval"][0] > 0
                  and early_change["mean_map_delta"] >= DECISION["early_min_delta"])
    write_predictions(ROOT / "results/ppr_complement_recommendations.csv", main.pop("predictions"), users)
    write_predictions(ROOT / "results/ppr_complement_early_recommendations.csv", early.pop("predictions"), users)
    with (ROOT / "results/ppr_complement_native_top50.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["method", "user_id", "native_top50"])
        for name, rows in main.pop("wide").items():
            writer.writerows([name, user, " ".join(map(str, items))] for user, items in zip(users, rows))
    early.pop("wide")
    write_metrics(results, early)
    dump(ROOT / "results/ppr_complement.json", {"main": main, "early": early, "selected": selected,
        "decision": DECISION, "signal_gate": SIGNAL_GATE, "convincing_for_v4": bool(convincing),
        "new_configurations": len(results) - 1, "implementation": signatures(),
        "reported_kaggle_v3": {"private": .39105, "source": "user-reported, never used for parameter selection"},
        "protocol": {"train": MAIN[:2], "validation": MAIN[1:], "users": 1260, "candidates": 6384, "ground_truth": "unchanged single-session >80%", "fallback": "unchanged unique-user popularity from 60s train", "seed": 42},
        "limitations": ["Repeated holdout selection; early temporal check is also previously used and is not independent.",
                        "Top50 novelty is diagnostic, not an oracle; no per-user validation-dependent features.",
                        "Absolute timezone/devices unknown; viewing proxy may differ from Kaggle."],
        "runtime": {"wall_seconds": time.perf_counter() - started, "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2 if sys.platform == "darwin" else 1024)}})
    print("Selected", selected, "V4 eligible", convincing, "main delta", main_change["mean_map_delta"], "early delta", early_change["mean_map_delta"], flush=True)


def final():
    result = json.loads((ROOT / "results/ppr_complement.json").read_text())
    if not result["convincing_for_v4"]:
        print("Убедительного прироста нет; V4 не создаётся.")
        return
    if result["implementation"] != signatures() or result["decision"] != DECISION:
        raise ValueError("Реализация изменилась после эксперимента")
    started = time.perf_counter()
    previous = {name: file_sha256(ROOT / name) for name in ("submission.csv", "submission_v2.csv", "submission_v3.csv")}
    baseline = json.loads((ROOT / "results/popularity_baseline.json").read_text())
    users = baseline["protocol"]["primary_user_ids"]
    start, cutoff, end = MAIN[0], MAIN[2], "2020-10-19 00:00:00"
    candidates = np.asarray(sorted(validation_candidates([ROOT / "data" / name for name in SCHEDULE_FILES], cutoff, end)), dtype=np.int64)
    history = history_for(ROOT / "data/processed/session_final_train_50.csv", start, cutoff, "final_train")
    pop, _ = fit_popularity(ROOT / "data/processed/final_interactions.csv", expected_period="final_train")
    fallback = top_popular(pop["unique_users"], candidates)
    recipe = result["main"]["methods"][result["selected"]]["recipe"]
    schedule = schedule_context([ROOT / "data" / name for name in SCHEDULE_FILES], cutoff, end, candidates)
    availability = schedule_affinity({}, schedule, users, candidates, "availability")
    def auxiliary(configuration):
        metadata = None
        if configuration["family"] == "content":
            metadata, _ = load_train_metadata([ROOT / "data" / name for name in SCHEDULE_FILES[:2]], start, cutoff)
        return model_scores(configuration["family"], configuration["config"], history, users, candidates, metadata)

    if recipe["family"] == "agreement":
        score, timing, audit = ppr_v3_scores(history, users, candidates, availability, V3, {})
        feature = np.zeros_like(score)
        component_times = {}
        for name, config in recipe["components"].items():
            extra, component_times[name], _ = auxiliary(config)
            feature += native_agreement(extra, candidates)
        score = bounded_rerank(score, feature / len(recipe["components"]), candidates, strength=recipe["strength"], limit=100)
        timing["components"] = component_times
    elif recipe["family"] == "ppr":
        score, timing, audit = ppr_v3_scores(history, users, candidates, availability, recipe, {})
    elif recipe["family"] == "knn":
        score, timing, audit = knn_scores(history, users, candidates)
    else:
        score, timing, audit = auxiliary(recipe)
    predicted, used = rank_with_fallback(score, candidates, fallback)
    checks = write_personal_submission(ROOT / "data/submission.csv", ROOT / "submission_v4.csv", dict(zip(users, predicted)), candidates)
    if any(file_sha256(ROOT / name) != digest for name, digest in previous.items()):
        raise AssertionError("Изменена предыдущая версия submission")
    dump(ROOT / "results/submission_v4.json", {"selected": result["selected"], "recipe": recipe,
        "train": [start, cutoff], "test": [cutoff, end], "history_pairs": history["counts"].nnz, "history_users": len(history["users"]), "history_items": len(history["items"]),
        "candidate_count": len(candidates), "fallback_users": int(sum(used)), "unique_items": len({i for row in predicted for i in row}),
        "submission": {"path": "submission_v4.csv", **checks}, "timing": timing, "audit": audit,
        "runtime_seconds": time.perf_counter() - started, "previous_submissions_unchanged": True, "kaggle_submitted": False})
    print("Готов submission_v4.csv", checks, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--final", action="store_true")
    arguments = parser.parse_args()
    with threadpool_limits(limits=1):
        final() if arguments.final else run()
