"""Переобучить выбранный strict-PPR и подготовить v2 без отправки в Kaggle."""

import csv
import json
import os
for _name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "1"
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluate_popularity import validation_candidates
from scripts.prepare_interactions import SCHEDULE_FILES, file_sha256
from scripts.prepare_sessions import prepare, protected_hashes
from scripts.strict_experiments import load_history, model_scores
from src.popularity import fit_popularity, top_popular
from src.ranking import rank_with_fallback


def write_personal_submission(template_path, output_path, predictions_by_user, candidates, expected_rows=1260):
    template_path, output_path = Path(template_path), Path(output_path)
    if template_path.resolve() == output_path.resolve():
        raise ValueError("Нельзя изменять исходный шаблон")
    with template_path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.reader(stream))
    if not rows or rows[0] != ["user_id", "tv_show_id"] or any(len(row) != 2 for row in rows[1:]):
        raise ValueError("Неверная схема шаблона")
    users = [row[0] for row in rows[1:]]
    if len(users) != expected_rows or len(set(users)) != expected_rows or any(not user for user in users) or set(users) != set(predictions_by_user):
        raise ValueError("Не совпали пользователи или число строк")
    pool, written = set(candidates), [rows[0]]
    for user in users:
        items = predictions_by_user[user]
        if len(items) != 5 or len(set(items)) != 5 or any(type(item) is not int or item <= 0 for item in items) or not set(items) <= pool:
            raise ValueError("Неверный Top-5")
        written.append([user, " ".join(map(str, items))])
    temporary = output_path.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream).writerows(written)
    with temporary.open(encoding="utf-8", newline="") as stream:
        if list(csv.reader(stream)) != written:
            raise AssertionError("Submission readback отличается от записанного")
    temporary.replace(output_path)
    return {"rows": len(users), "columns": rows[0], "users_and_order_match_template": True,
            "five_distinct_positive_candidates_per_row": True, "string_user_ids_preserved": True,
            "sha256": file_sha256(output_path), "bytes": output_path.stat().st_size}


def main():
    started = time.perf_counter()
    experiment_path = ROOT / "results/strict_experiments.json"
    experiment = json.loads(experiment_path.read_text())
    selected = experiment["selected_strategy"]
    if not experiment["paired_improvement"]["convincing_for_v2"]:
        print("Убедительного offline-прироста нет; submission_v2 не создаётся.")
        return
    method = experiment["methods"][selected]
    if method["family"] != "ppr":
        raise ValueError("Этот финальный скрипт соответствует выбранной PPR-стратегии; новая модель требует явного расширения")
    protected = protected_hashes()
    protected["results/strict_experiments.json"] = file_sha256(experiment_path)
    protected["results/strict_recommendations.csv"] = file_sha256(ROOT / "results/strict_recommendations.csv")
    manifest = prepare(final=True)
    preparation = json.loads((ROOT / "results/data_preparation.json").read_text())
    bounds = preparation["parameters"]["period_boundaries"]["test"]
    start, end = bounds["start_inclusive"], bounds["end_exclusive"]
    history = load_history(ROOT / manifest["outputs"]["final_train_50"]["path"],
        period="final_train", lower="2020-03-09 00:00:00", upper=start)
    candidates = np.asarray(sorted(validation_candidates([ROOT / "data" / name for name in SCHEDULE_FILES], start, end)), dtype=np.int64)
    if len(candidates) != preparation["candidate_programmes_by_calendar_period"]["test"]:
        raise AssertionError("Изменился test candidate pool")
    baseline = json.loads((ROOT / "results/popularity_baseline.json").read_text())
    users = baseline["protocol"]["primary_user_ids"]
    scores, _ = fit_popularity(ROOT / "data/processed/final_interactions.csv", expected_period="final_train")
    fallback = top_popular(scores["unique_users"], candidates)
    model_scores_matrix, timing, audit = model_scores("ppr", method["config"], history, users, candidates)
    predictions, used = rank_with_fallback(model_scores_matrix, candidates, fallback)
    output = ROOT / "submission_v2.csv"
    checks = write_personal_submission(ROOT / "data/submission.csv", output, dict(zip(users, predictions)), candidates)
    tests = subprocess.run([sys.executable, "-B", "-m", "unittest", "discover", "-s", str(ROOT / "tests"), "-v"], cwd=ROOT, capture_output=True, text=True)
    if tests.returncode:
        raise RuntimeError(tests.stdout + tests.stderr)
    for path, expected in protected.items():
        if file_sha256(ROOT / path) != expected:
            raise AssertionError("Изменился сохранённый артефакт: " + path)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result = {"selected_strategy": selected, "config": method["config"], "training_rule": manifest["policy"],
        "train_window": ["2020-03-09 00:00:00", start], "test_window": [start, end],
        "training_interactions": manifest["outputs"]["final_train_50"], "training_source": history["path"],
        "candidate_count": len(candidates), "fallback": {"method": "unique-user popularity, full 60-second final train", "top5": fallback,
            "users": int(sum(used)), "user_fraction": sum(used) / len(users)},
        "unique_recommended_programmes": len({item for row in predictions for item in row}),
        "validation_evidence": {"strict_map_at_5": method["strict_protocol"]["submission_all"]["map_at_5"],
                                "paired_improvement": experiment["paired_improvement"]},
        "model_audit": audit, "model_timing": timing, "submission": {"path": "submission_v2.csv", **checks},
        "checks": {"old_submission_and_all_old_results_unchanged": True, "no_test_viewing_used": True,
                   "whole_final_window_rebuilt_from_raw_sessions": True, "unit_tests_passed": True,
                   "unit_test_output": tests.stderr.strip()},
        "protected_sha256": protected,
        "runtime": {"wall_seconds": time.perf_counter() - started, "peak_rss_mib": peak / (1024**2 if sys.platform == "darwin" else 1024)},
        "kaggle_submitted": False, "kaggle_score_v2": None,
    }
    path = ROOT / "results/submission_v2.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
    print(json.dumps({"submission": result["submission"], "training": result["training_interactions"],
                      "unique_items": result["unique_recommended_programmes"], "fallback_users": int(sum(used)),
                      "runtime": result["runtime"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
