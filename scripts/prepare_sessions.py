"""Подготовить отдельную разметку single-session >50/>80%; старые файлы не менять."""

import argparse
import csv
import itertools
import json
import resource
import sys
import tempfile
import time
from array import array
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_interactions import BUCKETS, RECORD, PERIODS, build_channel_index, file_sha256, iso, read_views
from scripts.build_submission import FINAL_PERIODS
from src.session_interactions import match_sessions

LEGACY = (
    "submission.csv", "results/data_preparation.json", "results/popularity_baseline.json",
    "results/model_comparison.json", "results/model_recommendations.csv", "results/final_submission.json",
    "data/processed/train_interactions.csv", "data/processed/validation_interactions.csv",
    "data/processed/final_interactions.csv",
)
POLICY = {
    "id": "single-session-unambiguous-v1",
    "thresholds_percent": [50, 80], "comparison": "strictly greater, integer cross multiplication",
    "sitting": "one original log row after exact deduplication; no merging across original rows",
    "channel_ambiguity": "remove simultaneous different-channel time; each remaining continuous piece is independent",
    "denominator": "full canonical airing duration; period clipping never shortens denominator",
    "airing_deduplication": "one qualifying user/channel/airing irrespective of qualifying source-row count",
    "schedule_policy": "existing half-open channel/time index, zero IDs and conflicting airings excluded",
    "source_status": "participant description, not confirmed Kaggle judge code; devices and actual switches unobservable",
}


def protected_hashes():
    return {path: file_sha256(ROOT / path) for path in LEGACY}


def history_index():
    from scripts.prepare_interactions import wall_seconds
    records = defaultdict(lambda: (array("q"), array("q"), array("q")))
    for name in ("export_arh_11-20-final.csv", "export_arh_21-30-final.csv"):
        with (ROOT / "data" / name).open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                channel, start = int(row["channel_id"]), wall_seconds(row["start_time"], True)
                for target, value in zip(records[channel], (start, start + int(row["duration"]), int(row["tv_show_id"]))):
                    target.append(value)
    indices, audit = {}, Counter()
    for channel in list(records):
        values = records.pop(channel)
        indices[channel], result = build_channel_index(zip(*values))
        audit.update(result)
    return indices, dict(audit)


def prepare(final=False):
    started = time.perf_counter()
    result_path = ROOT / "results" / ("session_final_preparation.json" if final else "session_preparation.json")
    protected = protected_hashes()
    preparation = json.loads((ROOT / "results/data_preparation.json").read_text())
    inputs = {"data/" + name: info["sha256"] for name, info in preparation["inputs"].items()}
    for path, expected in inputs.items():
        if file_sha256(ROOT / path) != expected:
            raise ValueError("Изменился исходный CSV: " + path)
    implementation = file_sha256(ROOT / "src/session_interactions.py")
    if result_path.exists():
        cached = json.loads(result_path.read_text())
        if cached["policy"] == POLICY and cached["inputs_sha256"] == inputs and cached["implementation_sha256"] == implementation:
            if all((ROOT / info["path"]).is_file() and file_sha256(ROOT / info["path"]) == info["sha256"] for info in cached["outputs"].values()):
                print("Session preparation cache:", result_path.relative_to(ROOT), flush=True)
                return cached
    periods = FINAL_PERIODS if final else PERIODS
    specifications = [(name, threshold, False) for name, _, _ in periods for threshold in (50, 80) if name != "validation" or threshold == 80]
    if not final:
        specifications.append(("validation", 80, True))
    paths = {(name, threshold, sensitivity): ROOT / "data/processed" / f"session_{name}_{threshold}{'_row_only' if sensitivity else ''}.csv"
             for name, threshold, sensitivity in specifications}
    streams, writers = {}, {}
    summaries = {key: Counter() for key in paths}
    users_by_file, items_by_file = {key: set() for key in paths}, {key: set() for key in paths}
    diagnostics = {name: Counter() for name, _, _ in periods}
    indices, schedule_audit = history_index()
    try:
        for key, path in paths.items():
            streams[key] = path.with_suffix(".csv.tmp").open("w", encoding="utf-8", newline="")
            writers[key] = csv.writer(streams[key])
            writers[key].writerow(["user_id", "tv_show_id", "watch_count", "total_watch_seconds", "last_watch_time", "period"])
        with tempfile.TemporaryDirectory(prefix=".sessions-", dir=ROOT / "data/processed") as temporary:
            work = Path(temporary)
            view_stats, users, *_ = read_views(work)
            for bucket in range(BUCKETS):
                rows = sorted(RECORD.iter_unpack((work / f"views_{bucket:02}.bin").read_bytes()))
                for number, group in itertools.groupby(rows, key=lambda row: row[0]):
                    views = defaultdict(list)
                    for _, channel, start, end, _ in group:
                        views[channel].append((start, end))
                    interactions, audit = match_sessions(views, indices, periods, (50, 80), True)
                    for name in diagnostics:
                        diagnostics[name].update(audit[name])
                    row_only = None
                    if not final:
                        row_only, _ = match_sessions(views, indices, [PERIODS[1]], (80,), False)
                    for key in paths:
                        name, threshold, sensitivity = key
                        values = (row_only if sensitivity else interactions)[name][threshold]
                        if values:
                            users_by_file[key].add(users[number])
                        for item, (count, seconds, last) in sorted(values.items()):
                            writers[key].writerow([users[number], item, count, seconds, iso(last), name])
                            summaries[key]["user_programme_pairs"] += 1
                            summaries[key]["watch_count"] += count
                            summaries[key]["total_watch_seconds"] += seconds
                            items_by_file[key].add(item)
                if (bucket + 1) % 8 == 0:
                    print(f"Session blocks {bucket + 1}/{BUCKETS}", flush=True)
        for stream in streams.values():
            stream.close()
        for path in paths.values():
            path.with_suffix(".csv.tmp").replace(path)
    finally:
        for stream in streams.values():
            stream.close()
    for path, expected in {**inputs, **protected}.items():
        if file_sha256(ROOT / path) != expected:
            raise AssertionError("Изменился защищённый файл: " + path)
    outputs = {}
    for key, path in paths.items():
        name, threshold, sensitivity = key
        outputs[f"{name}_{threshold}{'_row_only' if sensitivity else ''}"] = {
            "path": str(path.relative_to(ROOT)), **dict(summaries[key]), "users": len(users_by_file[key]),
            "programmes": len(items_by_file[key]), "sha256": file_sha256(path), "bytes": path.stat().st_size,
        }
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result = {"policy": POLICY, "periods": {name: [iso(lower), iso(upper)] for name, lower, upper in periods},
        "inputs_sha256": inputs, "protected_sha256": protected, "implementation_sha256": implementation,
        "outputs": outputs, "session_diagnostics": {name: dict(value) for name, value in diagnostics.items()},
        "schedule_audit": schedule_audit, "source_view_rows": view_stats["rows"],
        "runtime": {"wall_seconds": time.perf_counter() - started, "peak_rss_mib": peak / (1024**2 if sys.platform == "darwin" else 1024)}}
    temporary = result_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(result_path)
    print(json.dumps(outputs, ensure_ascii=False, indent=2), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final", action="store_true", help="Единое final-train окно; не строить validation")
    prepare(parser.parse_args().final)
