"""Собрать квалифицировавшиеся эфиры; development делает main+early за один проход."""

import argparse
import csv
import itertools
import json
import sys
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_interactions import BUCKETS, RECORD, file_sha256, iso, match_user, read_views, wall_seconds
from scripts.prepare_sessions import history_index
from src.session_interactions import match_sessions

DATA = ROOT / "data/processed"
AGGREGATE_FIELDS = ["user_id", "tv_show_id", "watch_count", "total_watch_seconds", "last_watch_time", "period"]
AIRING_FIELDS = ["user_id", "tv_show_id", "channel_id", "airing_start", "airing_end", "watch_start", "watch_end", "period"]
WINDOWS = {
    "main": {"train": ["2020-03-09 00:00:00", "2020-05-18 00:00:00"],
             "validation": ["2020-05-18 00:00:00", "2020-07-27 00:00:00"]},
    "early": {"train": ["2020-03-09 00:00:00", "2020-04-13 00:00:00"],
              "validation": ["2020-04-13 00:00:00", "2020-05-18 00:00:00"]},
    "final": {"final_train": ["2020-03-09 00:00:00", "2020-07-27 00:00:00"]},
}


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _references(mode):
    if mode == "early":
        return {}
    if mode == "final":
        manifest = ROOT / "results/session_final_preparation.json"
        return {"train_50": _read_json(manifest)["outputs"]["final_train_50"]} if manifest.is_file() else {}
    sessions = _read_json(ROOT / "results/session_preparation.json")["outputs"]
    original = _read_json(ROOT / "results/data_preparation.json")
    train = original["interactions"]["train"]
    return {
        "train_50": sessions["train_50"], "validation_80": sessions["validation_80"],
        "train_60": {"path": "data/processed/train_interactions.csv",
                     **original["checks"]["output_validation"]["train"],
                     "user_programme_pairs": train["user_programme_pairs"],
                     "watch_count": train["watch_count"], "total_watch_seconds": train["total_watch_seconds"]},
    }


def _signature(mode, references):
    paths = [ROOT / "data" / name for name in (
        "dataset11-30.csv", "export_arh_11-20-final.csv", "export_arh_21-30-final.csv",
    )]
    paths.extend(ROOT / record["path"] for record in references.values())
    return {str(path.relative_to(ROOT)): {"bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
            for path in paths if path.is_file()}


def _cached(mode, signature, implementation):
    path = ROOT / f"results/history_context_{mode}.json"
    if not path.is_file():
        return None
    record = _read_json(path)
    if record.get("source_signature") != signature or record.get("implementation_sha256") != implementation:
        return None
    for key in record["generated_outputs"]:
        output = record["outputs"][key]
        target = ROOT / output["path"]
        if not target.is_file() or file_sha256(target) != output["sha256"]:
            return None
    return record


def prepare(mode="main"):
    """Single mode -> manifest; development -> {main: manifest, early: manifest}."""
    if mode not in (*WINDOWS, "development"):
        raise ValueError("Допустимые mode: main, early, final, development")
    started = time.perf_counter()
    v2_path = ROOT / "submission_v2.csv"
    v2_before = file_sha256(v2_path)
    requested = ("main", "early") if mode == "development" else (mode,)
    implementation = {name: file_sha256(ROOT / name) for name in (
        "scripts/prepare_history_context.py", "scripts/prepare_interactions.py",
        "scripts/prepare_sessions.py", "src/session_interactions.py",
    )}
    references = {name: _references(name) for name in requested}
    signatures = {name: _signature(name, references[name]) for name in requested}
    completed = {name: _cached(name, signatures[name], implementation) for name in requested}
    pending = [name for name in requested if completed[name] is None]
    if pending:
        DATA.mkdir(exist_ok=True)
        specifications, period_lookup, thresholds = [], {}, {50}
        for name in pending:
            labels = ["final_train"] if name == "final" else ["train", "validation"] if name == "early" else ["train"]
            for label in labels:
                internal = f"{name}_{label}"
                bounds = WINDOWS[name][label]
                specifications.append((internal, *map(wall_seconds, bounds)))
                period_lookup[internal] = (name, label, 80 if label == "validation" else 50)
                if label == "validation":
                    thresholds.add(80)
        indices, index_audit = history_index()
        streams, writers, paths, summaries = {}, {}, {}, {}
        source_stats = None
        diagnostics = {internal: Counter() for internal, _, _ in specifications}
        published = False
        try:
            for name in pending:
                generated = {"train_airings": DATA / f"history_{name}_airings.csv"}
                if name == "early":
                    generated.update({key: DATA / f"early_{key}.csv" for key in ("train_50", "train_60", "validation_80")})
                elif "train_50" not in references[name] or not (ROOT / references[name]["train_50"]["path"]).is_file():
                    generated["train_50"] = DATA / f"history_{name}_train_50.csv"
                    references[name].pop("train_50", None)
                for key, path in generated.items():
                    identity = name, key
                    paths[identity] = path
                    streams[identity] = path.with_suffix(".csv.tmp").open("w", encoding="utf-8", newline="")
                    writers[identity] = csv.writer(streams[identity])
                    writers[identity].writerow(AIRING_FIELDS if key == "train_airings" else AGGREGATE_FIELDS)
                summaries[name] = defaultdict(Counter)
            with tempfile.TemporaryDirectory(prefix=".history-context-", dir=DATA) as temporary:
                work = Path(temporary)
                source_stats, users, *_ = read_views(work)
                early_period = [("early_train_60", *map(wall_seconds, WINDOWS["early"]["train"]))]
                for bucket in range(BUCKETS):
                    rows = sorted(RECORD.iter_unpack((work / f"views_{bucket:02}.bin").read_bytes()))
                    for number, group in itertools.groupby(rows, key=lambda row: row[0]):
                        views = defaultdict(list)
                        for _, channel, start, end, _ in group:
                            views[channel].append((start, end))
                        events = []
                        interactions, audit = match_sessions(views, indices, specifications, tuple(sorted(thresholds)), True, qualified_events=events)
                        user = users[number]
                        for internal, values in interactions.items():
                            name, label, percent = period_lookup[internal]
                            diagnostics[internal].update(audit[internal])
                            key = "validation_80" if label == "validation" else "train_50"
                            stats = summaries[name][key]
                            for item, (count, seconds, last) in sorted(values[percent].items()):
                                stats["user_programme_pairs"] += 1
                                stats["watch_count"] += count
                                stats["total_watch_seconds"] += seconds
                                if (name, key) in writers:
                                    writers[name, key].writerow([user, item, count, seconds, iso(last), label])
                        for internal, percent, item, channel, start, end, watch_start, watch_end in sorted(events):
                            name, label, _ = period_lookup[internal]
                            if percent == 50 and label != "validation":
                                writers[name, "train_airings"].writerow([
                                    user, item, channel, iso(start), iso(end), iso(watch_start), iso(watch_end), label,
                                ])
                                summaries[name]["train_airings"]["rows"] += 1
                        if "early" in pending:
                            old, _ = match_user(views, indices, periods=early_period, threshold=60)
                            for item, (count, seconds, last) in sorted(old["early_train_60"].items()):
                                writers["early", "train_60"].writerow([user, item, count, seconds, iso(last), "train"])
                                stats = summaries["early"]["train_60"]
                                stats["user_programme_pairs"] += 1
                                stats["watch_count"] += count
                                stats["total_watch_seconds"] += seconds
                    del rows
                    if (bucket + 1) % 8 == 0:
                        print(f"History context blocks {bucket + 1}/{BUCKETS}", flush=True)
            for stream in streams.values():
                stream.close()
            for name in pending:
                for key, stats in summaries[name].items():
                    if key in references[name] and key == "train_50":
                        for field in ("user_programme_pairs", "watch_count", "total_watch_seconds"):
                            if stats[field] != references[name][key][field]:
                                raise AssertionError(f"Изменились стандартные агрегаты {name}/{key}/{field}")
                train_label = "final_train" if name == "final" else "train"
                if summaries[name]["train_airings"]["rows"] != diagnostics[f"{name}_{train_label}"]["qualified_user_airings_50"]:
                    raise AssertionError("Число эфирных событий не совпало с qualification")
            for path in paths.values():
                path.with_suffix(".csv.tmp").replace(path)
            published = True
            for name in pending:
                outputs = dict(references[name])
                generated_keys = []
                for (owner, key), path in paths.items():
                    if owner == name:
                        outputs[key] = {"path": str(path.relative_to(ROOT)), **dict(summaries[name][key]),
                                        "bytes": path.stat().st_size, "sha256": file_sha256(path)}
                        generated_keys.append(key)
                record = {
                    "mode": name, "periods": WINDOWS[name], "outputs": outputs, "generated_outputs": generated_keys,
                    "source_signature": _signature(name, references[name]), "implementation_sha256": implementation,
                    "source_view_rows": source_stats["rows"], "history_index": index_audit,
                    "session_diagnostics": {label: dict(diagnostics[f"{name}_{label}"]) for label in WINDOWS[name] if f"{name}_{label}" in diagnostics},
                    "event_policy": "one >50% qualified user/channel/canonical-airing event; full airing bounds and first/last qualified watch endpoints",
                    "earlier_window_limitation": "earlier temporal robustness check, not an untouched confirmatory test" if name == "early" else None,
                    "runtime_seconds_shared_pass": time.perf_counter() - started,
                }
                target = ROOT / f"results/history_context_{name}.json"
                temporary_manifest = target.with_suffix(".json.tmp")
                temporary_manifest.write_text(json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                temporary_manifest.replace(target)
                completed[name] = record
        finally:
            for stream in streams.values():
                stream.close()
            if not published:
                for path in paths.values():
                    path.with_suffix(".csv.tmp").unlink(missing_ok=True)
    if file_sha256(v2_path) != v2_before:
        raise AssertionError("Изменился submission_v2.csv")
    return completed if mode == "development" else completed[mode]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=(*WINDOWS, "development"), default="main")
    result = prepare(parser.parse_args().mode)
    print(json.dumps(result, ensure_ascii=False, indent=2))
