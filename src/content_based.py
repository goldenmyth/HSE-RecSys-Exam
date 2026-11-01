"""Независимый content-based профиль из доступных в train метаданных."""

import csv
import re
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import sparse

from scripts.prepare_interactions import wall_seconds


def _normalise(value):
    return " ".join(value.casefold().split())


def _people(value):
    return {_normalise(name) for name in re.split(r"[,;]|\s+/\s+", value) if name.strip()}


def load_train_metadata(schedule_paths, start_iso, end_iso):
    """Объединить признаки ID только для эфиров, пересекающих календарный train.

    Названия передач не используются. Год — известный начальный год диапазона;
    неизвестные годы и концы открытых диапазонов не восстанавливаются.
    """
    lower, upper = wall_seconds(start_iso), wall_seconds(end_iso)
    if lower >= upper:
        raise ValueError("Начало train должно предшествовать его окончанию")
    latest_year = int(end_iso[:4])
    metadata, counts = {}, Counter()
    for source in schedule_paths:
        with Path(source).open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            required = {
                "start_time", "duration", "tv_show_id", "tv_show_category",
                "tv_show_genre_1", "tv_show_genre_2", "tv_show_genre_3",
                "year_of_production", "director", "actors",
            }
            if not required.issubset(reader.fieldnames or []):
                raise ValueError("В расписании отсутствуют необходимые столбцы")
            for row in reader:
                counts["source_rows"] += 1
                item, duration = int(row["tv_show_id"]), int(row["duration"])
                if item <= 0 or duration <= 0:
                    counts["excluded_nonpositive_id_or_duration_rows"] += 1
                    continue
                start = wall_seconds(row["start_time"], schedule=True)
                if start >= upper or start + duration <= lower:
                    counts["excluded_outside_train_rows"] += 1
                    continue
                counts["included_train_airing_rows"] += 1
                attributes = metadata.setdefault(item, {"basic": set(), "extended": set()})
                category = _normalise(row["tv_show_category"])
                if category:
                    attributes["basic"].add("category:" + category)
                for column in ("tv_show_genre_1", "tv_show_genre_2", "tv_show_genre_3"):
                    genre = _normalise(row[column])
                    if genre:
                        attributes["basic"].add("genre:" + genre)
                attributes["extended"].update(attributes["basic"])
                for column, prefix in (("director", "director:"), ("actors", "actor:")):
                    attributes["extended"].update(prefix + name for name in _people(row[column]))
                raw_year = row["year_of_production"].strip()
                initial = re.match(r"^(\d{4})(?=\D|$)", raw_year)
                year = int(initial.group(1)) if initial else 0
                if 1800 <= year <= latest_year:
                    attributes["extended"].add(f"year_decade:{year // 10 * 10}")
                elif raw_year:
                    counts["included_rows_without_valid_initial_year"] += 1
    audit = {
        **dict(counts),
        "items": len(metadata),
        "calendar_train_start_inclusive": start_iso,
        "calendar_train_end_exclusive": end_iso,
        "metadata_scope": "positive-duration nonzero-ID airings intersecting calendar train only",
        "aggregation": "set union per programme ID; no frequency weight from schedule repetitions",
        "year_policy": f"observed initial year in [1800, {latest_year}]; decade only; no inferred end year",
        "people_delimiters": [",", ";", " / "],
        "normalisation": "casefold and collapsed whitespace; no transliteration or title matching",
        "title_used": False,
    }
    for feature_set in ("basic", "extended"):
        tokens = {token for attributes in metadata.values() for token in attributes[feature_set]}
        audit[feature_set + "_unique_tokens"] = len(tokens)
        audit[feature_set + "_item_token_pairs"] = sum(len(a[feature_set]) for a in metadata.values())
        audit[feature_set + "_items_without_features"] = sum(not a[feature_set] for a in metadata.values())
    return metadata, audit


def _l2_rows(matrix):
    squared = np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel()
    norms = np.sqrt(squared)
    scale = np.divide(1.0, norms, out=np.zeros_like(norms), where=norms > 0)
    return matrix.multiply(scale[:, None]).tocsr(), norms > 0


def fit_content_model(metadata, train_counts_csr, train_item_ids, feature_set="basic", history_weighting="binary"):
    """TF-IDF item-векторы и нормированные профили из train watch_count."""
    if feature_set not in ("basic", "extended"):
        raise ValueError("feature_set должен быть basic или extended")
    if history_weighting not in ("binary", "log1p"):
        raise ValueError("history_weighting должен быть binary или log1p")
    counts = sparse.csr_matrix(train_counts_csr, dtype=np.float32, copy=True)
    train_item_ids = [int(item) for item in train_item_ids]
    if counts.shape[1] != len(train_item_ids) or len(set(train_item_ids)) != len(train_item_ids):
        raise ValueError("Столбцам train_counts должны соответствовать уникальные train_item_ids")
    if not np.all(np.isfinite(counts.data)) or np.any(counts.data < 0):
        raise ValueError("watch_count должны быть конечными и неотрицательными")
    counts.sum_duplicates()
    counts.eliminate_zeros()
    counts.sort_indices()
    item_ids = np.asarray(sorted(metadata), dtype=np.int64)
    item_index = {int(item): position for position, item in enumerate(item_ids)}
    feature_names = sorted({token for item in item_ids for token in metadata[int(item)][feature_set]})
    vocabulary = {token: position for position, token in enumerate(feature_names)}
    indptr, indices, frequencies = [0], [], np.zeros(len(feature_names), dtype=np.int64)
    for item in item_ids:
        columns = sorted(vocabulary[token] for token in set(metadata[int(item)][feature_set]))
        indices.extend(columns)
        frequencies[columns] += 1
        indptr.append(len(indices))
    idf = np.log((1 + len(item_ids)) / (1 + frequencies)) + 1
    vectors = sparse.csr_matrix(
        (idf[indices].astype(np.float32), np.asarray(indices, dtype=np.int32), np.asarray(indptr, dtype=np.int32)),
        shape=(len(item_ids), len(feature_names)), dtype=np.float32,
    )
    vectors, item_nonempty = _l2_rows(vectors)
    matched_columns = [position for position, item in enumerate(train_item_ids) if item in item_index]
    alignment = sparse.csr_matrix(
        (np.ones(len(matched_columns), dtype=np.float32),
         (matched_columns, [item_index[train_item_ids[position]] for position in matched_columns])),
        shape=(len(train_item_ids), len(item_ids)), dtype=np.float32,
    )
    weights = counts.copy()
    if history_weighting == "binary":
        weights.data.fill(1)
    else:
        np.log1p(weights.data, out=weights.data)
    profiles, profile_nonempty = _l2_rows((weights @ alignment @ vectors).tocsr())
    return {
        "item_ids": item_ids, "item_index": item_index, "feature_names": feature_names,
        "vectors": vectors, "profiles": profiles,
        "item_nonempty": item_nonempty, "profile_nonempty": profile_nonempty,
        "audit": {
            "feature_set": feature_set, "history_weighting": history_weighting,
            "metadata_items": len(item_ids), "features": len(feature_names),
            "item_feature_nnz": int(vectors.nnz), "profile_feature_nnz": int(profiles.nnz),
            "train_users": counts.shape[0], "train_observed_items": counts.shape[1],
            "train_observed_items_without_metadata": counts.shape[1] - len(matched_columns),
            "items_without_features": int(np.count_nonzero(~item_nonempty)),
            "users_without_profile": int(np.count_nonzero(~profile_nonempty)),
            "tf": "binary presence per item, not schedule airing frequency",
            "idf": "log((1 + number_of_train_metadata_items)/(1 + item_document_frequency)) + 1",
            "normalisation": "L2 item vectors and L2 weighted-sum user profiles",
            "metadata_scope": "provided calendar-train metadata only; no future attributes",
        },
    }


def score_content(model, user_indices, candidate_ids):
    """Cosine-оценки; неизвестные пользователи/признаки и пустые профили — NaN."""
    user_indices = np.asarray(user_indices, dtype=np.int64)
    candidate_ids = [int(item) for item in candidate_ids]
    result = np.full((len(user_indices), len(candidate_ids)), np.nan, dtype=np.float32)
    candidate_positions, candidate_rows = [], []
    for position, item in enumerate(candidate_ids):
        row = model["item_index"].get(item)
        if row is not None and model["item_nonempty"][row]:
            candidate_positions.append(position)
            candidate_rows.append(row)
    if not candidate_positions:
        return result
    candidate_vectors = model["vectors"][candidate_rows].T.tocsr()
    for begin in range(0, len(user_indices), 64):
        positions = np.arange(begin, min(begin + 64, len(user_indices)))
        selected = user_indices[positions]
        valid = (selected >= 0) & (selected < model["profiles"].shape[0])
        positions, selected = positions[valid], selected[valid]
        nonempty = model["profile_nonempty"][selected]
        positions, selected = positions[nonempty], selected[nonempty]
        if len(selected):
            scores = (model["profiles"][selected] @ candidate_vectors).toarray()
            result[np.ix_(positions, candidate_positions)] = np.clip(scores, -1, 1)
    return result
