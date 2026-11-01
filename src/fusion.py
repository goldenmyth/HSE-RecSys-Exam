"""Ранговое объединение компонентов с единственным общим fallback."""

from collections import defaultdict
from math import fsum, isfinite

import numpy as np


def ranked_lists(scores, candidate_ids, limit=100):
    """Top-limit конечных scores без заполнения неизвестных popularity items."""
    ids = np.asarray(list(candidate_ids), dtype=np.int64)
    values = np.asarray(scores)
    if ids.ndim != 1 or len(set(ids)) != len(ids) or np.any(ids <= 0):
        raise ValueError("Нужны различные положительные ID кандидатов")
    if limit <= 0 or values.ndim != 2 or values.shape[1] != len(ids):
        raise ValueError("Некорректные limit или размеры scores")
    result = []
    for row in values:
        known = np.flatnonzero(np.isfinite(row))
        order = np.lexsort((ids[known], -row[known]))[:limit]
        result.append([int(item) for item in ids[known[order]]])
    return result


def native_agreement(native_scores, candidate_ids, top_n=20):
    """Бинарный [0,1] признак native Top-N; неизвестные items не дополняются.

    Любой конечный score, включая отрицательный ALS score, участвует в рангах.
    У пользователя без native scores строка признака полностью нулевая.
    """
    if not isinstance(top_n, (int, np.integer)) or top_n <= 0:
        raise ValueError("top_n должен быть положительным целым")
    ids = np.asarray(list(candidate_ids), dtype=np.int64)
    lists = ranked_lists(native_scores, ids, limit=int(top_n))
    feature = np.zeros(np.asarray(native_scores).shape, dtype=np.float64)
    columns = {int(item): column for column, item in enumerate(ids)}
    for row, items in enumerate(lists):
        feature[row, [columns[item] for item in items]] = 1
    return feature


def weighted_rrf(component_lists, weights, candidate_ids, fallback, offset=60, k=5):
    """Сумма weight/(offset+исходный rank); fallback применяется один раз."""
    candidates = [int(item) for item in candidate_ids]
    pool = set(candidates)
    if k <= 0 or len(pool) < k or len(pool) != len(candidates) or any(item <= 0 for item in pool):
        raise ValueError("Недостаточно различных положительных кандидатов")
    if not isfinite(offset) or offset <= 0:
        raise ValueError("Offset должен быть конечным и положительным")
    if not component_lists or set(component_lists) != set(weights):
        raise ValueError("Названия компонентов и weights должны совпадать")
    if any(not isfinite(weight) or weight < 0 for weight in weights.values()):
        raise ValueError("Weights должны быть конечными и неотрицательными")
    populations = {len(rows) for rows in component_lists.values()}
    if len(populations) != 1:
        raise ValueError("Компоненты должны описывать одну популяцию пользователей")
    fallback = [int(item) for item in fallback]
    if len(fallback) < k or len(set(fallback)) != len(fallback) or not set(fallback) <= pool:
        raise ValueError("Fallback должен содержать k различных допустимых ID")

    predictions, fallback_used = [], []
    for user in range(next(iter(populations))):
        contributions = defaultdict(list)
        for name, rows in component_lists.items():
            items = rows[user]
            if len(set(items)) != len(items) or not set(items) <= pool:
                raise ValueError("Повторяющийся или недопустимый ID в компоненте")
            if weights[name] == 0:
                continue
            for rank, item in enumerate(items, start=1):
                contributions[int(item)].append(weights[name] / (offset + rank))
        scores = {item: fsum(parts) for item, parts in contributions.items()}
        selected = sorted(scores, key=lambda item: (-scores[item], item))[:k]
        fallback_used.append(len(selected) < k)
        for item in fallback:
            if len(selected) == k:
                break
            if item not in selected:
                selected.append(item)
        predictions.append(selected)
    return predictions, fallback_used
