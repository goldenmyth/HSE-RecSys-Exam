"""Общий числовой tie-break, popularity fallback и fusion рангов."""

import numpy as np


def rank_with_fallback(scores, candidate_ids, fallback, k=5):
    ids = np.asarray(candidate_ids, dtype=np.int64)
    scores = np.asarray(scores)
    if k <= 0 or len(ids) < k or len(set(ids)) != len(ids) or np.any(ids <= 0):
        raise ValueError("Нужны различные положительные кандидаты и допустимый k")
    if scores.ndim != 2 or scores.shape[1] != len(ids):
        raise ValueError("Размер scores не соответствует кандидатам")
    pool = set(ids)
    if len(set(fallback)) != len(fallback) or len(fallback) < k or not set(fallback) <= pool:
        raise ValueError("Fallback должен содержать k различных допустимых ID")
    predictions, used_fallback = [], []
    for row in scores:
        known = np.flatnonzero(np.isfinite(row))
        order = np.lexsort((ids[known], -row[known]))[:k]
        selected = [int(item) for item in ids[known[order]]]
        used_fallback.append(len(selected) < k)
        for item in fallback:
            if len(selected) == k:
                break
            if item not in selected:
                selected.append(int(item))
        predictions.append(selected)
    return predictions, used_fallback


def reciprocal_rank_fusion(personal_scores, candidate_ids, popularity, personal_weight, offset=10):
    """Взвешенная сумма 1/(offset+rank), без смешивания исходных scores.

    У неизвестного модели item персональная составляющая равна нулю.
    Для полностью неизвестного пользователя остаётся popularity-ранжирование.
    """
    if not 0 <= personal_weight <= 1 or offset <= 0:
        raise ValueError("Некорректные параметры fusion")
    ids = np.asarray(candidate_ids, dtype=np.int64)
    pop_order = np.asarray(sorted(range(len(ids)), key=lambda pos: (-popularity.get(int(ids[pos]), 0), int(ids[pos]))))
    pop = np.empty(len(ids), dtype=np.float32)
    pop[pop_order] = 1 / (offset + np.arange(1, len(ids) + 1))
    fused = np.tile((1 - personal_weight) * pop, (len(personal_scores), 1))
    for row_number, row in enumerate(personal_scores):
        known = np.flatnonzero(np.isfinite(row))
        if not len(known):
            fused[row_number] = pop
            continue
        order = known[np.lexsort((ids[known], -row[known]))]
        fused[row_number, order] += personal_weight / (offset + np.arange(1, len(order) + 1))
    return fused
