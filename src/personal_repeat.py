"""Повторные просмотры: исходная частота и опциональное затухание давности."""

import numpy as np
from scipy.sparse import csr_matrix, issparse


def repeat_scores(
    counts_csr,
    last_watch_seconds_csr,
    user_indices,
    item_indices,
    cutoff_seconds,
    half_life_days=None,
):
    """Вернуть count или count × 2**(-age_days/half_life_days).

    Матрицы содержат уже агрегированные user/item пары. Неизвестные индексы
    (-1) и программы без истории пользователя получают NaN для общего fallback.
    """
    if not issparse(counts_csr) or not issparse(last_watch_seconds_csr):
        raise TypeError("Ожидались разреженные counts и last_watch_seconds")
    if not np.isfinite(cutoff_seconds):
        raise ValueError("Cutoff должен быть конечным")
    if half_life_days is not None and (
        not np.isfinite(half_life_days) or half_life_days <= 0
    ):
        raise ValueError("Half-life должен быть конечным и положительным")
    if not np.all(np.isfinite(counts_csr.data)) or np.any(counts_csr.data <= 0):
        raise ValueError("Counts должны быть конечными и положительными")
    if not np.all(np.isfinite(last_watch_seconds_csr.data)):
        raise ValueError("Время последнего просмотра должно быть конечным")
    if np.any(last_watch_seconds_csr.data > cutoff_seconds):
        raise ValueError("Время последнего просмотра позже cutoff")

    counts = csr_matrix(counts_csr, dtype=np.float64, copy=True)
    last_watch = csr_matrix(last_watch_seconds_csr, dtype=np.float64, copy=True)
    for matrix in (counts, last_watch):
        source_nnz = matrix.nnz
        matrix.sum_duplicates()
        if matrix.nnz != source_nnz:
            raise ValueError("Ожидалась одна агрегированная запись на user/item")
        matrix.sort_indices()
    if (
        counts.shape != last_watch.shape
        or not np.array_equal(counts.indptr, last_watch.indptr)
        or not np.array_equal(counts.indices, last_watch.indices)
    ):
        raise ValueError("Counts и last_watch_seconds должны описывать одинаковые пары")

    users = np.asarray(user_indices, dtype=np.int64)
    items = np.asarray(item_indices, dtype=np.int64)
    if users.ndim != 1 or items.ndim != 1:
        raise ValueError("Индексы должны быть одномерными")
    if np.any(users < -1) or np.any(users >= counts.shape[0]):
        raise ValueError("Некорректный индекс пользователя")
    if np.any(items < -1) or np.any(items >= counts.shape[1]):
        raise ValueError("Некорректный индекс программы")

    scores = np.full((len(users), len(items)), np.nan, dtype=np.float64)
    known_users, known_items = np.flatnonzero(users >= 0), np.flatnonzero(items >= 0)
    if len(known_users) and len(known_items):
        selected = counts[users[known_users]][:, items[known_items]].tocoo()
        values = selected.data.copy()
        if half_life_days is not None:
            selected_last = last_watch[users[known_users]][:, items[known_items]].tocoo()
            age_days = (cutoff_seconds - selected_last.data) / 86400
            values *= np.exp2(-age_days / half_life_days)
        scores[known_users[selected.row], known_items[selected.col]] = values
    return scores
