"""Персональный PageRank по train co-view графу; NumPy и SciPy sparse."""

import time

import numpy as np
from scipy import sparse


def _counts_matrix(counts):
    if not sparse.issparse(counts):
        raise TypeError("counts должна быть разреженной матрицей users × items")
    result = sparse.csr_matrix(counts, dtype=np.float64, copy=True)
    if not np.all(np.isfinite(result.data)) or np.any(result.data < 0):
        raise ValueError("counts должны быть конечными и неотрицательными")
    result.sum_duplicates()
    result.eliminate_zeros()
    result.sort_indices()
    return result


def build_graph(counts_csr, item_ids, normalisation="user_cosine", neighbors=50, block_size=64):
    """Направленный top-K co-view граф без полной плотной item × item матрицы.

    В user_cosine каждый binary user-row сначала делится на sqrt(его степени).
    Затем в обоих вариантах берётся cosine столбцов. После top-K отсечения
    граф в общем случае несимметричен; дополнительные обратные рёбра не добавляются.
    """
    started = time.perf_counter()
    if normalisation not in ("user_cosine", "cosine"):
        raise ValueError("normalisation должна быть user_cosine или cosine")
    if not isinstance(neighbors, int) or neighbors <= 0:
        raise ValueError("neighbors должен быть положительным целым")
    if not isinstance(block_size, int) or not 1 <= block_size <= 64:
        raise ValueError("block_size должен быть целым от 1 до 64")
    counts = _counts_matrix(counts_csr)
    ids = np.asarray(item_ids, dtype=np.int64)
    if ids.ndim != 1 or len(ids) != counts.shape[1] or len(set(ids)) != len(ids) or np.any(ids <= 0):
        raise ValueError("Столбцам counts должны соответствовать уникальные положительные item_ids")
    binary = counts.copy()
    binary.data.fill(1)
    if normalisation == "user_cosine":
        degree = np.diff(binary.indptr).astype(np.float64)
        scale = np.divide(1.0, np.sqrt(degree), out=np.zeros_like(degree), where=degree > 0)
        values = binary.multiply(scale[:, None]).tocsr()
    else:
        values = binary
    norms = np.sqrt(np.asarray(values.multiply(values).sum(axis=0)).ravel())
    inverse = np.divide(1.0, norms, out=np.zeros_like(norms), where=norms > 0)
    values = values.multiply(inverse[None, :]).tocsc()
    data, indices, indptr = [], [], [0]
    max_block_nnz = max_block_rows = 0
    for begin in range(0, len(ids), block_size):
        end = min(begin + block_size, len(ids))
        block = (values[:, begin:end].T @ values).tocsr()
        max_block_nnz = max(max_block_nnz, block.nnz)
        max_block_rows = max(max_block_rows, end - begin)
        for local_row in range(end - begin):
            lower, upper = block.indptr[local_row:local_row + 2]
            columns, similarities = block.indices[lower:upper], block.data[lower:upper]
            keep = (columns != begin + local_row) & (similarities > 0)
            columns, similarities = columns[keep], np.minimum(similarities[keep], 1.0)
            order = np.lexsort((ids[columns], -similarities))[:neighbors]
            indices.extend(columns[order])
            data.extend(similarities[order])
            indptr.append(len(indices))
    transition = sparse.csr_matrix(
        (np.asarray(data, dtype=np.float64), np.asarray(indices, dtype=np.int32), np.asarray(indptr, dtype=np.int32)),
        shape=(len(ids), len(ids)),
    )
    sums = np.asarray(transition.sum(axis=1)).ravel()
    scale = np.divide(1.0, sums, out=np.zeros_like(sums), where=sums > 0)
    transition = transition.multiply(scale[:, None]).tocsr()
    transition.sort_indices()
    degree = np.diff(transition.indptr)
    final_sums = np.asarray(transition.sum(axis=1)).ravel()
    max_error = float(np.max(np.abs(final_sums[sums > 0] - 1))) if np.any(sums > 0) else 0.0
    return transition, {
        "normalisation": normalisation,
        "history_signal": "binary train interactions; raw counts do not weight graph edges",
        "user_normalisation": "divide each binary user row by sqrt(user degree)" if normalisation == "user_cosine" else "none",
        "edge_policy": "directed per-row top-K column cosine, no self loops and no symmetric closure",
        "edge_tie_break": "similarity descending, numeric item ID ascending",
        "users": int(counts.shape[0]), "items": int(len(ids)), "train_user_item_pairs": int(binary.nnz),
        "neighbors": neighbors, "block_size": block_size,
        "edges": int(transition.nnz), "dangling_items": int(np.count_nonzero(degree == 0)),
        "max_out_degree": int(degree.max()) if len(degree) else 0,
        "max_similarity_block_shape": [max_block_rows, int(len(ids))],
        "max_similarity_block_nnz": int(max_block_nnz),
        "full_dense_item_matrix_created": False,
        "max_nonempty_row_sum_error": max_error,
        "elapsed_seconds": time.perf_counter() - started,
    }


def personalized_scores(transition, counts_csr, user_indices, item_indices, restart=0.15,
                        weighting="binary", tol=1e-8, max_iter=200, batch_size=32,
                        restart_weights=None):
    """Персональные вероятности после диффузии по всему train-графу.

    x_next = restart*s + (1-restart)*(x@P + sum(x[dangling])*s).
    У seed и x сумма 1; кандидатный фильтр применяется только к готовому x.
    Неизвестные индексы и пользователи без собственного seed возвращают NaN.
    Внешние restart_weights задают веса только собственных train items;
    к ним применяется лишь нормировка суммы, без binary/log-преобразования.
    """
    started = time.perf_counter()
    if not np.isfinite(restart) or not 0 < restart <= 1:
        raise ValueError("Вероятность restart должна быть в (0, 1]")
    if weighting not in ("binary", "log_count"):
        raise ValueError("weighting должна быть binary или log_count")
    if not np.isfinite(tol) or tol <= 0:
        raise ValueError("tol должна быть конечной и положительной")
    if not isinstance(max_iter, int) or max_iter <= 0 or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("max_iter и batch_size должны быть положительными целыми")
    counts = _counts_matrix(counts_csr)
    seed_weights = counts
    if restart_weights is not None:
        if not sparse.issparse(restart_weights):
            raise TypeError("restart_weights должна быть разреженной матрицей users × items")
        seed_weights = _counts_matrix(restart_weights)
        if seed_weights.shape != counts.shape:
            raise ValueError("Размер restart_weights должен совпадать с counts")
        if not np.all(np.isfinite(seed_weights.data)):
            raise ValueError("Суммарные restart_weights должны быть конечными")
        if seed_weights.multiply(counts > 0).nnz != seed_weights.nnz:
            raise ValueError("restart_weights может содержать только собственные положительные train items")
    if not sparse.issparse(transition):
        raise TypeError("transition должна быть разреженной матрицей")
    matrix = sparse.csr_matrix(transition, dtype=np.float64, copy=True)
    if matrix.shape != (counts.shape[1], counts.shape[1]):
        raise ValueError("Размер графа должен совпадать с числом item-столбцов counts")
    if not np.all(np.isfinite(matrix.data)) or np.any(matrix.data < 0):
        raise ValueError("Веса переходов должны быть конечными и неотрицательными")
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    row_sums = np.asarray(matrix.sum(axis=1)).ravel()
    nonempty = row_sums > 0
    if np.any(np.abs(row_sums[nonempty] - 1) > 1e-7):
        raise ValueError("Непустые строки transition должны иметь сумму 1")
    # Remove only rounding drift, including when a valid float32 P was supplied.
    inverse = np.divide(1.0, row_sums, out=np.zeros_like(row_sums), where=nonempty)
    matrix = matrix.multiply(inverse[:, None]).tocsr()
    transpose = matrix.T.tocsr()
    users, items = np.asarray(user_indices, dtype=np.int64), np.asarray(item_indices, dtype=np.int64)
    if users.ndim != 1 or items.ndim != 1 or np.any(users < -1) or np.any(items < -1):
        raise ValueError("Нужны одномерные индексы; -1 обозначает неизвестный индекс")
    scores = np.full((len(users), len(items)), np.nan, dtype=np.float64)
    known_items = np.flatnonzero((items >= 0) & (items < matrix.shape[0]))
    known_users = (users >= 0) & (users < counts.shape[0])
    iterations, seeded_rows, unseeded_rows = [], 0, 0
    max_residual = max_mass_error = 0.0
    for begin in range(0, len(users), batch_size):
        positions = np.arange(begin, min(begin + batch_size, len(users)))
        positions = positions[known_users[positions]]
        if not len(positions):
            continue
        seeds = seed_weights[users[positions]].copy()
        if restart_weights is None:
            if weighting == "binary":
                seeds.data.fill(1)
            else:
                np.log1p(seeds.data, out=seeds.data)
        seeds = seeds.toarray()
        sums = seeds.sum(axis=1)
        if not np.all(np.isfinite(sums)):
            raise ValueError("Сумма restart weights должна быть конечной")
        has_seed = sums > 0
        unseeded_rows += int(np.count_nonzero(~has_seed))
        positions, seeds, sums = positions[has_seed], seeds[has_seed], sums[has_seed]
        if not len(positions):
            continue
        seeded_rows += len(positions)
        seeds /= sums[:, None]
        values = seeds.copy()
        residual = float("inf")
        for iteration in range(1, max_iter + 1):
            following = (transpose @ values.T).T
            following += values[:, ~nonempty].sum(axis=1)[:, None] * seeds
            following *= 1 - restart
            following += restart * seeds
            mass_error = float(np.max(np.abs(following.sum(axis=1) - 1)))
            if mass_error > 1e-9 or np.any(following < 0) or not np.all(np.isfinite(following)):
                raise RuntimeError("PPR потерял вероятностную массу или породил некорректные значения")
            max_mass_error = max(max_mass_error, mass_error)
            residual = float(np.max(np.abs(following - values).sum(axis=1)))
            values = following
            if residual <= tol:
                break
        else:
            raise RuntimeError(f"PPR не сошёлся за {max_iter} итераций: max L1 residual={residual:g}")
        iterations.append(iteration)
        max_residual = max(max_residual, residual)
        scores[np.ix_(positions, known_items)] = values[:, items[known_items]]
    audit = {
        "restart_probability": float(restart), "continuation_probability": float(1 - restart),
        "weighting": weighting, "seed_normalisation": "own train binary/log1p counts divided by row sum",
        "dangling_policy": "redistribute dangling mass to the same user's seed",
        "repeat_policy": "seed items remain eligible; no already-watched filter",
        "query_rows": int(len(users)), "seeded_user_rows": int(seeded_rows),
        "unknown_user_rows": int(np.count_nonzero(~known_users)),
        "unseeded_known_user_rows": int(unseeded_rows),
        "known_candidate_columns": int(len(known_items)), "unknown_candidate_columns": int(len(items) - len(known_items)),
        "dangling_items": int(np.count_nonzero(~nonempty)),
        "tol_l1_per_user": float(tol), "max_iter": max_iter, "batch_size": batch_size,
        "iterations_per_batch": iterations, "max_iterations_used": max(iterations, default=0),
        "max_l1_residual": max_residual, "max_probability_mass_error": max_mass_error,
        "all_seeded_rows_converged": True, "elapsed_seconds": time.perf_counter() - started,
    }
    if restart_weights is not None:
        audit["seed_source"] = "external_restart_weights"
        audit["seed_normalisation"] = "supplied nonnegative own train-item weights divided by row sum; no binary/log transform"
    return scores, audit
