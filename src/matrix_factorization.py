"""Implicit-feedback ALS: confidence из counts, затем сырые скалярные scores."""

import numpy as np
from implicit.als import AlternatingLeastSquares
from scipy.sparse import csr_matrix, issparse


def confidence_matrix(counts_csr, weighting="binary", alpha=10):
    """Ненулевые counts задают p_ui=1 и c_ui=1+alpha*f(count).

    Структурные нули остаются отсутствующими: implicit использует для них
    p_ui=0, c_ui=1. Эти значения не являются явными рейтингами.
    """
    if not issparse(counts_csr):
        raise TypeError("counts должна быть разреженной матрицей")
    if weighting not in ("binary", "log_count"):
        raise ValueError("Неизвестное взвешивание confidence")
    if not np.isfinite(alpha) or alpha < 0:
        raise ValueError("confidence alpha должна быть конечной и неотрицательной")
    if not np.all(np.isfinite(counts_csr.data)) or np.any(counts_csr.data <= 0):
        raise ValueError("Сохранённые counts должны быть конечными и положительными")

    confidence = csr_matrix(counts_csr, dtype=np.float32, copy=True)
    confidence.sum_duplicates()
    confidence.sort_indices()
    weights = (
        np.ones_like(confidence.data)
        if weighting == "binary"
        else np.log1p(confidence.data)
    )
    confidence.data = np.asarray(1 + alpha * weights, dtype=np.float32)
    if not np.all(np.isfinite(confidence.data)):
        raise ValueError("Confidence не помещается в float32")
    return confidence


def fit_als(counts_csr, config, seed=42):
    """Обучить CPU ALS на CSR с ориентацией users × items."""
    confidence = confidence_matrix(
        counts_csr,
        weighting=config["weighting"],
        alpha=config["confidence_alpha"],
    )
    model = AlternatingLeastSquares(
        factors=config["factors"],
        regularization=config["regularization"],
        alpha=1.0,
        dtype=np.float32,
        use_gpu=False,
        iterations=config.get("iterations", 20),
        num_threads=1,
        random_state=seed,
    )
    # Confidence уже включает alpha; повторное масштабирование библиотекой
    # изменило бы согласованное c_ui, поэтому model.alpha всегда равна 1.
    model.fit(confidence, show_progress=False)
    return model


def score_als(model, user_indices, item_indices):
    """Скалярные произведения факторов; неизвестный индекс -1 даёт NaN."""
    users = np.asarray(user_indices, dtype=np.int64)
    items = np.asarray(item_indices, dtype=np.int64)
    if users.ndim != 1 or items.ndim != 1:
        raise ValueError("Индексы пользователей и программ должны быть одномерными")
    if np.any(users < -1) or np.any(items < -1):
        raise ValueError("Только -1 обозначает неизвестный индекс")
    scores = np.full((len(users), len(items)), np.nan, dtype=np.float32)
    known_users = np.flatnonzero(users >= 0)
    known_items = np.flatnonzero(items >= 0)
    if len(known_users) and len(known_items):
        scores[np.ix_(known_users, known_items)] = (
            model.user_factors[users[known_users]]
            @ model.item_factors[items[known_items]].T
        )
    return scores
