"""AP@K и MAP@K для явно заданной популяции пользователей."""

from itertools import islice
from math import fsum


def average_precision_at_k(relevant, recommendations, k=5):
    """Повтор рекомендации занимает позицию, но не даёт повторного попадания."""
    if k <= 0:
        raise ValueError("k должен быть положительным")
    relevant = set(relevant)
    if not relevant:
        return 0.0

    seen = set()
    precisions = []
    hits = 0
    for rank, item in enumerate(islice(recommendations, k), start=1):
        if item in relevant and item not in seen:
            hits += 1
            precisions.append(hits / rank)
        seen.add(item)
    return fsum(precisions) / min(k, len(relevant))


def mean_average_precision_at_k(
    relevant_by_user, recommendations_by_user, user_ids, k=5
):
    """Учитывать всех user_ids; отсутствующие truth или predictions дают ноль."""
    if k <= 0:
        raise ValueError("k должен быть положительным")
    population = tuple(user_ids)
    if len(population) != len(set(population)):
        raise ValueError("Популяция содержит повторяющиеся user_id")
    if not population:
        return 0.0
    return fsum(
        average_precision_at_k(
            relevant_by_user.get(user, ()),
            recommendations_by_user.get(user, ()),
            k,
        )
        for user in population
    ) / len(population)
