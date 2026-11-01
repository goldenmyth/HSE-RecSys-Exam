"""Три train-only оценки популярности и общий детерминированный Top-K."""

import csv
from collections import Counter

VARIANTS = ("unique_users", "watch_count", "watch_seconds")


def fit_popularity(train_path, expected_period="train"):
    """Одна строка на user/item; группы пользователей — как в подготовленных CSV."""
    if expected_period not in ("train", "final_train"):
        raise ValueError("Допустимы только train и final_train")
    scores = {variant: Counter() for variant in VARIANTS}
    activity, seen_users, current_items, current_user = Counter(), set(), set(), None
    with open(train_path, encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            user, item = row["user_id"], int(row["tv_show_id"])
            count, seconds = int(row["watch_count"]), int(row["total_watch_seconds"])
            if not user or row["period"] != expected_period or item <= 0 or count <= 0 or seconds <= 0:
                raise ValueError("Ожидались корректные положительные train interactions")
            if user != current_user:
                if user in seen_users:
                    raise ValueError("Группа пользователя повторяется; ожидается CSV подготовки")
                seen_users.add(user)
                current_items.clear()
                current_user = user
            if item in current_items:
                raise ValueError("Дублирующаяся пара user/item")
            current_items.add(item)
            activity[user] += 1
            scores["unique_users"][item] += 1
            scores["watch_count"][item] += count
            scores["watch_seconds"][item] += seconds
    return scores, activity


def top_popular(scores, candidates, k=5):
    if k <= 0:
        raise ValueError("k должен быть положительным")
    pool = {int(item) for item in candidates}
    if len(pool) < k or any(item <= 0 for item in pool):
        raise ValueError("Недостаточно различных положительных кандидатов")
    return sorted(pool, key=lambda item: (-scores.get(item, 0), item))[:k]
