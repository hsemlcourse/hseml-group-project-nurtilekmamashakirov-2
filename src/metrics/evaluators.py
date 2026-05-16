import numpy as np
import pandas as pd

from src.models.base import BaseModel


def precision_at_k(recommended: list, relevant: set, k: int) -> float:
    recommended_k = recommended[:k]
    hits = sum(1 for item in recommended_k if item in relevant)
    return hits / k


def hit_rate_at_k(recommended: list, relevant: set, k: int) -> float:
    return float(any(item in relevant for item in recommended[:k]))


def ap_at_k(recommended: list, relevant: set, k: int) -> float:
    recommended_k = recommended[:k]
    hits, score = 0, 0.0
    for i, item in enumerate(recommended_k, start=1):
        if item in relevant:
            hits += 1
            score += hits / i
    return score / min(len(relevant), k) if relevant else 0.0


def ndcg_at_k(recommended: list, relevant: set, k: int) -> float:
    recommended_k = recommended[:k]
    dcg = sum(
        1.0 / np.log2(i + 2)
        for i, item in enumerate(recommended_k)
        if item in relevant
    )
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_recommenders(
    recommenders: dict[str, BaseModel],
    ground_truth: pd.DataFrame,
    k: int = 10,
    user_col: str = 'uid',
    relevant_col: str = 'relevant_items',
) -> pd.DataFrame:
    """
    Считает ndcg@k, map@k, precision@k, hit_rate@k для каждого рекоммендера.

    Args:
        recommenders: dict[str, RecommenderProtocol] – словарь {name: recommender},
                      у каждого должен быть метод recommend_items(user_id, N) -> list
        ground_truth: pd.DataFrame – индекс = user_id, колонка relevant_col со списком релевантных item_id
        k: int – cutoff для метрик
        user_col: str – название индекса в ground_truth
        relevant_col: str – колонка с релевантными айтемами

    Returns:
        pd.DataFrame: строки = рекоммендеры, колонки = метрики
    """
    user_ids = ground_truth.index.tolist()

    # Словарь user_id -> set релевантных айтемов
    relevant_map: dict = {
        user_id: set(items)
        for user_id, items in ground_truth[relevant_col].items()
    }

    results = {}

    for name, recommender in recommenders.items():
        print(f"Evaluating '{name}'...")

        # Батчевый запрос рекомендаций для всех юзеров сразу
        all_recommendations: list[list] = recommender.recommend_items(user_ids, N=k)

        precision_scores, hit_rate_scores, ap_scores, ndcg_scores = [], [], [], []

        for user_id, recommended in zip(user_ids, all_recommendations):
            relevant = relevant_map[user_id]

            precision_scores.append(precision_at_k(recommended, relevant, k))
            hit_rate_scores.append(hit_rate_at_k(recommended, relevant, k))
            ap_scores.append(ap_at_k(recommended, relevant, k))
            ndcg_scores.append(ndcg_at_k(recommended, relevant, k))

        results[name] = {
            f'precision@{k}':  np.mean(precision_scores),
            f'hit_rate@{k}':   np.mean(hit_rate_scores),
            f'map@{k}':        np.mean(ap_scores),
            f'ndcg@{k}':       np.mean(ndcg_scores),
        }

        print(f"  precision@{k} = {results[name][f'precision@{k}']:.4f}")
        print(f"  hit_rate@{k}  = {results[name][f'hit_rate@{k}']:.4f}")
        print(f"  map@{k}       = {results[name][f'map@{k}']:.4f}")
        print(f"  ndcg@{k}      = {results[name][f'ndcg@{k}']:.4f}")

    return pd.DataFrame(results).T.rename_axis(user_col)