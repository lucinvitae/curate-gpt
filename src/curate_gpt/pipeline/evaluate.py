from typing import Union

import dspy


def normalize(hpo_id: str) -> str:
    return hpo_id.strip()


def normalize_list(hpo_ids: list[str]) -> list[str]:
    return list(filter(None, [normalize(r) for r in hpo_ids]))


def calculate_recall(
    ground_truth: list[str], predictions: Union[list[str], str], k: int = 10
) -> float:
    """
    Given ground truth and predictions for HPOs, normalize and compute recall.
    """
    if isinstance(predictions, str):
        predictions = predictions.split(",")

    ground_truth = normalize_list(ground_truth)
    predictions = normalize_list(predictions)[:k]

    ground_truth, predictions = set(ground_truth), set(predictions)

    intersection = ground_truth.intersection(predictions)

    recall = len(intersection) / len(ground_truth)
    return recall


def calculate_recall_at_k(
    ground_truth: list[str], predictions: Union[list[str], str], k: int = 10
) -> float:
    return calculate_recall(ground_truth, predictions, k=k)


def dspy_metric_recall10(
    ground_truth: dspy.Example, predictions: dspy.Example, trace=None
) -> float:
    """
    Wrap the recall@K metric so it can handle dspy.Example objects.
    """
    return calculate_recall_at_k(ground_truth.labels().hpo_ids, predictions.hpo_ids, k=10)
