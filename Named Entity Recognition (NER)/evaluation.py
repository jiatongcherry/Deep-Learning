import evaluate
from transformers.trainer_utils import EvalPrediction
import numpy as np
import evaluate
from dataset import label_names

metric_evaluator = evaluate.load("seqeval")

def compute_metrics(eval_preds: EvalPrediction):
    """
    Compute evaluation metrics (precision, recall, f1) for predictions.

    First takes the argmax of the logits to convert them to predictions.
    Then we have to convert both labels and predictions from integers to strings.
    We remove all the values where the label is -100, then pass the results to the metric.compute() method.
    Finally, we return the overall precision, recall, and f1 score.

    Args:
        eval_predictions: Evaluation predictions.

    Returns:
        Dictionary with evaluation metrics. Keys: precision, recall, f1.

    NOTE: You can use `metric_evaluator` to compute metrics for a list of predictions and references.
    """
    # Write your code here.

    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)

    # Remove ignored index (special tokens) and convert to labels
    true_labels = [[label_names[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    all_metrics = metric_evaluator.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": all_metrics["overall_precision"],
        "recall": all_metrics["overall_recall"],
        "f1": all_metrics["overall_f1"],
        "accuracy": all_metrics["overall_accuracy"],
    }

