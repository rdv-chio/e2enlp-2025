"""
Classification Metrics Evaluation
Comprehensive evaluation for classification tasks.

Usage:
    python classification_metrics.py --demo
    python classification_metrics.py --predictions pred.txt --labels true.txt
"""

import argparse
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate_classifier(y_true, y_pred, y_proba=None, labels=None):
    """
    Comprehensive classifier evaluation.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Prediction probabilities (optional, for ROC-AUC)
        labels: Label names (optional)

    Returns:
        Dictionary of metrics
    """
    results = {}

    # Basic metrics
    results['accuracy'] = accuracy_score(y_true, y_pred)
    results['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    results['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    results['f1'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    # Confusion matrix
    results['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()

    # Detailed report
    if labels:
        results['classification_report'] = classification_report(
            y_true, y_pred, target_names=labels, output_dict=True, zero_division=0
        )
    else:
        results['classification_report'] = classification_report(
            y_true, y_pred, output_dict=True, zero_division=0
        )

    # ROC-AUC for binary classification
    if y_proba is not None and len(np.unique(y_true)) == 2:
        try:
            results['roc_auc'] = roc_auc_score(y_true, y_proba)
        except:
            logger.warning("Could not calculate ROC-AUC")

    return results


def plot_confusion_matrix(cm, labels=None, save_path=None):
    """Plot confusion matrix."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Confusion matrix saved to {save_path}")

    plt.show()


def plot_roc_curve(y_true, y_proba, save_path=None):
    """Plot ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"ROC curve saved to {save_path}")

    plt.show()


def demo_evaluation():
    """Run a demo evaluation."""
    logger.info("Running demo evaluation...")

    # Sample data
    y_true = [1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0]
    y_pred = [1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1]
    y_proba = [0.9, 0.2, 0.85, 0.45, 0.1, 0.95, 0.3, 0.6, 0.88, 0.15,
               0.92, 0.87, 0.25, 0.48, 0.55]

    labels = ['Negative', 'Positive']

    # Evaluate
    results = evaluate_classifier(y_true, y_pred, y_proba, labels)

    # Print results
    logger.info("\n" + "="*50)
    logger.info("EVALUATION RESULTS")
    logger.info("="*50)

    logger.info(f"\nOverall Metrics:")
    logger.info(f"  Accuracy:  {results['accuracy']:.4f}")
    logger.info(f"  Precision: {results['precision']:.4f}")
    logger.info(f"  Recall:    {results['recall']:.4f}")
    logger.info(f"  F1 Score:  {results['f1']:.4f}")
    if 'roc_auc' in results:
        logger.info(f"  ROC-AUC:   {results['roc_auc']:.4f}")

    logger.info(f"\nPer-Class Metrics:")
    for label in labels:
        if label in results['classification_report']:
            metrics = results['classification_report'][label]
            logger.info(f"  {label}:")
            logger.info(f"    Precision: {metrics['precision']:.4f}")
            logger.info(f"    Recall:    {metrics['recall']:.4f}")
            logger.info(f"    F1-Score:  {metrics['f1-score']:.4f}")

    # Plot confusion matrix
    cm = np.array(results['confusion_matrix'])
    plot_confusion_matrix(cm, labels, 'confusion_matrix.png')

    # Plot ROC curve
    if 'roc_auc' in results:
        plot_roc_curve(y_true, y_proba, 'roc_curve.png')

    # Save results to JSON
    with open('evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    logger.info("\nResults saved to evaluation_results.json")


def main(args):
    """Main evaluation function."""
    if args.demo:
        demo_evaluation()
        return

    # Load predictions and labels
    logger.info("Loading data...")

    if args.predictions_file and args.labels_file:
        with open(args.predictions_file) as f:
            y_pred = [int(line.strip()) for line in f]

        with open(args.labels_file) as f:
            y_true = [int(line.strip()) for line in f]

        y_proba = None
        if args.probabilities_file:
            with open(args.probabilities_file) as f:
                y_proba = [float(line.strip()) for line in f]

        # Evaluate
        results = evaluate_classifier(y_true, y_pred, y_proba)

        # Print results
        logger.info("\nEvaluation Results:")
        for key, value in results.items():
            if key not in ['confusion_matrix', 'classification_report']:
                logger.info(f"  {key}: {value:.4f}")

        # Save results
        output_file = args.output or 'evaluation_results.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"\nResults saved to {output_file}")

    else:
        logger.error("Please provide --predictions-file and --labels-file, or use --demo")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate classification models")

    parser.add_argument("--demo", action="store_true", help="Run demo evaluation")
    parser.add_argument("--predictions-file", type=str, help="File with predictions")
    parser.add_argument("--labels-file", type=str, help="File with true labels")
    parser.add_argument("--probabilities-file", type=str, help="File with prediction probabilities")
    parser.add_argument("--output", type=str, help="Output JSON file")

    args = parser.parse_args()
    main(args)
