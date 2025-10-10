"""
Sequence Evaluation Metrics
BLEU, ROUGE, METEOR for text generation tasks.

Usage:
    python sequence_metrics.py --reference "The cat is on the mat" --candidate "The cat sits on the mat"
"""

import argparse
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import evaluate
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def calculate_bleu(reference, candidate):
    """Calculate BLEU score."""
    ref_tokens = reference.split()
    cand_tokens = candidate.split()
    smoothie = SmoothingFunction().method4

    bleu = sentence_bleu([ref_tokens], cand_tokens, smoothing_function=smoothie)

    # N-gram scores
    bleu_1 = sentence_bleu([ref_tokens], cand_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothie)
    bleu_2 = sentence_bleu([ref_tokens], cand_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothie)
    bleu_4 = sentence_bleu([ref_tokens], cand_tokens, smoothing_function=smoothie)

    return {
        'bleu': bleu,
        'bleu_1': bleu_1,
        'bleu_2': bleu_2,
        'bleu_4': bleu_4
    }


def calculate_rouge(reference, candidate):
    """Calculate ROUGE scores."""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, candidate)

    return {
        'rouge1_f1': scores['rouge1'].fmeasure,
        'rouge2_f1': scores['rouge2'].fmeasure,
        'rougeL_f1': scores['rougeL'].fmeasure,
    }


def calculate_meteor(reference, candidate):
    """Calculate METEOR score."""
    meteor = evaluate.load('meteor')
    score = meteor.compute(predictions=[candidate], references=[[reference]])
    return score['meteor']


def evaluate_generation(reference, candidate):
    """Comprehensive generation evaluation."""
    logger.info("\n" + "="*50)
    logger.info("SEQUENCE EVALUATION")
    logger.info("="*50)

    logger.info(f"\nReference: {reference}")
    logger.info(f"Candidate: {candidate}")

    # BLEU
    bleu_scores = calculate_bleu(reference, candidate)
    logger.info(f"\nBLEU Scores:")
    for key, value in bleu_scores.items():
        logger.info(f"  {key}: {value:.4f}")

    # ROUGE
    rouge_scores = calculate_rouge(reference, candidate)
    logger.info(f"\nROUGE Scores:")
    for key, value in rouge_scores.items():
        logger.info(f"  {key}: {value:.4f}")

    # METEOR
    try:
        meteor_score = calculate_meteor(reference, candidate)
        logger.info(f"\nMETEOR Score: {meteor_score:.4f}")
    except Exception as e:
        logger.warning(f"Could not calculate METEOR: {e}")
        meteor_score = None

    return {
        **bleu_scores,
        **rouge_scores,
        'meteor': meteor_score
    }


def main(args):
    """Main function."""
    if args.reference and args.candidate:
        results = evaluate_generation(args.reference, args.candidate)
    else:
        # Demo
        logger.info("Running demo...")
        reference = "The cat is sitting on the mat in the living room."
        candidates = [
            "The cat is on the mat.",
            "A feline sits in the room.",
            "The dog is in the park."
        ]

        for i, candidate in enumerate(candidates, 1):
            logger.info(f"\n{'='*50}")
            logger.info(f"Candidate {i}")
            results = evaluate_generation(reference, candidate)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=str, help="Reference text")
    parser.add_argument("--candidate", type=str, help="Candidate text")
    args = parser.parse_args()
    main(args)
