"""
Embedding-based Metrics
BERTScore and semantic similarity using embeddings.

Usage:
    python embedding_metrics.py --text1 "The cat sat" --text2 "A feline rested"
"""

import argparse
from bert_score import score as bert_score
from sentence_transformers import SentenceTransformer, util
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def calculate_bertscore(reference, candidate):
    """Calculate BERTScore."""
    P, R, F1 = bert_score([candidate], [reference], lang="en", verbose=False)
    return {
        'bertscore_precision': P.mean().item(),
        'bertscore_recall': R.mean().item(),
        'bertscore_f1': F1.mean().item()
    }


def calculate_semantic_similarity(text1, text2, model_name='all-MiniLM-L6-v2'):
    """Calculate semantic similarity using sentence transformers."""
    model = SentenceTransformer(model_name)

    emb1 = model.encode(text1)
    emb2 = model.encode(text2)

    similarity = util.cos_sim(emb1, emb2)[0][0].item()
    return similarity


def main(args):
    """Main function."""
    if args.text1 and args.text2:
        logger.info("\n" + "="*50)
        logger.info("EMBEDDING-BASED EVALUATION")
        logger.info("="*50)
        logger.info(f"\nText 1: {args.text1}")
        logger.info(f"Text 2: {args.text2}")

        # BERTScore
        bert_scores = calculate_bertscore(args.text1, args.text2)
        logger.info(f"\nBERTScore:")
        for key, value in bert_scores.items():
            logger.info(f"  {key}: {value:.4f}")

        # Semantic similarity
        similarity = calculate_semantic_similarity(args.text1, args.text2)
        logger.info(f"\nSemantic Similarity: {similarity:.4f}")
    else:
        logger.info("Please provide --text1 and --text2")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text1", type=str)
    parser.add_argument("--text2", type=str)
    args = parser.parse_args()
    main(args)
