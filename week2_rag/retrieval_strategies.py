"""
Advanced Retrieval Strategies for RAG
Different approaches to retrieval including dense, sparse, hybrid, and reranking.

Usage:
    python retrieval_strategies.py --strategy similarity
    python retrieval_strategies.py --strategy mmr
    python retrieval_strategies.py --strategy hybrid
    python retrieval_strategies.py --strategy all
"""

import argparse
import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS, Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain.chains import RetrievalQA
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Sample knowledge base
SAMPLE_DOCUMENTS = [
    """Machine learning is a subset of artificial intelligence that enables systems to learn
    and improve from experience without being explicitly programmed. It uses statistical
    techniques to give computers the ability to learn from data.""",

    """Deep learning is a subset of machine learning based on artificial neural networks
    with multiple layers. It excels at processing unstructured data like images, audio,
    and text. Deep learning powers modern AI applications.""",

    """Natural Language Processing (NLP) is a branch of AI focused on enabling computers
    to understand, interpret, and generate human language. Modern NLP uses transformer
    models like BERT, GPT, and T5.""",

    """Transformers revolutionized NLP in 2017 with the attention mechanism. Unlike RNNs,
    transformers can process sequences in parallel, making them much faster to train.
    They form the basis of models like GPT-4 and BERT.""",

    """Fine-tuning is the process of taking a pretrained model and training it further on
    a specific task or domain. This transfer learning approach allows models to adapt to
    specialized use cases with relatively little data.""",

    """Retrieval Augmented Generation (RAG) combines retrieval with generation. It retrieves
    relevant documents from a knowledge base and uses them to generate informed responses.
    RAG reduces hallucinations and grounds answers in facts.""",

    """Vector databases store embeddings and enable efficient similarity search. Popular
    options include Pinecone, Weaviate, Chroma, and FAISS. They are essential for RAG systems.""",

    """Embeddings are dense vector representations of text that capture semantic meaning.
    Similar texts have similar embeddings. Modern embeddings models include OpenAI's
    text-embedding-3, Cohere's embeddings, and sentence-transformers.""",
]


def create_documents():
    """Create Document objects from sample texts."""
    docs = [Document(page_content=doc, metadata={"source": f"doc_{i}"})
            for i, doc in enumerate(SAMPLE_DOCUMENTS)]

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    splits = text_splitter.split_documents(docs)

    logger.info(f"Created {len(splits)} document chunks")
    return splits


# ============================================================================
# 1. SIMILARITY SEARCH - Standard dense vector retrieval
# ============================================================================

def demo_similarity_search():
    """
    Similarity Search: Standard cosine similarity between query and document embeddings.
    Simple and effective for most use cases.
    """
    logger.info("\n" + "="*80)
    logger.info("STRATEGY 1: SIMILARITY SEARCH")
    logger.info("="*80)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    splits = create_documents()

    # Create vector store
    vectorstore = FAISS.from_documents(splits, embeddings)

    # Create retriever with similarity search
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

    query = "What are transformers in NLP?"
    docs = retriever.invoke(query)

    print(f"\nQUERY: {query}")
    print(f"\nRETRIEVED {len(docs)} DOCUMENTS:\n")
    for i, doc in enumerate(docs, 1):
        print(f"{i}. [Score: similarity] {doc.page_content[:200]}...")
        print(f"   Source: {doc.metadata.get('source', 'unknown')}\n")

    return docs


# ============================================================================
# 2. MMR (Maximal Marginal Relevance) - Diverse results
# ============================================================================

def demo_mmr_search():
    """
    MMR Search: Balance between relevance and diversity.
    Prevents returning too many similar documents.
    Good for: Exploratory search, avoiding redundancy
    """
    logger.info("\n" + "="*80)
    logger.info("STRATEGY 2: MMR (MAXIMAL MARGINAL RELEVANCE)")
    logger.info("="*80)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    splits = create_documents()

    vectorstore = FAISS.from_documents(splits, embeddings)

    # MMR retriever - balances relevance vs diversity
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 3,
            "fetch_k": 10,  # Fetch more candidates
            "lambda_mult": 0.5  # 0 = max diversity, 1 = max relevance
        }
    )

    query = "Tell me about machine learning"
    docs = retriever.invoke(query)

    print(f"\nQUERY: {query}")
    print(f"\nRETRIEVED {len(docs)} DIVERSE DOCUMENTS:\n")
    for i, doc in enumerate(docs, 1):
        print(f"{i}. {doc.page_content[:200]}...")
        print(f"   Source: {doc.metadata.get('source', 'unknown')}\n")

    return docs


# ============================================================================
# 3. SIMILARITY SCORE THRESHOLD - Filter by confidence
# ============================================================================

def demo_threshold_search():
    """
    Threshold Search: Only return documents above a similarity score threshold.
    Good for: High precision retrieval, filtering out irrelevant results
    """
    logger.info("\n" + "="*80)
    logger.info("STRATEGY 3: SIMILARITY SCORE THRESHOLD")
    logger.info("="*80)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    splits = create_documents()

    vectorstore = FAISS.from_documents(splits, embeddings)

    # Threshold retriever - only return docs above similarity threshold
    retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "score_threshold": 0.5,  # Only return docs with score > 0.5
            "k": 5  # Maximum number to return
        }
    )

    queries = [
        "What are transformers?",  # Should match well
        "How do I bake cookies?"   # Should not match (off-topic)
    ]

    for query in queries:
        docs = retriever.invoke(query)

        print(f"\nQUERY: {query}")
        print(f"RETRIEVED {len(docs)} DOCUMENTS (threshold=0.5):\n")

        if docs:
            for i, doc in enumerate(docs, 1):
                print(f"{i}. {doc.page_content[:150]}...\n")
        else:
            print("No documents above threshold.\n")


# ============================================================================
# 4. HYBRID SEARCH - Combine dense + sparse retrieval
# ============================================================================

def demo_hybrid_search():
    """
    Hybrid Search: Combines dense (embeddings) and sparse (BM25/keyword) retrieval.
    Best of both worlds: semantic understanding + keyword matching.
    Good for: Production systems, handling diverse queries
    """
    logger.info("\n" + "="*80)
    logger.info("STRATEGY 4: HYBRID SEARCH (Dense + Sparse)")
    logger.info("="*80)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    splits = create_documents()

    # Dense retriever (embeddings)
    vectorstore = FAISS.from_documents(splits, embeddings)
    dense_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Sparse retriever (BM25 - keyword based)
    bm25_retriever = BM25Retriever.from_documents(splits)
    bm25_retriever.k = 3

    # Ensemble: Combine both retrievers
    ensemble_retriever = EnsembleRetriever(
        retrievers=[dense_retriever, bm25_retriever],
        weights=[0.5, 0.5]  # Equal weight to both
    )

    query = "What is RAG and how does it use retrieval?"
    docs = ensemble_retriever.invoke(query)

    print(f"\nQUERY: {query}")
    print(f"\nRETRIEVED {len(docs)} DOCUMENTS (Hybrid):\n")
    for i, doc in enumerate(docs, 1):
        print(f"{i}. {doc.page_content[:200]}...")
        print(f"   Source: {doc.metadata.get('source', 'unknown')}\n")

    return docs


# ============================================================================
# 5. CONTEXTUAL COMPRESSION - Compress retrieved docs
# ============================================================================

def demo_contextual_compression():
    """
    Contextual Compression: Retrieve documents, then compress/extract only relevant parts.
    Reduces noise and focuses on query-relevant information.
    Good for: Long documents, improving relevance
    """
    logger.info("\n" + "="*80)
    logger.info("STRATEGY 5: CONTEXTUAL COMPRESSION")
    logger.info("="*80)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    splits = create_documents()

    # Base retriever
    vectorstore = FAISS.from_documents(splits, embeddings)
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # Add compression layer
    compressor = LLMChainExtractor.from_llm(llm)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever
    )

    query = "How do transformers process sequences?"

    # Compare: Before and after compression
    print(f"\nQUERY: {query}\n")

    print("BEFORE COMPRESSION:")
    docs_before = base_retriever.invoke(query)
    print(f"Retrieved {len(docs_before)} documents")
    for i, doc in enumerate(docs_before[:2], 1):
        print(f"\n{i}. {doc.page_content[:200]}...")

    print("\n" + "-"*80)
    print("\nAFTER COMPRESSION:")
    docs_after = compression_retriever.invoke(query)
    print(f"Retrieved {len(docs_after)} compressed documents")
    for i, doc in enumerate(docs_after, 1):
        print(f"\n{i}. {doc.page_content}")

    return docs_after


# ============================================================================
# 6. MULTI-QUERY RETRIEVAL - Generate multiple query variations
# ============================================================================

def demo_multi_query():
    """
    Multi-Query Retrieval: Generate multiple variations of the query and retrieve for each.
    Improves recall by capturing different phrasings.
    Good for: Complex queries, improving coverage
    """
    logger.info("\n" + "="*80)
    logger.info("STRATEGY 6: MULTI-QUERY RETRIEVAL")
    logger.info("="*80)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    splits = create_documents()

    vectorstore = FAISS.from_documents(splits, embeddings)

    original_query = "What is deep learning?"

    # Generate query variations
    prompt = f"""Generate 3 different versions of this query for retrieving relevant documents:

Original query: {original_query}

Variations (one per line):"""

    variations_text = llm.invoke(prompt).content
    variations = [original_query] + [v.strip() for v in variations_text.split('\n') if v.strip()]

    print(f"ORIGINAL QUERY: {original_query}\n")
    print(f"GENERATED VARIATIONS:")
    for i, var in enumerate(variations[1:], 1):
        print(f"{i}. {var}")

    # Retrieve for each variation
    all_docs = []
    for variation in variations:
        docs = vectorstore.similarity_search(variation, k=2)
        all_docs.extend(docs)

    # Deduplicate
    unique_docs = []
    seen = set()
    for doc in all_docs:
        doc_hash = hash(doc.page_content)
        if doc_hash not in seen:
            seen.add(doc_hash)
            unique_docs.append(doc)

    print(f"\n\nRETRIEVED {len(unique_docs)} UNIQUE DOCUMENTS:")
    for i, doc in enumerate(unique_docs[:3], 1):
        print(f"\n{i}. {doc.page_content[:200]}...")


# ============================================================================
# 7. PARENT DOCUMENT RETRIEVAL - Retrieve chunks, return full docs
# ============================================================================

def demo_parent_document_retrieval():
    """
    Parent Document Retrieval: Search on small chunks, but return larger parent documents.
    Balances precise matching with context.
    Good for: Maintaining context, avoiding fragmentation
    """
    logger.info("\n" + "="*80)
    logger.info("STRATEGY 7: PARENT DOCUMENT RETRIEVAL")
    logger.info("="*80)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # Simulate parent-child relationship
    # In production, use ParentDocumentRetriever from langchain
    parent_docs = [Document(page_content=doc, metadata={"doc_id": f"parent_{i}"})
                   for i, doc in enumerate(SAMPLE_DOCUMENTS)]

    # Create smaller chunks for searching
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
    child_docs = []
    for parent in parent_docs:
        chunks = child_splitter.split_documents([parent])
        for chunk in chunks:
            chunk.metadata["parent_id"] = parent.metadata["doc_id"]
            child_docs.append(chunk)

    # Search on child chunks
    vectorstore = FAISS.from_documents(child_docs, embeddings)

    query = "What are embeddings?"
    retrieved_chunks = vectorstore.similarity_search(query, k=2)

    # Get parent documents
    parent_ids = [chunk.metadata.get("parent_id") for chunk in retrieved_chunks]
    parent_docs_retrieved = [doc for doc in parent_docs
                             if doc.metadata["doc_id"] in parent_ids]

    print(f"\nQUERY: {query}\n")
    print(f"RETRIEVED CHUNKS (for matching):")
    for i, chunk in enumerate(retrieved_chunks, 1):
        print(f"{i}. {chunk.page_content}\n")

    print("-"*80)
    print(f"\nRETURNED PARENT DOCUMENTS (for context):")
    for i, parent in enumerate(parent_docs_retrieved, 1):
        print(f"{i}. {parent.page_content}\n")


# ============================================================================
# 8. COMPARISON - Run all strategies on same query
# ============================================================================

def demo_comparison():
    """
    Compare all retrieval strategies on the same query.
    """
    logger.info("\n" + "="*80)
    logger.info("STRATEGY COMPARISON")
    logger.info("="*80)

    query = "Explain transformers in NLP"

    print(f"\nQUERY: {query}\n")
    print("="*80)

    strategies = [
        ("Similarity", demo_similarity_search),
        ("MMR", demo_mmr_search),
        ("Hybrid", demo_hybrid_search),
    ]

    for name, func in strategies:
        try:
            logger.info(f"\n Running {name}...")
            func()
        except Exception as e:
            logger.error(f"Error in {name}: {e}")


# ============================================================================
# MAIN
# ============================================================================

def main(args):
    """Main function."""
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY not found")
        logger.error("Please set it: export OPENAI_API_KEY='your-key-here'")
        return

    logger.info("Advanced Retrieval Strategies for RAG")

    if args.strategy == "similarity":
        demo_similarity_search()
    elif args.strategy == "mmr":
        demo_mmr_search()
    elif args.strategy == "threshold":
        demo_threshold_search()
    elif args.strategy == "hybrid":
        demo_hybrid_search()
    elif args.strategy == "compression":
        demo_contextual_compression()
    elif args.strategy == "multiquery":
        demo_multi_query()
    elif args.strategy == "parent":
        demo_parent_document_retrieval()
    elif args.strategy == "compare":
        demo_comparison()
    elif args.strategy == "all":
        demo_similarity_search()
        demo_mmr_search()
        demo_threshold_search()
        demo_hybrid_search()
        demo_contextual_compression()
        demo_multi_query()
        demo_parent_document_retrieval()
    else:
        logger.error(f"Unknown strategy: {args.strategy}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Advanced retrieval strategies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Standard similarity search
    python retrieval_strategies.py --strategy similarity

    # MMR for diverse results
    python retrieval_strategies.py --strategy mmr

    # Hybrid search (dense + sparse)
    python retrieval_strategies.py --strategy hybrid

    # Contextual compression
    python retrieval_strategies.py --strategy compression

    # Compare strategies
    python retrieval_strategies.py --strategy compare

    # Run all demos
    python retrieval_strategies.py --strategy all
        """
    )

    parser.add_argument(
        "--strategy",
        type=str,
        default="compare",
        choices=["similarity", "mmr", "threshold", "hybrid", "compression",
                 "multiquery", "parent", "compare", "all"],
        help="Which retrieval strategy to demonstrate"
    )

    args = parser.parse_args()
    main(args)
