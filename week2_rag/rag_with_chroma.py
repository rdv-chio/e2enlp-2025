"""
RAG with ChromaDB
Persistent vector store using ChromaDB.

Usage:
    python rag_with_chroma.py --add-docs  # Add documents
    python rag_with_chroma.py --query "What is NLP?"
"""

import argparse
import os
from pathlib import Path
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_core.documents import Document
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Persistent directory for ChromaDB
CHROMA_DIR = "./chroma_db"


SAMPLE_DOCUMENTS = [
    {
        "content": """Python is a high-level programming language known for its simplicity and readability.
        It was created by Guido van Rossum and first released in 1991. Python supports multiple
        programming paradigms including procedural, object-oriented, and functional programming.""",
        "metadata": {"topic": "programming", "language": "python"}
    },
    {
        "content": """Machine learning is a subset of artificial intelligence that enables systems to learn
        from data without explicit programming. It includes supervised, unsupervised, and reinforcement
        learning approaches.""",
        "metadata": {"topic": "ai", "subtopic": "machine-learning"}
    },
    {
        "content": """Natural Language Processing allows computers to understand and generate human language.
        Modern NLP uses transformer models like BERT and GPT for various tasks including translation,
        summarization, and question answering.""",
        "metadata": {"topic": "ai", "subtopic": "nlp"}
    },
    {
        "content": """Deep learning uses neural networks with multiple layers to learn hierarchical
        representations. It has achieved breakthrough results in computer vision, speech recognition,
        and natural language understanding.""",
        "metadata": {"topic": "ai", "subtopic": "deep-learning"}
    },
    {
        "content": """FastAPI is a modern web framework for building APIs with Python. It's fast, easy to use,
        and comes with automatic interactive documentation. It's built on Starlette and Pydantic.""",
        "metadata": {"topic": "programming", "framework": "fastapi"}
    },
]


def add_documents_to_chroma(documents, embeddings):
    """
    Add documents to ChromaDB.

    Args:
        documents: List of document dictionaries with content and metadata
        embeddings: Embeddings model

    Returns:
        Chroma vector store
    """
    logger.info("Adding documents to ChromaDB...")

    # Convert to Document objects
    docs = [
        Document(page_content=doc["content"], metadata=doc["metadata"])
        for doc in documents
    ]

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    splits = text_splitter.split_documents(docs)

    logger.info(f"Created {len(splits)} document chunks")

    # Create or update ChromaDB
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )

    logger.info(f"Documents added to ChromaDB at {CHROMA_DIR}")

    return vectorstore


def load_chroma_db(embeddings):
    """
    Load existing ChromaDB.

    Args:
        embeddings: Embeddings model

    Returns:
        Chroma vector store or None if not found
    """
    if not Path(CHROMA_DIR).exists():
        logger.warning(f"ChromaDB not found at {CHROMA_DIR}")
        logger.warning("Run with --add-docs first to create the database")
        return None

    logger.info(f"Loading ChromaDB from {CHROMA_DIR}")

    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings
    )

    # Check if database has documents
    collection_count = vectorstore._collection.count()
    logger.info(f"Found {collection_count} documents in ChromaDB")

    if collection_count == 0:
        logger.warning("ChromaDB is empty. Add documents with --add-docs")
        return None

    return vectorstore


def query_with_filter(vectorstore, llm, question, filter_dict=None):
    """
    Query with optional metadata filtering.

    Args:
        vectorstore: Vector store
        llm: Language model
        question: Question to ask
        filter_dict: Optional metadata filter

    Returns:
        Dictionary with answer and sources
    """
    # Create retriever with optional filter
    search_kwargs = {"k": 3}
    if filter_dict:
        search_kwargs["filter"] = filter_dict

    retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)

    # Create QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

    logger.info(f"\nQuery: {question}")
    if filter_dict:
        logger.info(f"Filter: {filter_dict}")

    result = qa_chain.invoke({"query": question})

    return {
        "answer": result["result"],
        "sources": [
            {
                "content": doc.page_content,
                "metadata": doc.metadata
            }
            for doc in result["source_documents"]
        ]
    }


def main(args):
    """Main function."""
    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY not found")
        return

    # Initialize embeddings and LLM
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Add documents mode
    if args.add_docs:
        vectorstore = add_documents_to_chroma(SAMPLE_DOCUMENTS, embeddings)
        logger.info("\nDocuments added successfully!")
        logger.info("Now you can query with: python rag_with_chroma.py --query 'your question'")
        return

    # Load existing database
    vectorstore = load_chroma_db(embeddings)

    if vectorstore is None:
        return

    # Query mode
    if args.query:
        # Parse filter if provided
        filter_dict = None
        if args.filter:
            # Simple filter parsing: "topic=ai" or "subtopic=nlp"
            try:
                key, value = args.filter.split("=")
                filter_dict = {key: value}
            except:
                logger.warning(f"Invalid filter format: {args.filter}")

        result = query_with_filter(vectorstore, llm, args.query, filter_dict)

        print(f"\n{'='*80}")
        print(f"QUESTION: {args.query}")
        if filter_dict:
            print(f"FILTER: {filter_dict}")
        print(f"\n{'='*80}")
        print(f"ANSWER:\n{result['answer']}")
        print(f"\n{'='*80}")
        print("SOURCES:")
        for i, source in enumerate(result['sources'], 1):
            print(f"\n{i}. {source['content'][:200]}...")
            print(f"   Metadata: {source['metadata']}")
        print(f"{'='*80}\n")

    else:
        # Demo queries
        logger.info("\nDemo Mode - Running sample queries...\n")

        queries = [
            ("What is Python?", None),
            ("Tell me about NLP", {"topic": "ai"}),
            ("What frameworks are mentioned?", {"topic": "programming"}),
        ]

        for question, filter_dict in queries:
            result = query_with_filter(vectorstore, llm, question, filter_dict)

            print(f"\n{'='*80}")
            print(f"Q: {question}")
            if filter_dict:
                print(f"Filter: {filter_dict}")
            print(f"A: {result['answer']}")
            print(f"{'='*80}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG with ChromaDB")

    parser.add_argument(
        "--add-docs",
        action="store_true",
        help="Add sample documents to ChromaDB"
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Query to ask"
    )
    parser.add_argument(
        "--filter",
        type=str,
        help="Metadata filter (e.g., 'topic=ai')"
    )

    args = parser.parse_args()
    main(args)
