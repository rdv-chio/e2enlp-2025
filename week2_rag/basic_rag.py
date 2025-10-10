"""
Basic RAG Implementation
Simple RAG system using FAISS and OpenAI.

Usage:
    python basic_rag.py --query "What is machine learning?"
"""

import argparse
import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_core.documents import Document
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Sample knowledge base
SAMPLE_DOCUMENTS = [
    """Machine learning is a subset of artificial intelligence that enables systems to learn
    and improve from experience without being explicitly programmed. It focuses on developing
    algorithms that can access data and use it to learn for themselves.""",

    """Deep learning is a subset of machine learning based on artificial neural networks.
    It uses multiple layers to progressively extract higher-level features from raw input.
    Common applications include computer vision, speech recognition, and natural language processing.""",

    """Natural Language Processing (NLP) is a branch of AI that helps computers understand,
    interpret, and manipulate human language. NLP combines computational linguistics with
    statistical machine learning and deep learning models.""",

    """Transformers revolutionized NLP by introducing the attention mechanism, allowing models
    to process sequences in parallel. Models like BERT and GPT are based on transformer architecture.""",

    """Fine-tuning is the process of taking a pretrained model and training it further on a
    specific task or domain. This allows the model to adapt its knowledge to specialized use cases
    with relatively little additional data.""",
]


def create_vector_store(documents, embeddings):
    """
    Create a vector store from documents.

    Args:
        documents: List of text documents
        embeddings: Embeddings model

    Returns:
        FAISS vector store
    """
    logger.info("Creating vector store...")

    # Convert to Document objects
    docs = [Document(page_content=doc) for doc in documents]

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    splits = text_splitter.split_documents(docs)

    logger.info(f"Created {len(splits)} document chunks")

    # Create vector store
    vectorstore = FAISS.from_documents(splits, embeddings)

    return vectorstore


def create_rag_chain(vectorstore, llm):
    """
    Create a RAG QA chain.

    Args:
        vectorstore: Vector store for retrieval
        llm: Language model

    Returns:
        RetrievalQA chain
    """
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        verbose=True
    )

    return qa_chain


def query_rag(qa_chain, question):
    """
    Query the RAG system.

    Args:
        qa_chain: QA chain
        question: Question to ask

    Returns:
        Dictionary with answer and sources
    """
    logger.info(f"\nQuery: {question}")

    result = qa_chain.invoke({"query": question})

    return {
        "answer": result["result"],
        "sources": [doc.page_content for doc in result["source_documents"]]
    }


def main(args):
    """Main function."""
    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY not found in environment variables")
        logger.error("Please set it: export OPENAI_API_KEY='your-key-here'")
        return

    logger.info("Initializing RAG system...")

    # Initialize embeddings and LLM
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Create vector store
    vectorstore = create_vector_store(SAMPLE_DOCUMENTS, embeddings)

    # Create RAG chain
    qa_chain = create_rag_chain(vectorstore, llm)

    logger.info("RAG system ready!\n")

    # Interactive mode or single query
    if args.interactive:
        logger.info("Interactive mode. Type 'quit' to exit.\n")
        while True:
            question = input("\nYour question: ").strip()

            if question.lower() in ['quit', 'exit', 'q']:
                logger.info("Goodbye!")
                break

            if not question:
                continue

            try:
                result = query_rag(qa_chain, question)

                print(f"\n{'='*80}")
                print(f"ANSWER:\n{result['answer']}")
                print(f"\n{'='*80}")
                print("SOURCES:")
                for i, source in enumerate(result['sources'], 1):
                    print(f"\n{i}. {source[:200]}...")
                print(f"{'='*80}")

            except Exception as e:
                logger.error(f"Error: {e}")

    else:
        # Single query mode
        if args.query:
            result = query_rag(qa_chain, args.query)

            print(f"\n{'='*80}")
            print(f"QUESTION: {args.query}")
            print(f"\n{'='*80}")
            print(f"ANSWER:\n{result['answer']}")
            print(f"\n{'='*80}")
            print("SOURCES:")
            for i, source in enumerate(result['sources'], 1):
                print(f"\n{i}. {source[:200]}...")
            print(f"{'='*80}\n")

        else:
            # Demo mode - ask multiple questions
            demo_questions = [
                "What is machine learning?",
                "How does deep learning differ from machine learning?",
                "What are transformers?",
            ]

            for question in demo_questions:
                result = query_rag(qa_chain, question)

                print(f"\n{'='*80}")
                print(f"Q: {question}")
                print(f"A: {result['answer']}")
                print(f"{'='*80}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Basic RAG implementation")

    parser.add_argument(
        "--query",
        type=str,
        help="Single query to ask"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Interactive mode"
    )

    args = parser.parse_args()
    main(args)
