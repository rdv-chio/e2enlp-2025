"""
RAG with Custom Documents
Load documents from files and build RAG system.

Usage:
    python rag_with_custom_docs.py --docs-dir ./my_documents
    python rag_with_custom_docs.py --docs-dir ./my_documents --query "Your question"
"""

import argparse
import os
from pathlib import Path
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import (
    TextLoader,
    DirectoryLoader,
    PyPDFLoader
)
from langchain_core.documents import Document
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sample_documents(docs_dir):
    """Create sample documents for testing."""
    docs_path = Path(docs_dir)
    docs_path.mkdir(exist_ok=True)

    sample_docs = {
        "machine_learning.txt": """
Machine Learning Fundamentals

Machine learning is a method of data analysis that automates analytical model building.
It is a branch of artificial intelligence based on the idea that systems can learn from data,
identify patterns, and make decisions with minimal human intervention.

Types of Machine Learning:
1. Supervised Learning: The algorithm learns from labeled training data
2. Unsupervised Learning: The algorithm finds patterns in unlabeled data
3. Reinforcement Learning: The algorithm learns through trial and error

Common Applications:
- Image recognition
- Natural language processing
- Recommendation systems
- Fraud detection
- Autonomous vehicles
""",

        "deep_learning.txt": """
Deep Learning Overview

Deep learning is a subset of machine learning that uses neural networks with multiple layers.
These networks can learn increasingly abstract representations of data.

Key Concepts:
- Neural Networks: Computational models inspired by the human brain
- Layers: Input, hidden, and output layers that process information
- Activation Functions: Functions that introduce non-linearity
- Backpropagation: Algorithm for training neural networks

Popular Architectures:
1. Convolutional Neural Networks (CNNs) - for images
2. Recurrent Neural Networks (RNNs) - for sequences
3. Transformers - for natural language

Applications:
- Computer vision
- Speech recognition
- Natural language understanding
- Game playing (AlphaGo, Chess)
""",

        "nlp_basics.txt": """
Natural Language Processing

Natural Language Processing (NLP) enables computers to understand, interpret, and generate
human language in a valuable way.

Core NLP Tasks:
1. Text Classification: Categorizing text into predefined classes
2. Named Entity Recognition: Identifying entities like names, locations
3. Sentiment Analysis: Determining emotional tone
4. Machine Translation: Converting text between languages
5. Question Answering: Extracting answers from text
6. Text Summarization: Creating concise summaries

Modern NLP Approaches:
- Transformer models (BERT, GPT, T5)
- Transfer learning and fine-tuning
- Few-shot and zero-shot learning
- Retrieval Augmented Generation (RAG)

Tools and Libraries:
- Hugging Face Transformers
- spaCy
- NLTK
- OpenAI API
- LangChain
"""
    }

    for filename, content in sample_docs.items():
        filepath = docs_path / filename
        with open(filepath, 'w') as f:
            f.write(content)

    logger.info(f"Created {len(sample_docs)} sample documents in {docs_dir}")


def load_documents(docs_dir):
    """
    Load documents from directory.

    Args:
        docs_dir: Directory containing documents

    Returns:
        List of loaded documents
    """
    docs_path = Path(docs_dir)

    if not docs_path.exists():
        logger.error(f"Directory not found: {docs_dir}")
        return []

    documents = []

    # Load text files
    txt_files = list(docs_path.glob("*.txt"))
    for txt_file in txt_files:
        try:
            loader = TextLoader(str(txt_file))
            docs = loader.load()
            documents.extend(docs)
            logger.info(f"Loaded: {txt_file.name}")
        except Exception as e:
            logger.error(f"Error loading {txt_file}: {e}")

    # Load PDF files (if any)
    pdf_files = list(docs_path.glob("*.pdf"))
    for pdf_file in pdf_files:
        try:
            loader = PyPDFLoader(str(pdf_file))
            docs = loader.load()
            documents.extend(docs)
            logger.info(f"Loaded: {pdf_file.name}")
        except Exception as e:
            logger.error(f"Error loading {pdf_file}: {e}")

    logger.info(f"\nTotal documents loaded: {len(documents)}")

    return documents


def build_rag_system(documents, embeddings, llm):
    """
    Build RAG system from documents.

    Args:
        documents: List of documents
        embeddings: Embeddings model
        llm: Language model

    Returns:
        RetrievalQA chain
    """
    if not documents:
        logger.error("No documents to process")
        return None

    logger.info("Building RAG system...")

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""]
    )
    splits = text_splitter.split_documents(documents)

    logger.info(f"Created {len(splits)} chunks from {len(documents)} documents")

    # Create vector store
    vectorstore = FAISS.from_documents(splits, embeddings)

    # Create retriever
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    # Create QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

    logger.info("RAG system ready!\n")

    return qa_chain


def query_documents(qa_chain, question):
    """Query the RAG system."""
    logger.info(f"Query: {question}")

    result = qa_chain.invoke({"query": question})

    return {
        "answer": result["result"],
        "sources": [
            {
                "content": doc.page_content,
                "source": doc.metadata.get("source", "Unknown")
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

    # Create sample docs if directory doesn't exist
    if not Path(args.docs_dir).exists():
        logger.info(f"Creating sample documents in {args.docs_dir}...")
        create_sample_documents(args.docs_dir)
        logger.info("\nSample documents created! You can now:")
        logger.info(f"  1. Add your own documents to {args.docs_dir}/")
        logger.info(f"  2. Run again to query the documents\n")

    # Load documents
    documents = load_documents(args.docs_dir)

    if not documents:
        logger.error("No documents found to process")
        return

    # Show document statistics
    print(f"\n{'='*80}")
    print("DOCUMENT STATISTICS")
    print(f"{'='*80}")
    for doc in documents:
        source = Path(doc.metadata.get("source", "Unknown")).name
        length = len(doc.page_content)
        print(f"  {source}: {length} characters")
    print(f"{'='*80}\n")

    # Initialize models
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Build RAG system
    qa_chain = build_rag_system(documents, embeddings, llm)

    if qa_chain is None:
        return

    # Query mode
    if args.query:
        result = query_documents(qa_chain, args.query)

        print(f"\n{'='*80}")
        print(f"QUESTION: {args.query}")
        print(f"\n{'='*80}")
        print(f"ANSWER:\n{result['answer']}")
        print(f"\n{'='*80}")
        print("SOURCES:")
        for i, source in enumerate(result['sources'], 1):
            print(f"\n{i}. File: {Path(source['source']).name}")
            print(f"   Content: {source['content'][:200]}...")
        print(f"{'='*80}\n")

    else:
        # Demo queries
        demo_questions = [
            "What are the types of machine learning?",
            "What is deep learning?",
            "What are common NLP tasks?",
            "What tools are mentioned for NLP?",
        ]

        for question in demo_questions:
            result = query_documents(qa_chain, question)

            print(f"\n{'='*80}")
            print(f"Q: {question}")
            print(f"A: {result['answer']}")
            print(f"{'='*80}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG with custom documents")

    parser.add_argument(
        "--docs-dir",
        type=str,
        default="./sample_documents",
        help="Directory containing documents"
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Question to ask"
    )

    args = parser.parse_args()
    main(args)
