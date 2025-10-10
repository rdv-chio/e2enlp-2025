"""
Local RAG with Ollama Models
Run RAG systems using local open-source models via Ollama.
Supports models like Qwen, Gemma, Llama, Mistral, and more.

Prerequisites:
    1. Install Ollama: https://ollama.ai/
    2. Pull models: ollama pull qwen2.5:latest
                   ollama pull gemma2:2b
                   ollama pull llama3.2:latest

Usage:
    # List available models
    python ollama_local_rag.py --list-models

    # Use with Qwen
    python ollama_local_rag.py --model qwen2.5:latest

    # Use with Gemma
    python ollama_local_rag.py --model gemma2:2b

    # Interactive mode
    python ollama_local_rag.py --model qwen2.5:latest --interactive

    # Test embeddings
    python ollama_local_rag.py --test-embeddings
"""

import argparse
import subprocess
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_core.documents import Document
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
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
    to process sequences in parallel. Models like BERT and GPT are based on transformer architecture.
    The attention mechanism allows models to focus on relevant parts of the input.""",

    """Fine-tuning is the process of taking a pretrained model and training it further on a
    specific task or domain. This allows the model to adapt its knowledge to specialized use cases
    with relatively little additional data. LoRA and QLoRA are popular fine-tuning techniques.""",

    """Retrieval Augmented Generation (RAG) combines retrieval with language model generation.
    It retrieves relevant documents from a knowledge base and uses them to ground the model's
    responses in factual information, reducing hallucinations.""",

    """Vector databases store high-dimensional embeddings and enable efficient similarity search.
    They are essential for RAG systems. Popular options include Pinecone, Weaviate, Chroma, and FAISS.
    They use approximate nearest neighbor (ANN) algorithms for fast retrieval.""",

    """Open-source language models like Llama, Mistral, Qwen, and Gemma can run locally.
    This provides privacy, cost savings, and no API rate limits. Ollama makes it easy to run
    these models on your own hardware.""",
]


# ============================================================================
# OLLAMA UTILITIES
# ============================================================================

def check_ollama_installed():
    """Check if Ollama is installed."""
    try:
        result = subprocess.run(
            ["ollama", "--version"],
            capture_output=True,
            text=True,
            check=True
        )
        logger.info(f"Ollama version: {result.stdout.strip()}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def list_ollama_models():
    """List available Ollama models."""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        logger.error(f"Error listing models: {e}")
        return None


def check_model_available(model_name):
    """Check if a specific model is available."""
    models_output = list_ollama_models()
    if models_output:
        return model_name.split(':')[0] in models_output
    return False


def pull_model(model_name):
    """Pull an Ollama model."""
    logger.info(f"Pulling model {model_name}...")
    try:
        subprocess.run(
            ["ollama", "pull", model_name],
            check=True
        )
        logger.info(f"Successfully pulled {model_name}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error pulling model: {e}")
        return False


# ============================================================================
# RAG SYSTEM
# ============================================================================

def create_vector_store_local(documents, embedding_model="nomic-embed-text"):
    """
    Create a vector store using local Ollama embeddings.

    Args:
        documents: List of text documents
        embedding_model: Ollama embedding model (default: nomic-embed-text)

    Returns:
        FAISS vector store
    """
    logger.info(f"Creating vector store with {embedding_model}...")

    # Check if embedding model is available
    if not check_model_available(embedding_model):
        logger.warning(f"{embedding_model} not found. Pulling...")
        pull_model(embedding_model)

    # Create embeddings
    embeddings = OllamaEmbeddings(model=embedding_model)

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


def create_local_rag_chain(vectorstore, model_name="qwen2.5:latest", streaming=False):
    """
    Create a RAG QA chain with local Ollama model.

    Args:
        vectorstore: Vector store for retrieval
        model_name: Ollama model name
        streaming: Whether to stream responses

    Returns:
        RetrievalQA chain
    """
    logger.info(f"Creating RAG chain with {model_name}...")

    # Check if model is available
    if not check_model_available(model_name):
        logger.warning(f"{model_name} not found. Pulling...")
        pull_model(model_name)

    # Create LLM
    if streaming:
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        llm = Ollama(
            model=model_name,
            callback_manager=callback_manager,
            temperature=0.1
        )
    else:
        llm = Ollama(model=model_name, temperature=0.1)

    # Create retriever
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

    # Create QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        verbose=True
    )

    return qa_chain


def query_local_rag(qa_chain, question):
    """
    Query the local RAG system.

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


# ============================================================================
# TESTING & DEMOS
# ============================================================================

def test_embeddings():
    """Test local embeddings."""
    logger.info("\n" + "="*80)
    logger.info("TESTING LOCAL EMBEDDINGS")
    logger.info("="*80)

    embedding_model = "nomic-embed-text"

    if not check_model_available(embedding_model):
        logger.info(f"Pulling {embedding_model}...")
        pull_model(embedding_model)

    embeddings = OllamaEmbeddings(model=embedding_model)

    # Test embedding generation
    texts = [
        "Machine learning is a subset of AI",
        "Deep learning uses neural networks",
        "How do I bake a cake?"
    ]

    logger.info(f"\nGenerating embeddings for {len(texts)} texts...")

    vectors = embeddings.embed_documents(texts)

    print(f"\nEmbedding dimension: {len(vectors[0])}")
    print(f"\nFirst 10 values of first embedding: {vectors[0][:10]}")

    # Compute similarity
    from numpy import dot
    from numpy.linalg import norm

    def cosine_similarity(a, b):
        return dot(a, b) / (norm(a) * norm(b))

    sim_1_2 = cosine_similarity(vectors[0], vectors[1])
    sim_1_3 = cosine_similarity(vectors[0], vectors[2])

    print(f"\nSimilarity between text 1 and 2 (related): {sim_1_2:.4f}")
    print(f"Similarity between text 1 and 3 (unrelated): {sim_1_3:.4f}")


def demo_model_comparison():
    """Compare different Ollama models."""
    logger.info("\n" + "="*80)
    logger.info("MODEL COMPARISON")
    logger.info("="*80)

    models = [
        "qwen2.5:latest",
        "gemma2:2b",
        "llama3.2:latest"
    ]

    available_models = [m for m in models if check_model_available(m)]

    if not available_models:
        logger.warning("No models available. Please pull at least one model:")
        for model in models:
            logger.info(f"  ollama pull {model}")
        return

    logger.info(f"Comparing models: {available_models}")

    # Create vector store once
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    docs = [Document(page_content=doc) for doc in SAMPLE_DOCUMENTS]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(docs)
    vectorstore = FAISS.from_documents(splits, embeddings)

    question = "What is RAG and why is it useful?"

    for model in available_models:
        print(f"\n{'='*80}")
        print(f"MODEL: {model}")
        print(f"{'='*80}\n")

        try:
            qa_chain = create_local_rag_chain(vectorstore, model)
            result = query_local_rag(qa_chain, question)

            print(f"\nANSWER:\n{result['answer']}")
            print(f"\n{'-'*80}\n")

        except Exception as e:
            logger.error(f"Error with {model}: {e}")


# ============================================================================
# MAIN
# ============================================================================

def main(args):
    """Main function."""
    logger.info("Local RAG with Ollama")

    # Check if Ollama is installed
    if not check_ollama_installed():
        logger.error("\n" + "="*80)
        logger.error("ERROR: Ollama is not installed!")
        logger.error("="*80)
        logger.error("\nPlease install Ollama:")
        logger.error("  macOS/Linux: curl https://ollama.ai/install.sh | sh")
        logger.error("  Windows: Download from https://ollama.ai/download")
        logger.error("\nThen pull a model:")
        logger.error("  ollama pull qwen2.5:latest")
        logger.error("  ollama pull gemma2:2b")
        logger.error("="*80 + "\n")
        return

    # List models
    if args.list_models:
        logger.info("\n" + "="*80)
        logger.info("AVAILABLE OLLAMA MODELS")
        logger.info("="*80 + "\n")
        models_output = list_ollama_models()
        if models_output:
            print(models_output)
        else:
            logger.warning("No models found. Pull a model with:")
            logger.info("  ollama pull qwen2.5:latest")
        return

    # Test embeddings
    if args.test_embeddings:
        test_embeddings()
        return

    # Compare models
    if args.compare_models:
        demo_model_comparison()
        return

    # Regular RAG
    model_name = args.model

    # Check model availability
    if not check_model_available(model_name):
        logger.warning(f"\nModel {model_name} not found locally.")
        response = input(f"Would you like to pull it? (y/n): ")
        if response.lower() == 'y':
            if not pull_model(model_name):
                logger.error("Failed to pull model. Exiting.")
                return
        else:
            logger.info("\nAvailable models:")
            print(list_ollama_models())
            return

    # Create RAG system
    logger.info(f"\nInitializing local RAG with {model_name}...")

    # Create vector store
    vectorstore = create_vector_store_local(SAMPLE_DOCUMENTS)

    # Create RAG chain
    qa_chain = create_local_rag_chain(vectorstore, model_name, streaming=args.stream)

    logger.info("Local RAG system ready!\n")

    # Interactive mode or single query
    if args.interactive:
        logger.info(f"Interactive mode with {model_name}. Type 'quit' to exit.\n")
        logger.info("NOTE: First response may be slower as model loads.\n")

        while True:
            question = input("\nYour question: ").strip()

            if question.lower() in ['quit', 'exit', 'q']:
                logger.info("Goodbye!")
                break

            if not question:
                continue

            try:
                result = query_local_rag(qa_chain, question)

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
        # Single query or demo mode
        if args.query:
            result = query_local_rag(qa_chain, args.query)

            print(f"\n{'='*80}")
            print(f"MODEL: {model_name}")
            print(f"QUESTION: {args.query}")
            print(f"\n{'='*80}")
            print(f"ANSWER:\n{result['answer']}")
            print(f"\n{'='*80}")
            print("SOURCES:")
            for i, source in enumerate(result['sources'], 1):
                print(f"\n{i}. {source[:200]}...")
            print(f"{'='*80}\n")

        else:
            # Demo mode
            demo_questions = [
                "What is machine learning?",
                "How does RAG work?",
                "What are the benefits of local models?",
            ]

            for question in demo_questions:
                result = query_local_rag(qa_chain, question)

                print(f"\n{'='*80}")
                print(f"Q: {question}")
                print(f"A: {result['answer']}")
                print(f"{'='*80}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Local RAG with Ollama models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # List available models
    python ollama_local_rag.py --list-models

    # Test embeddings
    python ollama_local_rag.py --test-embeddings

    # Use Qwen model
    python ollama_local_rag.py --model qwen2.5:latest

    # Use Gemma model
    python ollama_local_rag.py --model gemma2:2b

    # Interactive mode
    python ollama_local_rag.py --model qwen2.5:latest --interactive

    # With streaming
    python ollama_local_rag.py --model qwen2.5:latest --stream

    # Compare models
    python ollama_local_rag.py --compare-models

Popular models:
    - qwen2.5:latest (Alibaba - excellent performance)
    - gemma2:2b (Google - lightweight, fast)
    - llama3.2:latest (Meta - well-rounded)
    - mistral:latest (Mistral AI - strong reasoning)
    - phi3:latest (Microsoft - efficient)
        """
    )

    parser.add_argument(
        "--model",
        type=str,
        default="qwen2.5:latest",
        help="Ollama model to use"
    )
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
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream responses"
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available Ollama models"
    )
    parser.add_argument(
        "--test-embeddings",
        action="store_true",
        help="Test local embeddings"
    )
    parser.add_argument(
        "--compare-models",
        action="store_true",
        help="Compare different models"
    )

    args = parser.parse_args()
    main(args)
