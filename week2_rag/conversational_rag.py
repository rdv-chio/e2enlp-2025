"""
Conversational RAG
RAG system with conversation memory.

Usage:
    python conversational_rag.py --interactive
"""

import argparse
import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_core.documents import Document
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


KNOWLEDGE_BASE = [
    """The Eiffel Tower is a wrought-iron lattice tower located in Paris, France.
    It was completed in 1889 and stands 330 meters (1,083 feet) tall.
    It was designed by Gustave Eiffel for the 1889 World's Fair.""",

    """The Eiffel Tower was initially criticized by some of France's leading artists and intellectuals
    but has become a global cultural icon of France and one of the most recognizable structures in the world.
    About 7 million people visit the tower annually.""",

    """The tower has three levels accessible to visitors. The first and second levels have restaurants.
    The top level's observation deck is 276 meters above the ground.
    Visitors can reach the first and second levels by stairs or elevator.""",

    """Paris is the capital and largest city of France. It has an estimated population of 2.1 million people.
    Paris is known for its museums, architecture, and cafes. Major landmarks include the Louvre Museum,
    Notre-Dame Cathedral, and Arc de Triomphe.""",

    """The Louvre Museum is the world's largest art museum and historic monument in Paris.
    It houses approximately 38,000 objects from prehistory to the 21st century.
    The Mona Lisa is the museum's most famous painting.""",
]


def setup_conversational_rag():
    """
    Setup conversational RAG system.

    Returns:
        Tuple of (chain, memory)
    """
    logger.info("Setting up conversational RAG system...")

    # Initialize embeddings and LLM
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

    # Create documents
    docs = [Document(page_content=text) for text in KNOWLEDGE_BASE]

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    splits = text_splitter.split_documents(docs)

    # Create vector store
    vectorstore = FAISS.from_documents(splits, embeddings)

    # Create memory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

    # Create conversational chain
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        memory=memory,
        return_source_documents=True,
        verbose=True
    )

    logger.info("Conversational RAG ready!\n")

    return chain, memory


def chat_with_rag(chain, question):
    """
    Chat with the RAG system.

    Args:
        chain: Conversational RAG chain
        question: Question to ask

    Returns:
        Dictionary with answer and sources
    """
    result = chain.invoke({"question": question})

    return {
        "answer": result["answer"],
        "sources": [doc.page_content for doc in result["source_documents"]]
    }


def show_conversation_history(memory):
    """Display conversation history."""
    messages = memory.chat_memory.messages

    if not messages:
        print("\nNo conversation history yet.\n")
        return

    print(f"\n{'='*80}")
    print("CONVERSATION HISTORY")
    print(f"{'='*80}")

    for i, msg in enumerate(messages):
        role = "You" if msg.type == "human" else "Assistant"
        print(f"\n{role}: {msg.content}")

    print(f"\n{'='*80}\n")


def interactive_mode(chain, memory):
    """Interactive conversation mode."""
    print("\n" + "="*80)
    print("Conversational RAG - Interactive Mode")
    print("="*80)
    print("\nCommands:")
    print("  Type your question to chat")
    print("  'history' - Show conversation history")
    print("  'clear' - Clear conversation history")
    print("  'quit' - Exit\n")
    print("="*80 + "\n")

    while True:
        try:
            question = input("\n🤔 You: ").strip()

            if not question:
                continue

            if question.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!\n")
                break

            if question.lower() == 'history':
                show_conversation_history(memory)
                continue

            if question.lower() == 'clear':
                memory.clear()
                print("\n✨ Conversation history cleared!\n")
                continue

            # Get response
            result = chat_with_rag(chain, question)

            print(f"\n🤖 Assistant: {result['answer']}")

            # Show sources (optional, can be toggled)
            if len(result['sources']) > 0:
                print(f"\n📚 Sources used: {len(result['sources'])}")

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!\n")
            break
        except Exception as e:
            logger.error(f"Error: {e}")


def demo_conversation(chain):
    """Run a demo conversation."""
    print("\n" + "="*80)
    print("Demo Conversation")
    print("="*80 + "\n")

    conversation = [
        "Where is the Eiffel Tower?",
        "How tall is it?",
        "When was it built?",
        "What can visitors do there?",
        "What other landmarks are in that city?",
    ]

    for question in conversation:
        result = chat_with_rag(chain, question)

        print(f"\n🤔 You: {question}")
        print(f"🤖 Assistant: {result['answer']}")
        print("-" * 80)

    print("\n" + "="*80 + "\n")


def main(args):
    """Main function."""
    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY not found")
        return

    # Setup RAG system
    chain, memory = setup_conversational_rag()

    if args.interactive:
        interactive_mode(chain, memory)
    else:
        # Run demo
        demo_conversation(chain)

        # Show history
        print("\nConversation completed! Here's the history:\n")
        show_conversation_history(memory)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Conversational RAG")

    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Interactive chat mode"
    )

    args = parser.parse_args()
    main(args)
