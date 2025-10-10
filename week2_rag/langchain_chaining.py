"""
LangChain Utilities and Chaining Examples
Advanced LangChain patterns including sequential chains, parallel chains, and routing.

Usage:
    python langchain_chaining.py --demo sequential
    python langchain_chaining.py --demo parallel
    python langchain_chaining.py --demo routing
    python langchain_chaining.py --demo all
"""

import argparse
import os
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain, SequentialChain, SimpleSequentialChain
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# 1. SEQUENTIAL CHAINS - Steps executed one after another
# ============================================================================

def demo_sequential_chain():
    """
    Sequential Chain: Output of one chain feeds into the next.
    Use case: Multi-step transformations (e.g., translate -> summarize -> analyze)
    """
    logger.info("\n" + "="*80)
    logger.info("DEMO 1: SEQUENTIAL CHAIN")
    logger.info("="*80)

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

    # Step 1: Generate a story
    story_template = """You are a creative writer. Write a short 2-sentence story about: {topic}

    Story:"""
    story_prompt = PromptTemplate(input_variables=["topic"], template=story_template)
    story_chain = LLMChain(llm=llm, prompt=story_prompt, output_key="story")

    # Step 2: Translate to a different style
    style_template = """Rewrite the following story in a {style} style:

    Original Story: {story}

    Rewritten Story:"""
    style_prompt = PromptTemplate(input_variables=["story", "style"], template=style_template)
    style_chain = LLMChain(llm=llm, prompt=style_prompt, output_key="styled_story")

    # Step 3: Extract key themes
    theme_template = """What are the key themes in this story? List 2-3 themes.

    Story: {styled_story}

    Themes:"""
    theme_prompt = PromptTemplate(input_variables=["styled_story"], template=theme_template)
    theme_chain = LLMChain(llm=llm, prompt=theme_prompt, output_key="themes")

    # Combine into sequential chain
    sequential_chain = SequentialChain(
        chains=[story_chain, style_chain, theme_chain],
        input_variables=["topic", "style"],
        output_variables=["story", "styled_story", "themes"],
        verbose=True
    )

    # Execute
    result = sequential_chain.invoke({
        "topic": "a robot learning to paint",
        "style": "poetic"
    })

    print(f"\n{'='*80}")
    print(f"TOPIC: {result['topic']}")
    print(f"\n1. ORIGINAL STORY:\n{result['story']}")
    print(f"\n2. STYLED STORY ({result['style']}):\n{result['styled_story']}")
    print(f"\n3. THEMES:\n{result['themes']}")
    print(f"{'='*80}\n")

    return result


def demo_simple_sequential_chain():
    """
    Simple Sequential Chain: Simplified version where each chain has single input/output.
    """
    logger.info("\n" + "="*80)
    logger.info("DEMO 2: SIMPLE SEQUENTIAL CHAIN")
    logger.info("="*80)

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

    # Chain 1: Generate company name
    first_prompt = ChatPromptTemplate.from_template(
        "Suggest a creative name for a company that makes {product}"
    )
    chain_one = first_prompt | llm | StrOutputParser()

    # Chain 2: Generate tagline
    second_prompt = ChatPromptTemplate.from_template(
        "Write a catchy tagline for this company: {company_name}"
    )
    chain_two = second_prompt | llm | StrOutputParser()

    # Combine
    overall_chain = SimpleSequentialChain(
        chains=[chain_one, chain_two],
        verbose=True
    )

    result = overall_chain.invoke("eco-friendly water bottles")

    print(f"\n{'='*80}")
    print(f"PRODUCT: eco-friendly water bottles")
    print(f"RESULT: {result}")
    print(f"{'='*80}\n")

    return result


# ============================================================================
# 2. PARALLEL CHAINS - Execute multiple chains simultaneously
# ============================================================================

def demo_parallel_chain():
    """
    Parallel Chain: Execute multiple chains at once and combine results.
    Use case: Multiple analyses on same input (e.g., sentiment + entities + summary)
    """
    logger.info("\n" + "="*80)
    logger.info("DEMO 3: PARALLEL CHAIN (LCEL)")
    logger.info("="*80)

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Sample text to analyze
    text = """The new iPhone 15 Pro has been receiving mixed reviews.
    While users love the titanium design and improved camera, many are disappointed
    by the high price of $1199 and battery life concerns."""

    # Create parallel chains for different analyses
    sentiment_chain = (
        ChatPromptTemplate.from_template("Analyze the sentiment of this text (positive/negative/neutral): {text}")
        | llm
        | StrOutputParser()
    )

    entities_chain = (
        ChatPromptTemplate.from_template("Extract the main entities (products, companies, prices) from: {text}")
        | llm
        | StrOutputParser()
    )

    summary_chain = (
        ChatPromptTemplate.from_template("Summarize this text in one sentence: {text}")
        | llm
        | StrOutputParser()
    )

    # Combine in parallel
    parallel_chain = RunnableParallel(
        sentiment=sentiment_chain,
        entities=entities_chain,
        summary=summary_chain
    )

    # Execute all at once
    result = parallel_chain.invoke({"text": text})

    print(f"\n{'='*80}")
    print(f"INPUT TEXT:\n{text}")
    print(f"\n--- PARALLEL ANALYSIS ---")
    print(f"\nSENTIMENT:\n{result['sentiment']}")
    print(f"\nENTITIES:\n{result['entities']}")
    print(f"\nSUMMARY:\n{result['summary']}")
    print(f"{'='*80}\n")

    return result


# ============================================================================
# 3. ROUTING CHAINS - Conditional routing based on input
# ============================================================================

def demo_routing_chain():
    """
    Routing Chain: Route to different chains based on input characteristics.
    Use case: Different handling for different input types (e.g., technical vs. general questions)
    """
    logger.info("\n" + "="*80)
    logger.info("DEMO 4: ROUTING CHAIN")
    logger.info("="*80)

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

    # Router function
    def route_question(question):
        """Route question to appropriate expert."""
        question_lower = question.lower()

        if any(word in question_lower for word in ["code", "python", "function", "programming"]):
            return "technical"
        elif any(word in question_lower for word in ["recipe", "cook", "food", "ingredient"]):
            return "cooking"
        else:
            return "general"

    # Different expert chains
    technical_chain = (
        ChatPromptTemplate.from_template(
            "You are a senior software engineer. Answer this technical question: {question}"
        )
        | llm
        | StrOutputParser()
    )

    cooking_chain = (
        ChatPromptTemplate.from_template(
            "You are a professional chef. Answer this cooking question: {question}"
        )
        | llm
        | StrOutputParser()
    )

    general_chain = (
        ChatPromptTemplate.from_template(
            "You are a helpful assistant. Answer this question: {question}"
        )
        | llm
        | StrOutputParser()
    )

    # Router chain
    def route_and_execute(inputs):
        question = inputs["question"]
        route = route_question(question)

        logger.info(f"Routing to: {route.upper()} expert")

        if route == "technical":
            return technical_chain.invoke({"question": question})
        elif route == "cooking":
            return cooking_chain.invoke({"question": question})
        else:
            return general_chain.invoke({"question": question})

    router_chain = RunnableLambda(route_and_execute)

    # Test with different questions
    questions = [
        "How do I implement a binary search in Python?",
        "What's a good recipe for chocolate chip cookies?",
        "What is the capital of France?"
    ]

    for question in questions:
        result = router_chain.invoke({"question": question})

        print(f"\n{'='*80}")
        print(f"QUESTION: {question}")
        print(f"\nANSWER:\n{result}")
        print(f"{'='*80}\n")


# ============================================================================
# 4. LCEL (LangChain Expression Language) - Modern chaining
# ============================================================================

def demo_lcel_chain():
    """
    LCEL Chain: Modern LangChain syntax using | operator.
    More flexible and composable than legacy chains.
    """
    logger.info("\n" + "="*80)
    logger.info("DEMO 5: LCEL (LANGCHAIN EXPRESSION LANGUAGE)")
    logger.info("="*80)

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

    # Example 1: Simple chain
    prompt = ChatPromptTemplate.from_template("Tell me a joke about {topic}")
    chain = prompt | llm | StrOutputParser()

    result1 = chain.invoke({"topic": "programming"})
    print(f"\n{'='*80}")
    print(f"SIMPLE LCEL CHAIN:")
    print(f"JOKE: {result1}")
    print(f"{'='*80}\n")

    # Example 2: Chain with preprocessing
    def preprocess(inputs):
        """Uppercase the topic."""
        return {"topic": inputs["topic"].upper()}

    chain_with_preprocessing = (
        RunnableLambda(preprocess)
        | prompt
        | llm
        | StrOutputParser()
    )

    result2 = chain_with_preprocessing.invoke({"topic": "cats"})
    print(f"\n{'='*80}")
    print(f"LCEL WITH PREPROCESSING:")
    print(f"JOKE: {result2}")
    print(f"{'='*80}\n")

    # Example 3: Chain with passthrough
    prompt_with_context = ChatPromptTemplate.from_template(
        "Given this context: {context}\n\nAnswer: {question}"
    )

    chain_with_passthrough = (
        RunnablePassthrough.assign(
            context=lambda x: "You are a helpful AI assistant."
        )
        | prompt_with_context
        | llm
        | StrOutputParser()
    )

    result3 = chain_with_passthrough.invoke({"question": "What is 2+2?"})
    print(f"\n{'='*80}")
    print(f"LCEL WITH PASSTHROUGH:")
    print(f"ANSWER: {result3}")
    print(f"{'='*80}\n")


# ============================================================================
# 5. MAP-REDUCE CHAIN - Process multiple documents
# ============================================================================

def demo_map_reduce_chain():
    """
    Map-Reduce Chain: Process multiple documents individually, then combine.
    Use case: Summarizing multiple documents
    """
    logger.info("\n" + "="*80)
    logger.info("DEMO 6: MAP-REDUCE PATTERN")
    logger.info("="*80)

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Sample documents
    documents = [
        "Machine learning is a subset of AI that enables systems to learn from data.",
        "Deep learning uses neural networks with multiple layers to process information.",
        "Natural language processing helps computers understand human language."
    ]

    # Map: Summarize each document
    map_template = "Summarize this in 5 words: {doc}"
    map_prompt = ChatPromptTemplate.from_template(map_template)
    map_chain = map_prompt | llm | StrOutputParser()

    # Apply to all documents
    summaries = [map_chain.invoke({"doc": doc}) for doc in documents]

    # Reduce: Combine summaries
    reduce_template = "Combine these summaries into one sentence:\n{summaries}"
    reduce_prompt = ChatPromptTemplate.from_template(reduce_template)
    reduce_chain = reduce_prompt | llm | StrOutputParser()

    final_summary = reduce_chain.invoke({"summaries": "\n".join(summaries)})

    print(f"\n{'='*80}")
    print(f"ORIGINAL DOCUMENTS:")
    for i, doc in enumerate(documents, 1):
        print(f"{i}. {doc}")

    print(f"\nINDIVIDUAL SUMMARIES:")
    for i, summary in enumerate(summaries, 1):
        print(f"{i}. {summary}")

    print(f"\nFINAL COMBINED SUMMARY:")
    print(final_summary)
    print(f"{'='*80}\n")


# ============================================================================
# MAIN
# ============================================================================

def main(args):
    """Main function."""
    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY not found in environment variables")
        logger.error("Please set it: export OPENAI_API_KEY='your-key-here'")
        return

    logger.info("LangChain Chaining Examples")

    if args.demo == "sequential":
        demo_sequential_chain()
        demo_simple_sequential_chain()
    elif args.demo == "parallel":
        demo_parallel_chain()
    elif args.demo == "routing":
        demo_routing_chain()
    elif args.demo == "lcel":
        demo_lcel_chain()
    elif args.demo == "mapreduce":
        demo_map_reduce_chain()
    elif args.demo == "all":
        demo_sequential_chain()
        demo_simple_sequential_chain()
        demo_parallel_chain()
        demo_routing_chain()
        demo_lcel_chain()
        demo_map_reduce_chain()
    else:
        logger.error(f"Unknown demo: {args.demo}")
        logger.info("Available demos: sequential, parallel, routing, lcel, mapreduce, all")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LangChain chaining examples",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run sequential chain examples
    python langchain_chaining.py --demo sequential

    # Run parallel chain example
    python langchain_chaining.py --demo parallel

    # Run routing chain example
    python langchain_chaining.py --demo routing

    # Run all demos
    python langchain_chaining.py --demo all
        """
    )

    parser.add_argument(
        "--demo",
        type=str,
        default="all",
        choices=["sequential", "parallel", "routing", "lcel", "mapreduce", "all"],
        help="Which demo to run"
    )

    args = parser.parse_args()
    main(args)
