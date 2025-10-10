# Week 2: Prompt Engineering & RAG

Learn advanced prompting techniques and build Retrieval Augmented Generation systems.

## 📚 Materials

### Jupyter Notebook (Learning)
- `prompt_engineering_and_rag.ipynb` - Interactive learning notebook

### Python Scripts (Practice)
Multiple RAG implementations to explore:

1. **`basic_rag.py`** - Simple RAG with FAISS
2. **`rag_with_chroma.py`** - Persistent storage with ChromaDB
3. **`conversational_rag.py`** - RAG with conversation memory
4. **`rag_with_custom_docs.py`** - Load your own documents
5. **`langchain_chaining.py`** - LangChain utilities and chaining patterns (NEW)
6. **`retrieval_strategies.py`** - Advanced retrieval strategies (NEW)
7. **`ollama_local_rag.py`** - Local models with Ollama (NEW)

## 🚀 Quick Start

### 1. Basic RAG (FAISS)

```bash
# Demo mode
python basic_rag.py

# Single query
python basic_rag.py --query "What is machine learning?"

# Interactive mode
python basic_rag.py --interactive
```

**Features**:
- Simple in-memory vector store
- Fast setup
- Good for learning

---

### 2. RAG with ChromaDB (Persistent)

```bash
# First, add documents to database
python rag_with_chroma.py --add-docs

# Then query
python rag_with_chroma.py --query "What is NLP?"

# Query with metadata filter
python rag_with_chroma.py --query "Tell me about AI" --filter "topic=ai"
```

**Features**:
- Persistent storage (survives restarts)
- Metadata filtering
- Good for production

**Filters**:
```bash
--filter "topic=ai"              # AI topics only
--filter "subtopic=nlp"          # NLP subtopic only
--filter "language=python"       # Python-related only
```

---

### 3. Conversational RAG (Memory)

```bash
# Demo conversation
python conversational_rag.py

# Interactive mode
python conversational_rag.py --interactive
```

**Interactive Commands**:
- Type your question to chat
- `history` - Show conversation history
- `clear` - Clear conversation history
- `quit` - Exit

**Features**:
- Remembers context
- Follow-up questions work naturally
- Conversation history

---

### 4. RAG with Custom Documents

```bash
# Creates sample docs and queries them
python rag_with_custom_docs.py

# Use your own documents directory
python rag_with_custom_docs.py --docs-dir ./my_documents

# Query specific question
python rag_with_custom_docs.py --docs-dir ./my_documents --query "Your question"
```

**Supported Formats**:
- `.txt` files
- `.pdf` files (coming soon)
- More formats can be added

**Features**:
- Load from file system
- Supports multiple file types
- Automatic chunking
- Source tracking

---

### 5. LangChain Chaining (Advanced Patterns)

```bash
# Run all chaining examples
python langchain_chaining.py --demo all

# Sequential chains (multi-step)
python langchain_chaining.py --demo sequential

# Parallel chains (multiple analyses)
python langchain_chaining.py --demo parallel

# Routing chains (conditional logic)
python langchain_chaining.py --demo routing

# LCEL (modern syntax)
python langchain_chaining.py --demo lcel

# Map-reduce pattern
python langchain_chaining.py --demo mapreduce
```

**Features**:
- Sequential chains: Multi-step transformations
- Parallel chains: Multiple analyses simultaneously
- Routing chains: Conditional logic based on input
- LCEL (LangChain Expression Language): Modern composable chains
- Map-reduce: Process multiple documents

**Use cases**:
- Complex workflows (translate → summarize → analyze)
- Multi-aspect analysis (sentiment + entities + summary)
- Smart routing to specialized models
- Document batch processing

---

### 6. Advanced Retrieval Strategies

```bash
# Compare all strategies
python retrieval_strategies.py --strategy compare

# Similarity search (standard)
python retrieval_strategies.py --strategy similarity

# MMR (diverse results)
python retrieval_strategies.py --strategy mmr

# Hybrid search (dense + sparse)
python retrieval_strategies.py --strategy hybrid

# Contextual compression
python retrieval_strategies.py --strategy compression

# Multi-query retrieval
python retrieval_strategies.py --strategy multiquery

# Run all demos
python retrieval_strategies.py --strategy all
```

**Strategies**:
1. **Similarity Search**: Standard cosine similarity
2. **MMR**: Balance relevance and diversity
3. **Threshold**: Filter by confidence score
4. **Hybrid**: Combine dense (embeddings) + sparse (BM25)
5. **Contextual Compression**: Extract only relevant parts
6. **Multi-Query**: Generate query variations
7. **Parent Document**: Search chunks, return full docs

**Best practices**:
- Use **similarity** for general queries
- Use **MMR** to avoid redundant results
- Use **hybrid** for production systems
- Use **compression** for long documents
- Use **threshold** for high-precision needs

---

### 7. Local Models with Ollama

```bash
# First, install Ollama: https://ollama.ai/
# Then pull models:
ollama pull qwen2.5:latest
ollama pull gemma2:2b
ollama pull llama3.2:latest

# List available models
python ollama_local_rag.py --list-models

# Test embeddings
python ollama_local_rag.py --test-embeddings

# Use with Qwen
python ollama_local_rag.py --model qwen2.5:latest

# Use with Gemma
python ollama_local_rag.py --model gemma2:2b

# Interactive mode
python ollama_local_rag.py --model qwen2.5:latest --interactive

# With streaming
python ollama_local_rag.py --model qwen2.5:latest --stream

# Compare models
python ollama_local_rag.py --compare-models
```

**Popular Models**:
- **Qwen 2.5** (Alibaba): Excellent performance, multilingual
- **Gemma 2** (Google): Lightweight, fast, 2B/7B/27B variants
- **Llama 3.2** (Meta): Well-rounded, strong reasoning
- **Mistral** (Mistral AI): Efficient, good at coding
- **Phi 3** (Microsoft): Small but capable

**Benefits of Local Models**:
- Privacy: Data stays on your machine
- No API costs: Run unlimited queries
- No rate limits: Scale freely
- Offline capable: No internet needed
- Customizable: Full control over model

**Embedding Models**:
- **nomic-embed-text**: General purpose, 768 dimensions
- **mxbai-embed-large**: High quality, 1024 dimensions
- **all-minilm**: Fast, lightweight

---

## 📊 Comparison

### Core RAG Implementations

| Feature | Basic RAG | ChromaDB | Conversational | Custom Docs |
|---------|-----------|----------|----------------|-------------|
| Persistence | ❌ | ✅ | ❌ | ❌ |
| Memory | ❌ | ❌ | ✅ | ❌ |
| Metadata | ❌ | ✅ | ❌ | ✅ |
| File Loading | ❌ | ❌ | ❌ | ✅ |
| Best For | Learning | Production | Chatbots | Documents |

### Advanced Features

| Feature | Chaining | Retrieval Strategies | Ollama Local |
|---------|----------|---------------------|--------------|
| Focus | Workflow patterns | Retrieval optimization | Local models |
| API Required | ✅ | ✅ | ❌ |
| Learning Curve | Medium | Medium | Easy |
| Best For | Complex workflows | Production RAG | Privacy/Cost savings |

## 💡 Usage Examples

### Example 1: Quick Learning
```bash
# Start with basic RAG
python basic_rag.py --interactive

# Ask questions about ML/AI
> What is deep learning?
> How does it differ from machine learning?
> quit
```

### Example 2: Build Knowledge Base
```bash
# Add your documents to ChromaDB
python rag_with_chroma.py --add-docs

# Query with filtering
python rag_with_chroma.py --query "Explain transformers" --filter "topic=ai"
```

### Example 3: Conversational Assistant
```bash
# Interactive conversation
python conversational_rag.py --interactive

> Where is the Eiffel Tower?
> How tall is it?
> What can I do there?
> history  # See all questions
> quit
```

### Example 4: Document Q&A
```bash
# Create or use your document folder
mkdir my_docs
# Add .txt or .pdf files

# Query your documents
python rag_with_custom_docs.py --docs-dir ./my_docs --query "Summarize key points"
```

### Example 5: Advanced Chaining
```bash
# Multi-step workflow: generate → transform → analyze
python langchain_chaining.py --demo sequential

# Parallel analysis: sentiment + entities + summary
python langchain_chaining.py --demo parallel

# Smart routing to specialized models
python langchain_chaining.py --demo routing
```

### Example 6: Optimize Retrieval
```bash
# Compare retrieval strategies
python retrieval_strategies.py --strategy compare

# Use hybrid search for best results
python retrieval_strategies.py --strategy hybrid

# Compress results to reduce noise
python retrieval_strategies.py --strategy compression
```

### Example 7: Run RAG Locally
```bash
# Install Ollama and pull a model
ollama pull qwen2.5:latest

# Run local RAG (no API key needed!)
python ollama_local_rag.py --model qwen2.5:latest --interactive

# Compare different local models
python ollama_local_rag.py --compare-models
```

## 🛠️ How Each Works

### Basic RAG
1. Loads documents into memory
2. Creates FAISS vector store
3. Retrieves relevant chunks
4. Generates answer with LLM

### ChromaDB RAG
1. Persists vectors to disk
2. Loads database on startup
3. Filters by metadata
4. Queries and responds

### Conversational RAG
1. Maintains conversation memory
2. Understands context from history
3. Handles follow-up questions
4. Shows conversation history

### Custom Docs RAG
1. Loads files from directory
2. Chunks documents
3. Builds vector store
4. Tracks sources

## 📝 Assignment 1: Build Your RAG System

Choose a use case and build a RAG system:

### Ideas:
- **Company Knowledge Base**: Internal docs Q&A
- **Course Assistant**: Answer questions from course materials
- **Research Helper**: Query research papers
- **Customer Support**: Product documentation Q&A
- **Code Helper**: Search codebase with NL queries

### Requirements:
1. **Data**: At least 5 documents (can use provided samples or your own)
2. **Implementation**: Use one of the provided scripts as starting point
3. **Customization**:
   - Adjust chunk size/overlap
   - Try different embedding models
   - Experiment with retrieval parameters
4. **Testing**: Create 10+ test questions
5. **Evaluation**: Measure answer quality

### Deliverables:
- Working Python script
- Documents/knowledge base
- Test questions and answers
- Evaluation report
- README with instructions

### Evaluation Criteria:
- Retrieval quality (relevant chunks?)
- Answer accuracy (correct answers?)
- Source attribution (shows sources?)
- Code quality (clean, documented?)
- Documentation (clear instructions?)

## 🔧 Customization Tips

### Adjust Chunk Size
```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,      # Increase for more context
    chunk_overlap=50     # Increase for better continuity
)
```

### Change Retrieval Count
```python
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 5}  # Retrieve more documents
)
```

### Use Different Models
```python
# Different embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# Different LLM
llm = ChatOpenAI(model="gpt-4o", temperature=0.3)
```

### Add Metadata
```python
doc = Document(
    page_content="text",
    metadata={
        "source": "file.txt",
        "date": "2025-01-01",
        "category": "technical"
    }
)
```

## 🐛 Troubleshooting

**API Key Issues**:
```bash
export OPENAI_API_KEY='your-key-here'
# or create .env file
```

**ChromaDB Already Exists**:
```bash
# Remove old database
rm -rf chroma_db/
# Create new one
python rag_with_chroma.py --add-docs
```

**Memory Errors**:
- Reduce chunk size
- Retrieve fewer documents (lower k)
- Use smaller embedding model

**Poor Results**:
- Increase chunk overlap
- Retrieve more documents
- Improve document quality
- Try different prompt templates

## 📖 Additional Resources

- [LangChain RAG Guide](https://python.langchain.com/docs/use_cases/question_answering/)
- [ChromaDB Docs](https://docs.trychroma.com/)
- [FAISS Documentation](https://faiss.ai/)
- [OpenAI Embeddings](https://platform.openai.com/docs/guides/embeddings)

## 🎯 Learning Outcomes

After this week, you will:
- ✅ Understand RAG architecture
- ✅ Build RAG systems from scratch
- ✅ Use vector databases
- ✅ Implement conversational memory
- ✅ Load and process documents
- ✅ Evaluate retrieval quality
- ✅ Deploy RAG in production

---

**Next**: Complete the notebook, try all RAG scripts, and start Assignment 1!
